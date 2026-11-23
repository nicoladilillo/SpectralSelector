"""Persistent, content-addressed storage for expensive benchmark jobs."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ATTEMPT_SCHEMA_VERSION = 1


class CachedBenchmarkFailure(RuntimeError):
    """Raised when an identical job has a saved failure and rerun is disabled."""


class BenchmarkAlreadyRunning(RuntimeError):
    """Raised when another process owns the same benchmark job."""


class BenchmarkSchemaError(ValueError):
    """Raised when a saved attempt needs migration or is structurally invalid."""


@dataclass(frozen=True)
class BenchmarkOutcome:
    """Result and provenance returned by :class:`BenchmarkResultStore`."""

    run_key: str
    attempt: int
    cached: bool
    result: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _qualified_name(value: Any) -> str:
    cls = value if isinstance(value, type) else value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _array_content_digest(array: np.ndarray) -> str:
    """Hash array values without depending on object memory addresses."""
    if array.dtype.hasobject:
        payload = json.dumps(
            _manifest_value(array.tolist()),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    else:
        payload = np.ascontiguousarray(array).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _manifest_value(value: Any) -> Any:
    """Convert configuration into deterministic, JSON-compatible metadata."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (np.bool_, np.integer)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isfinite(number):
            return number
        return {"special_float": str(number)}
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "array": {
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "sha256": _array_content_digest(contiguous),
            }
        }
    if isinstance(value, Mapping):
        return {
            str(key): _manifest_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_manifest_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_manifest_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if hasattr(value, "get_params"):
        return {
            "object": _qualified_name(value),
            "parameters": _manifest_value(value.get_params(deep=True)),
        }
    if callable(value):
        module = getattr(value, "__module__", value.__class__.__module__)
        name = getattr(value, "__qualname__", value.__class__.__qualname__)
        return {"callable": f"{module}.{name}"}
    return {"object": _qualified_name(value), "representation": repr(value)}


def _result_value(value: Any) -> Any:
    """Convert result payloads to readable, strict JSON values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (np.bool_, np.integer)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Benchmark result values must be finite.")
        return number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _result_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _result_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_result_value(item) for item in value]
    raise TypeError(f"Unsupported benchmark result value: {_qualified_name(value)}")


def run_key(manifest: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return a stable SHA-256 key and normalized manifest."""
    normalized = _manifest_value(manifest)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), normalized


def arrays_fingerprint(*arrays: np.ndarray) -> str:
    """Hash array dtype, shape, and values in argument order."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(_array_content_digest(array).encode())
    return digest.hexdigest()


def source_tree_fingerprint(root: Path, patterns: tuple[str, ...]) -> str:
    """Hash source files so uncommitted code changes invalidate cached jobs."""
    digest = hashlib.sha256()
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in root.glob(pattern)
            if path.is_file() and "__pycache__" not in path.parts
        }
    )
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_attempt_record(record: Mapping[str, Any]) -> None:
    """Reject unsupported or incomplete persistent-attempt schemas explicitly."""
    version = record.get("schema_version")
    if version != ATTEMPT_SCHEMA_VERSION:
        raise BenchmarkSchemaError(
            "Unsupported benchmark attempt schema "
            f"{version!r}; this installation reads schema "
            f"{ATTEMPT_SCHEMA_VERSION}. Migrate or regenerate the cache."
        )
    required = {
        "run_key",
        "attempt",
        "status",
        "started_at_utc",
        "finished_at_utc",
        "duration_seconds",
        "manifest",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise BenchmarkSchemaError(
            f"Benchmark attempt is missing required fields: {missing}."
        )
    if record["status"] not in {"success", "failure"}:
        raise BenchmarkSchemaError(
            f"Unknown benchmark attempt status: {record['status']!r}."
        )
    payload = "result" if record["status"] == "success" else "error"
    if payload not in record:
        raise BenchmarkSchemaError(
            f"Benchmark {record['status']} attempt is missing {payload!r}."
        )


class BenchmarkResultStore:
    """Run, cache, and enumerate benchmark jobs using immutable attempts.

    Every unique normalized manifest maps to one directory. Each execution adds
    a numbered attempt instead of overwriting prior evidence. A successful
    attempt is reused by default; a failed attempt is only rerun explicitly.
    """

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        manifest: Mapping[str, Any],
        work: Callable[[], Mapping[str, Any]],
        *,
        force: bool = False,
        rerun_failed: bool = False,
    ) -> BenchmarkOutcome:
        """Return a cached success or execute and append a new attempt."""
        key, normalized_manifest = run_key(manifest)
        run_directory = self.root / key
        run_directory.mkdir(parents=True, exist_ok=True)

        with self._claim(run_directory):
            latest = self._latest_attempt(run_directory)
            if latest is not None and not force:
                if latest["status"] == "success":
                    return BenchmarkOutcome(
                        run_key=key,
                        attempt=int(latest["attempt"]),
                        cached=True,
                        result=dict(latest["result"]),
                    )
                if not rerun_failed:
                    error = latest.get("error", {})
                    message = error.get("message", "unknown cached failure")
                    raise CachedBenchmarkFailure(
                        f"Benchmark {key} previously failed: {message}. "
                        "Use rerun_failed=True or force=True to try again."
                    )

            attempt = 1 if latest is None else int(latest["attempt"]) + 1
            started_at = _utc_now()
            started = perf_counter()
            try:
                raw_result = work()
                if not isinstance(raw_result, Mapping):
                    raise TypeError("Benchmark work must return a mapping.")
                result = _result_value(raw_result)
            except Exception as error:
                record = {
                    "schema_version": ATTEMPT_SCHEMA_VERSION,
                    "run_key": key,
                    "attempt": attempt,
                    "status": "failure",
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc_now(),
                    "duration_seconds": perf_counter() - started,
                    "manifest": normalized_manifest,
                    "error": {
                        "type": _qualified_name(error),
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
                self._write_attempt(run_directory, attempt, record)
                raise

            record = {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "run_key": key,
                "attempt": attempt,
                "status": "success",
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "duration_seconds": perf_counter() - started,
                "manifest": normalized_manifest,
                "result": result,
            }
            self._write_attempt(run_directory, attempt, record)
            return BenchmarkOutcome(
                run_key=key,
                attempt=attempt,
                cached=False,
                result=dict(result),
            )

    def iter_attempts(self) -> Iterator[dict[str, Any]]:
        """Yield every saved attempt in stable key/attempt order."""
        for path in sorted(self.root.glob("*/attempt-*.json")):
            yield self._read_attempt(path)

    @staticmethod
    def _latest_attempt(run_directory: Path) -> dict[str, Any] | None:
        paths = sorted(run_directory.glob("attempt-*.json"))
        if not paths:
            return None
        return BenchmarkResultStore._read_attempt(paths[-1])

    @staticmethod
    def _read_attempt(path: Path) -> dict[str, Any]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkSchemaError(
                f"Cannot read benchmark attempt {path}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise BenchmarkSchemaError(
                f"Benchmark attempt {path} must contain a JSON object."
            )
        validate_attempt_record(record)
        return record

    @staticmethod
    def _write_attempt(
        run_directory: Path, attempt: int, record: Mapping[str, Any]
    ) -> None:
        final_path = run_directory / f"attempt-{attempt:04d}.json"
        temporary_path = run_directory / f".attempt-{attempt:04d}.{os.getpid()}.tmp"
        temporary_path.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, final_path)

    @contextmanager
    def _claim(self, run_directory: Path) -> Iterator[None]:
        lock_path = run_directory / "active.lock"
        owner = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at_utc": _utc_now(),
        }
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            if self._discard_dead_local_lock(lock_path):
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            else:
                raise BenchmarkAlreadyRunning(
                    f"Benchmark job is already active: {lock_path.parent.name}"
                ) from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(owner, stream, sort_keys=True)
        try:
            yield
        finally:
            try:
                saved_owner = json.loads(lock_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                saved_owner = None
            if saved_owner == owner:
                lock_path.unlink(missing_ok=True)

    @staticmethod
    def _discard_dead_local_lock(lock_path: Path) -> bool:
        try:
            owner = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if owner.get("hostname") != socket.gethostname():
            return False
        try:
            os.kill(int(owner["pid"]), 0)
        except (OSError, TypeError, ValueError):
            lock_path.unlink(missing_ok=True)
            return True
        return False
