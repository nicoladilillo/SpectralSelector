# JOSS manuscript

`paper.md` and `paper.bib` are the version-controlled manuscript sources.
Generated PDF, JATS, Crossref, HTML, TeX, and preprint files are local build
artifacts and are ignored by Git.

From the repository root, run the official Open Journals Inara container:

```bash
docker run --rm \
  -v "$PWD:/data" \
  -u "$(id -u):$(id -g)" \
  openjournals/inara \
  -o pdf,jats \
  paper/paper.md
```

Inspect both generated formats and resolve every warning before submission. A
running Docker-compatible daemon is required. When Docker is unavailable,
Pandoc can still detect unresolved citations and basic Markdown problems, but
it does not replace Inara's JOSS templates and metadata validation:

```bash
pandoc paper/paper.md \
  --citeproc \
  --bibliography=paper/paper.bib \
  --to=html5 \
  --output=/tmp/spectralselector-paper.html
```

The draft intentionally states that independent adoption, the public release,
and the archival DOI are pending. Update those claims only after the evidence
exists, then ensure the paper version matches the tagged package release.
