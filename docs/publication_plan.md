# Publication scope and release stages

SpectralSelector should be published in stages because a usable research
software release and a claim of paper-equivalent scientific validation are not
the same milestone.

## Stage 1: public alpha software release

The first publication means all of the following together:

- a public GitHub repository with CI, issue tracking, security reporting, and
  tagged source history;
- a `0.x` PyPI release built by trusted publishing;
- hosted Read the Docs documentation for the same tag; and
- a software archive on Zenodo linked to that GitHub release.

This stage is an engineering and early-adopter release. It may expose every
implemented selector because the documentation distinguishes paper-derived,
package-extension, and generic behavior. It must not claim that CARS, BOSS, GA,
or another method is numerically equivalent to author code without the pending
independent fixtures. The lettuce-health dataset and its future DOI are a
separate publication and are not a prerequisite for releasing the software.

Keep the package at `0.x` during this stage. The canonical API is tested but not
frozen; backwards-incompatible changes may still be necessary after independent
method validation. Such changes must be documented and, when practical, use a
warning-based deprecation period.

## Stage 2: scientific-validation release

Publish another tagged software/Zenodo version after the complete benchmark
matrix and independent numerical method comparisons are archived. This release
must identify the exact package commit, dependency environment, dataset
versions, fixed seeds, split hashes, resource budgets, failures, and complete
result archive DOI. Promote individual method claims only to the level supported
by that evidence.

## Stage 3: JOSS submission and stable API

JOSS submission comes after genuine public development history, external use,
domain review, a clean installation reproduction, and the official manuscript
check. Consider `1.0` only when the scientific scope and canonical API are
stable; JOSS submission alone does not require changing the version to `1.0`.

The dataset Zenodo record, software Zenodo archive, and eventual JOSS article
must use separate citations and DOIs. `CITATION.cff` should cite only records
that exist at the tagged release, never provisional identifiers.

## Stage-1 decision still owned by the maintainer

An anonymous availability check on 2026-09-08 found no registered
`spectralselector` project at either PyPI's project page or JSON API. PyPI name
availability is not a reservation and can change at any time, so repeat the
check immediately before the first upload.

Repository code cannot choose the final GitHub organization, reserve or control
the PyPI name, configure hosted accounts, or accept legal responsibility for
publication. Those external checks remain explicit release blockers even when
all local quality gates pass.
