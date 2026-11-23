# Dataset licences and provenance

The MIT licence in `LICENSE` applies to SpectralSelector source code and
documentation. It does not replace the terms attached to third-party datasets.

## No bundled research datasets

SpectralSelector does not distribute third-party research datasets in its wheel or
source archive. The lettuce water-stress supplementary dataset previously used
during development was removed because its CC BY-NC-ND 4.0 terms did not provide
sufficiently clear permission for package redistribution and label adaptation.
It must not be restored without explicit permission from the rights holder.

The dataset loaders are benchmark code in `benchmarks/_datasets/` of the source
repository; they are not part of the installed package.

`load_lettuce_water_stress(path)` is only a parser for a copy supplied by the
user. It performs no network request and the source observations remain outside
the installed package. Users remain responsible for the source licence.

## Downloaded datasets

Remote loaders download source data on demand, verify a fixed checksum, cache the
file on the user's machine, and return provenance in the dataset object's
`source`, `citation`, and `license` fields. SpectralSelector does not relicense these
resources. See `benchmarks/DATASETS.md` for the current catalogue and source-specific
terms.

## chemotools adapters

The coffee and fermentation loaders call the installed `chemotools` package.
Those data are not bundled by SpectralSelector. Consult the upstream chemotools
documentation and dataset sources for attribution and reuse conditions.
