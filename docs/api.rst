API reference
=============

Public selectors should be imported from :mod:`spectralselector`. Each selector
implements the scikit-learn transformer contract and inherits the common support
mask and wavelength metadata behavior from
:class:`~spectralselector.BaseWavelengthSelector`.

Core interface
--------------

.. currentmodule:: spectralselector

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseWavelengthSelector
   IndexSelector
   WavelengthSelectKBest
   MIVIFSelector
   CovSelSelector
   FCovSelSelector
   SPASelector
   VIPSelector
   SelectivityRatioSelector
   CARSSelector
   BOSSSelector
   GeneticAlgorithmSelector
   ICOSelector

Models
------

.. autosummary::
   :toctree: generated
   :nosignatures:

   PLSDA

Aliases
-------

:class:`VIPSSelector` is an alias for :class:`VIPSelector`,
:class:`SRSelector` is an alias for :class:`SelectivityRatioSelector`, and
:class:`GASelector` is an alias for :class:`GeneticAlgorithmSelector`.
These aliases are retained for compatibility; documentation and new code should
use the canonical class names. In particular, ``GASelector`` remains the generic
mask optimizer and does not imply a validated GA-iPLS or ICO implementation.

API stability
-------------

The package is currently an alpha ``0.x`` release. Canonical exports, aliases,
constructor parameters, and fitted attributes are covered by contract tests,
but the public API is not frozen while scientific reference validation remains
open. Any future deprecation must first be documented in the changelog and emit
a release-visible warning before removal. No PCA-LDA estimator is part of the
public API; classification uses :class:`PLSDA`.
