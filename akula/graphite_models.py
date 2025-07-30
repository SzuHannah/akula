from __future__ import annotations

"""Simplified wrappers for specialised LCA and TEA models used in the
project documentation. These classes adapt the generic :mod:`model_framework`
interfaces to the specific helper functions provided in ``lca.py`` and
``vector_model.py``. Only a minimal subset of functionality is implemented to
keep the repository lightweight while demonstrating how custom models can be
integrated."""

from typing import Sequence
import numpy as np

from .model_framework import LCAModel, TEAModel

# -- LCA wrapper ---------------------------------------------------------------
class GraphiteLCAModel(LCAModel):
    """LCA model using helper functions from ``lca.py``."""

    lca_module: object
    methods: Sequence[tuple[str, str, str]]

    def __init__(self, lca_module: object, methods: Sequence[tuple[str, str, str]]):
        super().__init__(np.empty((0, 0)))
        self.lca_module = lca_module
        self.methods = list(methods)

    def single_score(self, scenario: str, slate: tuple[float, float, float], allocation: str = "system_expansion") -> np.ndarray:
        """Compute characterised scores for a given scenario using the supplied
        LCA helper module."""
        return self.lca_module.graphite_lca_single(scenario, slate, allocation, self.methods)

# -- TEA wrapper ---------------------------------------------------------------
class GraphiteTEAModel(TEAModel):
    """TEA model using the vectorised implementation from ``vector_model.py``."""

    tea_model: object

    def __init__(self, tea_model: object):
        super().__init__(np.empty((0, 0)))
        self.tea_model = tea_model

    def run(self, **kwargs):
        return self.tea_model.run(**kwargs)
