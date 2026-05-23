"""Backward-compatible FACET matrix imports.

The implementation now lives in :mod:`analyze.facet_matrix_ops`, where the
analyzer is decomposed into chainable operators. Existing imports from
``analyze.facet_matrix`` are preserved.
"""
from __future__ import annotations

from analyze.facet_matrix_ops import *  # noqa: F401,F403
from analyze.facet_matrix_ops import __all__  # re-export the package contract
