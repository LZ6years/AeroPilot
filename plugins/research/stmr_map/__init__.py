"""Standalone STMR research module.

This package mirrors the Semantic-Topo-Metric Representation pipeline used in
the STMR paper.  The core remains standalone; UAV-Flow may explicitly install
the online adapter as its ``map`` context provider for controlled evaluation.
"""

from .core import IncrementalStmrMapState, IncrementalStmrUpdate, StmrMap, build_stmr_map
from .online import StmrOnlineMapBuilder, StmrOnlineMapResult

__all__ = [
    "IncrementalStmrMapState",
    "IncrementalStmrUpdate",
    "StmrMap",
    "StmrOnlineMapBuilder",
    "StmrOnlineMapResult",
    "build_stmr_map",
]
