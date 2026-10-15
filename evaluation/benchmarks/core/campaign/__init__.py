"""Dataset-independent campaign orchestration, recovery and supervision.

This package owns evaluation infrastructure only. Dataset adapters retain case
selection, original instructions, truth checks, runners and metric semantics.
Imports stay lazy so CLI help and offline validation do not initialize CUDA.
"""
