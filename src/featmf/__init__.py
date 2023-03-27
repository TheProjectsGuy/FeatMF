# Feature Matching Framework
"""
"""

# %%
# --- Build these things first (in the order of dependency) ---
from featmf.__about__ import __version__
# Templates
import featmf.templates
from featmf.templates import KptDetDescAlgo
# Utilities
import featmf.utilities
# Local Feature Detection and Description (LFDD) algorithms
import featmf.lfdd as lfdd

