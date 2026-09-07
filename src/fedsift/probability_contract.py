"""One probability-clipping contract shared by control and evaluation.

The fixed symmetric clip is a numerical definition of the reported binary
log loss, not an algorithm-specific safeguard.  It must be applied to every
method and every role before FedSift outer-test access is enabled.
"""

from __future__ import annotations

PROBABILITY_CLIP_EPSILON = 1e-07
PROBABILITY_CLIP_POLICY = "fixed_symmetric_1e_minus_7_all_methods_and_roles"
__all__ = ["PROBABILITY_CLIP_EPSILON", "PROBABILITY_CLIP_POLICY"]
