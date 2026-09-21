"""Paired group bootstrap for record-weighted losses on fixed OOF predictions."""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class PairedIntervals:
    effect: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    group_count: int
    replicates: int
    seed: int


def paired_group_intervals(differences, group_ids, *, seed: int, replicates: int = 10_000):
    """Resample whole groups with common weights for all comparison columns.

    Each draw samples G groups uniformly with replacement. All records in a
    sampled group receive its multiplicity; the denominator is the resulting
    record count, so unequal groups retain their record weights. Intervals are
    the 2.5 and 97.5 percentiles, conditional on the supplied predictions.
    """
    values = np.asarray(differences, dtype=np.float64)
    groups = np.asarray(group_ids)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("differences must be a nonempty finite record-by-comparison matrix")
    if groups.ndim != 1 or len(groups) != len(values):
        raise ValueError("group IDs must align with the prediction rows")
    if isinstance(replicates, bool) or not isinstance(replicates, Integral) or replicates < 2:
        raise ValueError("replicates must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    identifiers, membership = np.unique(groups, return_inverse=True)
    group_count = len(identifiers)
    counts = np.bincount(membership, minlength=group_count)
    totals = np.zeros((group_count, values.shape[1]), dtype=np.float64)
    np.add.at(totals, membership, values)
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, values.shape[1]), dtype=np.float64)
    for start in range(0, replicates, 256):
        stop = min(start + 256, replicates)
        weights = np.array([
            np.bincount(rng.integers(0, group_count, group_count), minlength=group_count)
            for _ in range(start, stop)
        ])
        draws[start:stop] = (weights @ totals) / (weights @ counts)[:, None]
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
    return PairedIntervals(values.mean(axis=0), lower, upper, group_count, replicates, int(seed))
