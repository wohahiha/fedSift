"""Pure record-level DP gradient kernel with explicit randomness.

The caller supplies per-record gradients, already sampled by its registered
Poisson slot plan, and an explicit Gaussian-noise realization.  This module
does not generate randomness, inspect data, or maintain an accountant.  It is
therefore suitable for literal numerical-oracle tests and later runner wiring.
"""

from __future__ import annotations
import math
from collections import OrderedDict
from numbers import Real
from typing import Mapping
import torch

GradientState = OrderedDict[str, torch.Tensor]


class DPKernelError(RuntimeError):
    """Raised when the registered DP-kernel contract is violated."""


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DPKernelError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise DPKernelError(f"{name} must be finite and positive")
    return number


def _validate_gradient_collection(
    per_record_gradients: Mapping[str, torch.Tensor], noise: Mapping[str, torch.Tensor]
) -> tuple[tuple[str, ...], int, torch.dtype, torch.device]:
    names = tuple(per_record_gradients)
    if not names or tuple(noise) != names:
        raise DPKernelError("gradient and noise parameter identities must align")
    record_count: int | None = None
    reference_dtype: torch.dtype | None = None
    reference_device: torch.device | None = None
    for name in names:
        gradient = per_record_gradients[name]
        noise_value = noise[name]
        if not isinstance(gradient, torch.Tensor) or not isinstance(noise_value, torch.Tensor):
            raise DPKernelError("gradient and noise values must be tensors")
        if gradient.ndim < 1:
            raise DPKernelError("per-record gradient requires a leading record axis")
        if not gradient.dtype.is_floating_point or noise_value.dtype != gradient.dtype:
            raise DPKernelError("gradient and noise require one floating-point dtype")
        if noise_value.device != gradient.device:
            raise DPKernelError("gradient and noise devices differ")
        if tuple(noise_value.shape) != tuple(gradient.shape[1:]):
            raise DPKernelError("noise shape differs from one parameter gradient")
        if not torch.isfinite(gradient).all() or not torch.isfinite(noise_value).all():
            raise DPKernelError("gradient or noise contains a non-finite value")
        if record_count is None:
            record_count = int(gradient.shape[0])
            reference_dtype = gradient.dtype
            reference_device = gradient.device
        if int(gradient.shape[0]) != record_count:
            raise DPKernelError("per-record gradient record counts differ")
        if gradient.dtype != reference_dtype or gradient.device != reference_device:
            raise DPKernelError("gradient parameter dtype or device differs")
    assert record_count is not None
    assert reference_dtype is not None
    assert reference_device is not None
    return (names, record_count, reference_dtype, reference_device)


def privatize_per_record_gradients(
    per_record_gradients: Mapping[str, torch.Tensor],
    gaussian_noise: Mapping[str, torch.Tensor],
    *,
    clip_norm: float,
    fixed_normalization: float,
) -> GradientState:
    """Jointly clip records, sum, add noise, then use fixed normalization.

    `gaussian_noise` is the already drawn additive noise on the clipped sum.  A
    conforming caller draws each tensor from `N(0, (sigma * clip_norm)^2 I)`.
    Empty Poisson draws are valid: the clipped sum is zero, supplied noise is
    still added, and the caller must still advance its accountant.
    """
    clip_norm = _positive_finite(clip_norm, "clip norm")
    fixed_normalization = _positive_finite(fixed_normalization, "fixed normalization")
    names, record_count, dtype, device = _validate_gradient_collection(
        per_record_gradients, gaussian_noise
    )
    squared_norm = torch.zeros(record_count, dtype=dtype, device=device)
    for name in names:
        gradient = per_record_gradients[name]
        parameter_width = math.prod((int(value) for value in gradient.shape[1:]))
        squared_norm = squared_norm + gradient.reshape(record_count, parameter_width).pow(2).sum(
            dim=1
        )
    record_norm = torch.sqrt(squared_norm)
    factors = torch.clamp(clip_norm / torch.clamp(record_norm, min=clip_norm), max=1.0)
    privatized: GradientState = OrderedDict()
    for name in names:
        gradient = per_record_gradients[name]
        factor_shape = (record_count,) + (1,) * (gradient.ndim - 1)
        clipped_sum = (gradient * factors.reshape(factor_shape)).sum(dim=0)
        value = (clipped_sum + gaussian_noise[name]) / fixed_normalization
        if not torch.isfinite(value).all():
            raise DPKernelError("privatized gradient is non-finite")
        privatized[name] = value
    return privatized


def add_post_privacy_correction(
    privatized_gradient: Mapping[str, torch.Tensor],
    data_independent_correction: Mapping[str, torch.Tensor],
) -> GradientState:
    """Add a fixed/public or prior-DP correction after the private mechanism."""
    names = tuple(privatized_gradient)
    if not names or tuple(data_independent_correction) != names:
        raise DPKernelError("private gradient and correction identities must align")
    output: GradientState = OrderedDict()
    for name in names:
        private = privatized_gradient[name]
        correction = data_independent_correction[name]
        if (
            not isinstance(private, torch.Tensor)
            or not isinstance(correction, torch.Tensor)
            or private.shape != correction.shape
            or (private.dtype != correction.dtype)
            or (private.device != correction.device)
            or (not private.dtype.is_floating_point)
            or (not torch.isfinite(private).all())
            or (not torch.isfinite(correction).all())
        ):
            raise DPKernelError("post-privacy correction contract is invalid")
        value = private + correction
        if not torch.isfinite(value).all():
            raise DPKernelError("post-privacy corrected gradient is non-finite")
        output[name] = value
    return output
