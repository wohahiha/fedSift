from __future__ import annotations
import unittest
from collections import OrderedDict
import torch
from fedsift.dp_kernel import (
    DPKernelError,
    add_post_privacy_correction,
    privatize_per_record_gradients,
)


class DPKernelGoldenTests(unittest.TestCase):

    def test_literal_clip_sum_noise_normalize_order(self) -> None:
        gradients = OrderedDict(
            w=torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64),
            b=torch.tensor([[0.0], [12.0]], dtype=torch.float64),
        )
        noise = OrderedDict(
            w=torch.tensor([1.0, -2.0], dtype=torch.float64),
            b=torch.tensor([3.0], dtype=torch.float64),
        )
        observed = privatize_per_record_gradients(
            gradients, noise, clip_norm=6.0, fixed_normalization=4.0
        )
        self.assertTrue(torch.equal(observed["w"], torch.tensor([1.0, 0.5], dtype=torch.float64)))
        self.assertTrue(torch.equal(observed["b"], torch.tensor([2.25], dtype=torch.float64)))

    def test_empty_draw_still_returns_noise_over_fixed_normalization(self) -> None:
        gradients = OrderedDict(
            w=torch.empty((0, 2), dtype=torch.float64), b=torch.empty((0, 1), dtype=torch.float64)
        )
        noise = OrderedDict(
            w=torch.tensor([2.0, -4.0], dtype=torch.float64),
            b=torch.tensor([6.0], dtype=torch.float64),
        )
        observed = privatize_per_record_gradients(
            gradients, noise, clip_norm=1.0, fixed_normalization=2.0
        )
        self.assertTrue(torch.equal(observed["w"], torch.tensor([1.0, -2.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(observed["b"], torch.tensor([3.0], dtype=torch.float64)))

    def test_correction_is_separate_and_post_privacy(self) -> None:
        corrected = add_post_privacy_correction(
            OrderedDict(x=torch.tensor([1.5], dtype=torch.float64)),
            OrderedDict(x=torch.tensor([-0.25], dtype=torch.float64)),
        )
        self.assertEqual(float(corrected["x"].item()), 1.25)

    def test_malformed_inputs_fail_closed(self) -> None:
        with self.assertRaises(DPKernelError):
            privatize_per_record_gradients(
                OrderedDict(x=torch.ones((2, 1), dtype=torch.float64)),
                OrderedDict(x=torch.ones((2, 1), dtype=torch.float64)),
                clip_norm=1.0,
                fixed_normalization=2.0,
            )
        with self.assertRaises(DPKernelError):
            privatize_per_record_gradients(
                OrderedDict(
                    x=torch.ones((2, 1), dtype=torch.float64),
                    y=torch.ones((3, 1), dtype=torch.float64),
                ),
                OrderedDict(
                    x=torch.zeros((1,), dtype=torch.float64),
                    y=torch.zeros((1,), dtype=torch.float64),
                ),
                clip_norm=1.0,
                fixed_normalization=2.0,
            )
        with self.assertRaises(DPKernelError):
            privatize_per_record_gradients(
                OrderedDict(x=torch.ones((1, 1), dtype=torch.float64)),
                OrderedDict(x=torch.zeros((1,), dtype=torch.float64)),
                clip_norm=True,
                fixed_normalization=1.0,
            )


if __name__ == "__main__":
    unittest.main()
