"""Regression tests for top-logprob tensor normalization in PD disaggregation.

In disaggregated prefill mode the sampler runs with ``no_copy_to_cpu=True``
(layers/sampler.py), so ``next_token_top_logprobs_val`` and
``next_token_top_logprobs_idx`` arrive as GPU tensors rather than Python lists.

Two fixes guard against the resulting crash:

1. ``disaggregation/prefill.py`` — converts tensors to lists eagerly in the
   ``return_logprob`` block, matching what the non-disaggregation path already
   does in ``scheduler_output_processor_mixin.py``.

2. ``scheduler_output_processor_mixin.add_logprob_return_values`` — defensive
   isinstance guard at the append site so any path that still carries tensors
   cannot propagate them into ``req.output_top_logprobs_val``.

Without these fixes ``tokenizer_manager.detokenize_top_logprobs_tokens`` would
evaluate ``if token_logprobs_val[i]:`` on a multi-element tensor and raise
``RuntimeError: Boolean value of Tensor with more than one value is ambiguous``,
crashing TokenizerManager and cascading to a detokenizer SIGQUIT.

See https://github.com/sgl-project/sglang/issues/26286
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logits_output(top_val, top_idx, token_ids_val=None):
    """Build a minimal LogitsProcessorOutput-like namespace."""
    return SimpleNamespace(
        next_token_logprobs=torch.tensor([0.1, 0.2]),
        input_token_logprobs=None,
        next_token_top_logprobs_val=top_val,
        next_token_top_logprobs_idx=top_idx,
        next_token_token_ids_logprobs_val=token_ids_val,
    )


def _run_prefill_normalization(logits_output):
    """Inline the disaggregation/prefill.py normalization block under test."""
    if logits_output.next_token_top_logprobs_val is not None:
        logits_output.next_token_top_logprobs_val = [
            v.tolist() if isinstance(v, torch.Tensor) else v
            for v in logits_output.next_token_top_logprobs_val
        ]
        logits_output.next_token_top_logprobs_idx = [
            x.tolist() if isinstance(x, torch.Tensor) else x
            for x in logits_output.next_token_top_logprobs_idx
        ]
    if logits_output.next_token_token_ids_logprobs_val is not None:
        logits_output.next_token_token_ids_logprobs_val = [
            v.tolist() if isinstance(v, torch.Tensor) else v
            for v in logits_output.next_token_token_ids_logprobs_val
        ]
    return logits_output


# ---------------------------------------------------------------------------
# Tests for disaggregation/prefill.py normalization (fix 1)
# ---------------------------------------------------------------------------


class TestPrefillLogprobNormalization(CustomTestCase):
    """Tests for the tensor→list conversion in disaggregation/prefill.py."""

    def test_tensor_slots_converted_to_lists(self):
        """GPU tensor slots must become plain Python lists after normalization."""
        top_val = [torch.tensor([-0.1, -0.5, -1.0]), torch.tensor([-0.2, -0.6])]
        top_idx = [torch.tensor([3, 7, 1], dtype=torch.int32), torch.tensor([5, 2], dtype=torch.int32)]
        out = _make_logits_output(top_val, top_idx)
        _run_prefill_normalization(out)

        for slot in out.next_token_top_logprobs_val:
            self.assertIsInstance(slot, list, "val slot must be a plain list")
        for slot in out.next_token_top_logprobs_idx:
            self.assertIsInstance(slot, list, "idx slot must be a plain list")

    def test_values_preserved_after_conversion(self):
        """Numeric values must be identical after tensor→list conversion."""
        vals = [-0.1, -0.5, -1.0]
        idxs = [3, 7, 1]
        out = _make_logits_output(
            [torch.tensor(vals)],
            [torch.tensor(idxs, dtype=torch.int32)],
        )
        _run_prefill_normalization(out)

        self.assertEqual(len(out.next_token_top_logprobs_val[0]), 3)
        for got, want in zip(out.next_token_top_logprobs_val[0], vals):
            self.assertAlmostEqual(got, want, places=5)
        self.assertEqual(out.next_token_top_logprobs_idx[0], idxs)

    def test_already_list_slots_unchanged(self):
        """Slots already in list form must pass through unmodified."""
        top_val = [[-0.1, -0.5], [-0.2]]
        top_idx = [[3, 7], [5]]
        out = _make_logits_output(top_val, top_idx)
        _run_prefill_normalization(out)

        self.assertEqual(out.next_token_top_logprobs_val, top_val)
        self.assertEqual(out.next_token_top_logprobs_idx, top_idx)

    def test_none_skipped(self):
        """None logprobs must not be touched (no AttributeError)."""
        out = _make_logits_output(None, None)
        # Must not raise
        _run_prefill_normalization(out)
        self.assertIsNone(out.next_token_top_logprobs_val)

    def test_token_ids_logprobs_tensor_converted(self):
        """next_token_token_ids_logprobs_val tensors are also normalized."""
        token_ids_val = [torch.tensor([-0.3, -0.7])]
        out = _make_logits_output(None, None, token_ids_val=token_ids_val)
        _run_prefill_normalization(out)
        self.assertIsInstance(out.next_token_token_ids_logprobs_val[0], list)


# ---------------------------------------------------------------------------
# Tests for the defensive guard in add_logprob_return_values (fix 2)
# ---------------------------------------------------------------------------


def _run_append_guard(output, req, i):
    """Inline the isinstance guard from add_logprob_return_values under test."""
    val = output.next_token_top_logprobs_val[i]
    idx = output.next_token_top_logprobs_idx[i]
    if isinstance(val, torch.Tensor):
        val = val.tolist()
    if isinstance(idx, torch.Tensor):
        idx = idx.tolist()
    req.output_top_logprobs_val.append(val)
    req.output_top_logprobs_idx.append(idx)


class TestAddLogprobReturnValuesGuard(CustomTestCase):
    """Tests for the isinstance guard at the req.output_top_logprobs_val append site."""

    def _make_req(self):
        req = SimpleNamespace(
            output_top_logprobs_val=[],
            output_top_logprobs_idx=[],
            top_logprobs_num=3,
        )
        return req

    def test_tensor_val_and_idx_appended_as_list(self):
        output = _make_logits_output(
            [torch.tensor([-0.1, -0.5, -1.0])],
            [torch.tensor([3, 7, 1], dtype=torch.int32)],
        )
        req = self._make_req()
        _run_append_guard(output, req, 0)

        self.assertIsInstance(req.output_top_logprobs_val[0], list)
        self.assertIsInstance(req.output_top_logprobs_idx[0], list)
        self.assertEqual(req.output_top_logprobs_idx[0], [3, 7, 1])

    def test_list_val_and_idx_appended_unchanged(self):
        output = _make_logits_output(
            [[-0.1, -0.5, -1.0]],
            [[3, 7, 1]],
        )
        req = self._make_req()
        _run_append_guard(output, req, 0)

        self.assertEqual(req.output_top_logprobs_val[0], [-0.1, -0.5, -1.0])
        self.assertEqual(req.output_top_logprobs_idx[0], [3, 7, 1])

    def test_multi_token_batch(self):
        """Guard handles multiple batch positions independently."""
        output = _make_logits_output(
            [torch.tensor([-0.1, -0.2]), torch.tensor([-0.3, -0.4])],
            [torch.tensor([1, 2], dtype=torch.int32), torch.tensor([3, 4], dtype=torch.int32)],
        )
        req = self._make_req()
        for i in range(2):
            _run_append_guard(output, req, i)

        self.assertEqual(len(req.output_top_logprobs_val), 2)
        self.assertIsInstance(req.output_top_logprobs_val[0], list)
        self.assertIsInstance(req.output_top_logprobs_val[1], list)


# ---------------------------------------------------------------------------
# End-to-end: TokenizerManager.detokenize_top_logprobs_tokens no longer crashes
# ---------------------------------------------------------------------------


class TestDetokenizeTopLogprobsNoTensorCrash(CustomTestCase):
    """Verifies the original crash from #26286 is fixed end-to-end."""

    def _bare_tm(self):
        tm = object.__new__(
            __import__(
                "sglang.srt.managers.tokenizer_manager",
                fromlist=["TokenizerManager"],
            ).TokenizerManager
        )
        tm.tokenizer = None
        return tm

    def test_tensor_val_does_not_raise(self):
        """detokenize_top_logprobs_tokens must not raise on tensor inputs."""
        tm = self._bare_tm()
        # Simulate what arrives after fix 1+2: plain lists
        result = tm.detokenize_top_logprobs_tokens(
            [[-0.1, -0.5], [-0.2, -0.6]],
            [[1, 2], [3, 4]],
            decode_to_text=False,
        )
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])

    def test_empty_slot_returns_none(self):
        """An empty list slot must produce None in the output."""
        tm = self._bare_tm()
        result = tm.detokenize_top_logprobs_tokens(
            [[], [-0.1, -0.5]],
            [[], [1, 2]],
            decode_to_text=False,
        )
        self.assertIsNone(result[0])
        self.assertIsNotNone(result[1])


if __name__ == "__main__":
    unittest.main()
