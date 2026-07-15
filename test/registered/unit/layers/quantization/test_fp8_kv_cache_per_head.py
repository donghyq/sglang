"""Unit tests for per-head FP8 KV cache scale handling."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.quantization.kv_cache import (
    BaseKVCacheMethod,
    _load_kv_scale,
    create_kv_scale_parameters,
    set_kv_scale,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    KVWriteLoc,
    MHATokenToKVPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_loader.weight_utils import (
    KVCacheScalePair,
    kv_cache_scales_loader,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPerHeadKVCacheScales(CustomTestCase):
    def _layer(self, num_heads=4):
        return SimpleNamespace(tp_k_head_num=num_heads)

    def test_create_per_head_parameters(self):
        layer = self._layer()
        create_kv_scale_parameters(layer, per_head=True)

        self.assertEqual(layer.k_scale.shape, (4,))
        self.assertEqual(layer.v_scale.shape, (4,))
        self.assertTrue(torch.equal(layer.k_scale, torch.full((4,), -1.0)))

    def test_scalar_checkpoint_scale_broadcasts_to_heads(self):
        layer = self._layer()
        create_kv_scale_parameters(layer, per_head=True)

        _load_kv_scale(layer.k_scale, torch.tensor(0.25))
        self.assertTrue(torch.equal(layer.k_scale, torch.full((4,), 0.25)))

    def test_set_scale_keeps_registered_parameters(self):
        layer = self._layer()
        create_kv_scale_parameters(layer, per_head=True)
        k_scale = layer.k_scale
        v_scale = layer.v_scale

        set_kv_scale(layer, 0.25)

        self.assertIs(layer.k_scale, k_scale)
        self.assertIs(layer.v_scale, v_scale)
        self.assertTrue(torch.equal(layer.k_scale, torch.full((4,), 0.25)))

    def test_set_scale_creates_legacy_scalar_parameters_when_missing(self):
        layer = self._layer()
        layer.k_scale = None
        layer.v_scale = None

        set_kv_scale(layer, 0.25)

        self.assertEqual(layer.k_scale.shape, ())
        self.assertEqual(layer.v_scale.shape, ())
        self.assertEqual(layer.k_scale.item(), 0.25)
        self.assertEqual(layer.v_scale.item(), 0.25)

    def test_vector_checkpoint_scale_is_preserved(self):
        layer = self._layer()
        create_kv_scale_parameters(layer, per_head=True)
        expected = torch.tensor([0.1, 0.2, 0.3, 0.4])

        _load_kv_scale(layer.k_scale, expected)
        self.assertTrue(torch.equal(layer.k_scale, expected))

    def test_separate_kv_scales_are_preserved(self):
        layer = self._layer()
        create_kv_scale_parameters(layer, per_head=True)

        set_kv_scale(
            layer,
            {
                "k": [0.1, 0.2, 0.3, 0.4],
                "v": [0.5, 0.6, 0.7, 0.8],
            },
        )

        torch.testing.assert_close(
            layer.k_scale, torch.tensor([0.1, 0.2, 0.3, 0.4])
        )
        torch.testing.assert_close(
            layer.v_scale, torch.tensor([0.5, 0.6, 0.7, 0.8])
        )

    def test_process_weights_supports_per_head_scales(self):
        layer = self._layer()
        create_kv_scale_parameters(layer, per_head=True)
        layer.k_scale.data.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        layer.v_scale.data.copy_(torch.tensor([0.5, 0.6, 0.7, 0.8]))
        method = BaseKVCacheMethod(SimpleNamespace())

        with patch(
            "sglang.srt.layers.quantization.kv_cache.is_fp8_fnuz",
            return_value=False,
        ):
            method.process_weights_after_loading(layer)

        self.assertIsNone(layer.k_scale_float)
        self.assertIsNone(layer.v_scale_float)
        self.assertTrue(
            torch.equal(layer.k_scale, torch.tensor([0.1, 0.2, 0.3, 0.4]))
        )

    def test_per_head_scale_broadcast_shape(self):
        cache = torch.ones((2, 4, 8))
        scale = torch.arange(1, 5, dtype=torch.float32)

        reshaped = MHATokenToKVPool._reshape_kv_scale(scale, cache)
        self.assertEqual(reshaped.shape, (1, 4, 1))
        torch.testing.assert_close(cache / reshaped, cache / scale.view(1, 4, 1))

    def test_per_head_scale_rejects_wrong_head_count(self):
        cache = torch.ones((2, 4, 8))
        with self.assertRaisesRegex(ValueError, "num_kv_heads"):
            MHATokenToKVPool._reshape_kv_scale(torch.ones(3), cache)


class TestPerHeadScaleHybridPools(CustomTestCase):
    def test_hybrid_linear_pool_forwards_tensor_scales(self):
        pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
        pool.use_mla = False
        pool.full_attention_layer_id_mapping = {3: 0}
        pool.full_kv_pool = MagicMock()
        layer = SimpleNamespace(layer_id=3)
        loc = torch.tensor([1, 2])
        full_loc = torch.tensor([5, 6])
        k_scale = torch.tensor([0.1, 0.2])
        v_scale = torch.tensor([0.3, 0.4])

        pool.set_kv_buffer(
            layer,
            KVWriteLoc(loc=loc, full_loc=full_loc),
            torch.ones((2, 2, 4)),
            torch.ones((2, 2, 4)),
            k_scale,
            v_scale,
        )

        args, kwargs = pool.full_kv_pool.set_kv_buffer.call_args
        self.assertIs(args[1], full_loc)
        self.assertIs(args[4], k_scale)
        self.assertIs(args[5], v_scale)
        self.assertEqual(kwargs["layer_id_override"], 0)

    def test_swa_pool_forwards_tensor_scales_to_swa_subpool(self):
        pool = SWAKVPool.__new__(SWAKVPool)
        pool.layer_transfer_counter = None
        pool.layers_mapping = {2: (0, True)}
        pool.swa_kv_pool = MagicMock()
        pool.full_kv_pool = MagicMock()
        layer = SimpleNamespace(layer_id=2)
        loc = torch.tensor([1, 2])
        swa_loc = torch.tensor([3, 4])
        k_scale = torch.tensor([0.1, 0.2])
        v_scale = torch.tensor([0.3, 0.4])

        pool.set_kv_buffer(
            layer,
            KVWriteLoc(loc=loc, swa_loc=swa_loc),
            torch.ones((2, 2, 4)),
            torch.ones((2, 2, 4)),
            k_scale,
            v_scale,
        )

        args, kwargs = pool.swa_kv_pool.set_kv_buffer.call_args
        self.assertIs(args[1], swa_loc)
        self.assertIs(args[4], k_scale)
        self.assertIs(args[5], v_scale)
        self.assertEqual(kwargs["layer_id_override"], 0)
        pool.full_kv_pool.set_kv_buffer.assert_not_called()


class TestPerHeadKVCacheScaleLoader(CustomTestCase):
    def _write_params(self, layer_scale):
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "kv_scales.json"
        path.write_text(
            json.dumps(
                {
                    "model_type": "dummy",
                    "kv_cache": {
                        "dtype": "float8_e4m3fn",
                        "scaling_factor": {"0": {"0": layer_scale}},
                    },
                }
            )
        )
        return temp_dir, path

    def test_loads_legacy_scalar_scale(self):
        temp_dir, path = self._write_params(0.25)
        self.addCleanup(temp_dir.cleanup)

        loaded = dict(kv_cache_scales_loader(str(path), 0, 1, 1, "dummy"))
        self.assertEqual(loaded[0], 0.25)

    def test_loads_separate_per_head_scales(self):
        temp_dir, path = self._write_params(
            {
                "k": [0.1, 0.2, 0.3, 0.4],
                "v": [0.5, 0.6, 0.7, 0.8],
            }
        )
        self.addCleanup(temp_dir.cleanup)

        loaded = dict(kv_cache_scales_loader(str(path), 0, 1, 1, "dummy"))
        self.assertIsInstance(loaded[0], KVCacheScalePair)
        self.assertEqual(loaded[0].k, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(loaded[0].v, [0.5, 0.6, 0.7, 0.8])


class TestPerHeadKVCacheServerArgs(CustomTestCase):
    def test_requires_fp8_e4m3(self):
        args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="auto",
            kv_cache_quant_granularity="per_head",
        )
        with self.assertRaisesRegex(ValueError, "fp8_e4m3"):
            args._handle_fp8_kv_cache_quant_granularity()

    def test_requires_fa3_for_both_phases(self):
        args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="fp8_e4m3",
            kv_cache_quant_granularity="per_head",
        )
        with patch.object(args, "use_mla_backend", return_value=False), patch.object(
            args,
            "_resolved_attention_backends",
            return_value=("fa3", "flashinfer"),
        ):
            with self.assertRaisesRegex(ValueError, "requires FA3"):
                args._handle_fp8_kv_cache_quant_granularity()

    def test_accepts_fa3_for_both_phases(self):
        args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="fp8_e4m3",
            kv_cache_quant_granularity="per_head",
        )
        with patch.object(args, "use_mla_backend", return_value=False), patch.object(
            args, "_resolved_attention_backends", return_value=("fa3", "fa3")
        ):
            args._handle_fp8_kv_cache_quant_granularity()


if __name__ == "__main__":
    unittest.main()
