"""Tests for automatic low-memory selection without physical GPUs."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import memory_policy as memory


class LowMemoryPolicyTests(TestCase):
    def make_args(self, **overrides):
        config = {
            "gpu": True,
            "provider": "intel",
            "gpu_index": 0,
            "directml_device_index": 0,
            "openvino_device": "GPU.0",
            "target_vendor": None,
        }
        config.update(overrides)
        return SimpleNamespace(**config)

    def test_ram_or_dedicated_vram_below_limit_is_sufficient(self):
        cases = [
            (15, 8, True, 1),
            (16, 7, True, 1),
            (15, 7, True, 2),
            (16, 8, False, 0),
            (32, 16, False, 0),
        ]
        for ram_gib, vram_gib, expected, reason_count in cases:
            with self.subTest(ram_gib=ram_gib, vram_gib=vram_gib), (
                patch.object(memory, "total_ram_bytes", return_value=ram_gib * memory.GIB),
                patch.object(memory, "dedicated_vram_bytes", return_value=vram_gib * memory.GIB),
            ):
                enabled, reasons, ram, vram = memory.detect_low_memory(self.make_args())
                self.assertEqual(enabled, expected)
                self.assertEqual(len(reasons), reason_count)
                self.assertEqual(ram, ram_gib * memory.GIB)
                self.assertEqual(vram, vram_gib * memory.GIB)

    def test_unknown_vram_does_not_hide_low_ram(self):
        with (
            patch.object(memory, "total_ram_bytes", return_value=12 * memory.GIB),
            patch.object(memory, "dedicated_vram_bytes", return_value=None),
        ):
            enabled, reasons, _, vram = memory.detect_low_memory(self.make_args())
            self.assertTrue(enabled)
            self.assertEqual(len(reasons), 1)
            self.assertIsNone(vram)

    def test_unknown_ram_does_not_hide_low_vram(self):
        with (
            patch.object(memory, "total_ram_bytes", return_value=None),
            patch.object(memory, "dedicated_vram_bytes", return_value=4 * memory.GIB),
        ):
            enabled, reasons, ram, _ = memory.detect_low_memory(self.make_args())
            self.assertTrue(enabled)
            self.assertEqual(len(reasons), 1)
            self.assertIsNone(ram)

    def test_integrated_gpu_shared_system_memory_is_not_dedicated_vram(self):
        adapters = [{"vendor": 0x8086, "vram": 0, "name": "card0"}]
        with (
            patch.object(memory.platform, "system", return_value="Linux"),
            patch.object(memory, "_linux_cards", return_value=adapters),
            patch.object(memory, "total_ram_bytes", return_value=32 * memory.GIB),
        ):
            enabled, reasons, _, vram = memory.detect_low_memory(self.make_args())
            self.assertTrue(enabled)
            self.assertEqual(vram, 0)
            self.assertEqual(len(reasons), 1)
            self.assertIn("VRAM", reasons[0])

    def test_select_openvino_gpu_number_and_vendor(self):
        adapters = [
            {"vendor": 0x8086, "vram": 0, "name": "card0"},
            {"vendor": 0x10DE, "vram": 8 * memory.GIB, "name": "card1"},
            {"vendor": 0x8086, "vram": 12 * memory.GIB, "name": "card2"},
        ]
        self.assertEqual(memory._selected_adapter(adapters, "intel", 0, 0, "GPU.1", None)["vram"], 12 * memory.GIB)
        self.assertEqual(memory._selected_adapter(adapters, "intel", 0, 0, "GPU.0", None)["vram"], 0)
        self.assertIsNone(memory._selected_adapter(adapters, "ncnn", 0, 0, None, None))

    def test_directml_selects_exact_dxgi_adapter_index(self):
        adapters = [
            {"vendor": 0x8086, "vram": 128 * 1024 ** 2, "name": "iGPU", "index": 0},
            {"vendor": 0x10DE, "vram": 8 * memory.GIB, "name": "RTX", "index": 2},
        ]
        self.assertEqual(memory._selected_adapter(adapters, "directml", 0, 2, None, None)["vram"], 8 * memory.GIB)
        self.assertEqual(memory._selected_adapter(adapters, "directml", 0, 0, None, None)["vram"], 128 * 1024 ** 2)
        self.assertIsNone(memory._selected_adapter(adapters, "directml", 0, 1, None, None))

    def test_cpu_does_not_probe_vram(self):
        with (
            patch.object(memory, "total_ram_bytes", return_value=32 * memory.GIB),
            patch.object(memory, "dedicated_vram_bytes") as gpu,
        ):
            enabled, reasons, _, vram = memory.detect_low_memory(
                self.make_args(gpu=False, provider="cpu"))
            self.assertFalse(enabled)
            self.assertEqual(reasons, [])
            self.assertIsNone(vram)
            gpu.assert_not_called()
