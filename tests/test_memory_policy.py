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

    def test_ram_or_selected_gpu_memory_below_limit_is_sufficient(self):
        cases = [
            (15, 8, True, 1),
            (16, 7, True, 1),
            (15, 7, True, 2),
            (16, 8, False, 0),
            (32, 16, False, 0),
        ]
        for ram_gib, vram_gib, expected, reason_count in cases:
            with (
                self.subTest(ram_gib=ram_gib, vram_gib=vram_gib),
                patch.object(memory, "total_ram_bytes", return_value=ram_gib * memory.GIB),
                patch.object(memory, "gpu_memory_capacity", return_value=(vram_gib * memory.GIB, "専用VRAM")),
            ):
                enabled, reasons, ram, vram = memory.detect_low_memory(self.make_args())
                self.assertEqual(enabled, expected)
                self.assertEqual(len(reasons), reason_count)
                self.assertEqual(ram, ram_gib * memory.GIB)
                self.assertEqual(vram, vram_gib * memory.GIB)

    def test_unknown_vram_does_not_hide_low_ram(self):
        with (
            patch.object(memory, "total_ram_bytes", return_value=12 * memory.GIB),
            patch.object(memory, "gpu_memory_capacity", return_value=(None, None)),
        ):
            enabled, reasons, _, vram = memory.detect_low_memory(self.make_args())
            self.assertTrue(enabled)
            self.assertEqual(len(reasons), 1)
            self.assertIsNone(vram)

    def test_unknown_ram_does_not_hide_low_vram(self):
        with (
            patch.object(memory, "total_ram_bytes", return_value=None),
            patch.object(memory, "gpu_memory_capacity", return_value=(4 * memory.GIB, "共有GPUメモリ")),
        ):
            enabled, reasons, ram, _ = memory.detect_low_memory(self.make_args())
            self.assertTrue(enabled)
            self.assertEqual(len(reasons), 1)
            self.assertIsNone(ram)

    def test_integrated_gpu_uses_shared_memory_when_dedicated_is_absent(self):
        adapters = [{"vendor": 0x8086, "dedicated": 0, "shared": 12 * memory.GIB, "name": "card0"}]
        with (
            patch.object(memory.platform, "system", return_value="Linux"),
            patch.object(memory, "_linux_cards", return_value=adapters),
            patch.object(memory, "_openvino_gpu_memory", return_value=(None, None)),
            patch.object(memory, "total_ram_bytes", return_value=32 * memory.GIB),
        ):
            enabled, reasons, _, vram = memory.detect_low_memory(self.make_args())
            self.assertFalse(enabled)
            self.assertEqual(vram, 12 * memory.GIB)
            self.assertEqual(reasons, [])

    def test_openvino_igpu_exposes_shared_memory(self):
        for kind, capacity, expected in (
            ("Type.iGPU", 32 * memory.GIB, (None, 32 * memory.GIB)),
            ("Type.dGPU", 12 * memory.GIB, (12 * memory.GIB, None)),
        ):
            with self.subTest(kind=kind), patch.object(memory.subprocess, "run",
                    return_value=SimpleNamespace(stdout=f"{kind}\n{capacity}\n")):
                self.assertEqual(memory._openvino_gpu_memory("GPU.0"), expected)

    def test_select_openvino_gpu_number_and_vendor(self):
        adapters = [
            {"vendor": 0x8086, "dedicated": 0, "shared": 16 * memory.GIB, "name": "card0"},
            {"vendor": 0x10DE, "dedicated": 8 * memory.GIB, "shared": 16 * memory.GIB, "name": "card1"},
            {"vendor": 0x8086, "dedicated": 12 * memory.GIB, "shared": 16 * memory.GIB, "name": "card2"},
        ]
        self.assertEqual(memory._selected_adapter(adapters, "intel", 0, 0, "GPU.1", None)["dedicated"], 12 * memory.GIB)
        self.assertEqual(memory._selected_adapter(adapters, "intel", 0, 0, "GPU.0", None)["dedicated"], 0)
        self.assertIsNone(memory._selected_adapter(adapters, "ncnn", 0, 0, None, None))

    def test_directml_selects_exact_dxgi_adapter_index(self):
        adapters = [
            {"vendor": 0x8086, "dedicated": 128 * 1024 ** 2, "shared": 16 * memory.GIB, "name": "iGPU", "index": 0},
            {"vendor": 0x10DE, "dedicated": 8 * memory.GIB, "shared": 16 * memory.GIB, "name": "RTX", "index": 2},
        ]
        self.assertEqual(memory._selected_adapter(adapters, "directml", 0, 2, None, None)["dedicated"], 8 * memory.GIB)
        self.assertEqual(memory._selected_adapter(adapters, "directml", 0, 0, None, None)["dedicated"], 128 * 1024 ** 2)
        self.assertIsNone(memory._selected_adapter(adapters, "directml", 0, 1, None, None))

    def test_dedicated_is_preferred_over_shared_without_adding(self):
        cases = [
            (4, 20, 4, "専用VRAM"),
            (8, 16, 8, "専用VRAM"),
            (0, 12, 12, "共有GPUメモリ"),
            (None, 7, 7, "共有GPUメモリ"),
            (0, None, None, None),
        ]
        for dedicated, shared, expected, label in cases:
            with self.subTest(dedicated=dedicated, shared=shared):
                selected, source = memory.select_gpu_memory(
                    None if dedicated is None else dedicated * memory.GIB,
                    None if shared is None else shared * memory.GIB,
                )
                self.assertEqual(selected, None if expected is None else expected * memory.GIB)
                self.assertEqual(source, label)

    def test_linux_integrated_openvino_shared_memory_overrides_unknown_sysfs(self):
        adapters = [{"vendor": 0x8086, "dedicated": None,
                     "shared": None, "name": "card0"}]
        with (
            patch.object(memory.platform, "system", return_value="Linux"),
            patch.object(memory, "_linux_cards", return_value=adapters),
            patch.object(memory, "_openvino_gpu_memory",
                         return_value=(None, 12 * memory.GIB)),
            patch.object(memory, "total_ram_bytes", return_value=32 * memory.GIB),
        ):
            enabled, reasons, _, vram = memory.detect_low_memory(self.make_args())
            self.assertFalse(enabled)
            self.assertEqual(reasons, [])
            self.assertEqual(vram, 12 * memory.GIB)

    def test_windows_dedicated_has_priority_even_when_shared_is_larger(self):
        adapters = [{"vendor": 0x8086, "dedicated": 128 * 1024 ** 2,
                     "shared": 12 * memory.GIB, "name": "HD 530", "index": 0}]
        with (
            patch.object(memory.platform, "system", return_value="Windows"),
            patch.object(memory, "_windows_dxgi_adapters", return_value=adapters),
            patch.object(memory, "_openvino_gpu_memory",
                         return_value=(None, 12 * memory.GIB)),
            patch.object(memory, "total_ram_bytes", return_value=32 * memory.GIB),
        ):
            enabled, reasons, _, vram = memory.detect_low_memory(self.make_args())
            self.assertTrue(enabled)
            self.assertEqual(vram, 128 * 1024 ** 2)
            self.assertIn("専用VRAM", reasons[0])

    def test_cpu_does_not_probe_vram(self):
        with (
            patch.object(memory, "total_ram_bytes", return_value=32 * memory.GIB),
            patch.object(memory, "gpu_memory_capacity") as gpu,
        ):
            enabled, reasons, _, vram = memory.detect_low_memory(
                self.make_args(gpu=False, provider="cpu"))
            self.assertFalse(enabled)
            self.assertEqual(reasons, [])
            self.assertIsNone(vram)
            gpu.assert_not_called()
