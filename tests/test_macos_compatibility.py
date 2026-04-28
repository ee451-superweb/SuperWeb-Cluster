"""macOS and Metal compatibility regression tests."""

from __future__ import annotations

import unittest
from unittest import mock

from core.hardware import collect_hardware_profile
from compute_node.performance_metrics import device_overview
from compute_node.performance_metrics.result_format import _extract_device_name


class MacOsCompatibilityTests(unittest.TestCase):
    """Validate friendly macOS hardware reporting and Metal labeling."""

    def test_detect_cpu_name_uses_shared_processor_name_fallback(self) -> None:
        with (
            mock.patch.object(device_overview, "_run_powershell_json", return_value=None),
            mock.patch.object(device_overview, "detect_processor_name", return_value="Apple M1 Pro"),
        ):
            self.assertEqual(device_overview.detect_cpu_name(), "Apple M1 Pro")

    def test_detect_gpu_devices_parses_macos_system_profiler_inventory(self) -> None:
        payload = {
            "SPDisplaysDataType": [
                {
                    "_name": "Apple M1 Pro",
                    "sppci_model": "Apple M1 Pro",
                    "spdisplays_vendor": "sppci_vendor_Apple",
                    "sppci_cores": "16",
                    "spdisplays_mtlgpufamilysupport": "spdisplays_metal4",
                    "sppci_bus": "spdisplays_builtin",
                }
            ]
        }
        with (
            mock.patch.object(device_overview, "list_windows_display_adapters", return_value=([], "")),
            mock.patch.object(device_overview, "_run_system_profiler_json", return_value=payload),
        ):
            self.assertEqual(
                device_overview.detect_gpu_devices(),
                [
                    {
                        "name": "Apple M1 Pro",
                        "vendor": "Apple",
                        "pnp_device_id": "",
                        "core_count": 16,
                        "metal_family": "spdisplays_metal4",
                        "bus": "spdisplays_builtin",
                    }
                ],
            )

    def test_collect_hardware_profile_uses_friendly_processor_name(self) -> None:
        with (
            mock.patch("core.hardware.network.resolve_local_ip", return_value="10.0.0.2"),
            mock.patch("core.hardware.network.get_local_mac_address", return_value="aa:bb:cc:dd:ee:ff"),
            mock.patch("core.hardware.socket.gethostname", return_value="worker-a"),
            mock.patch("core.hardware.platform.system", return_value="Darwin"),
            mock.patch("core.hardware.platform.release", return_value="25.2.0"),
            mock.patch("core.hardware.platform.machine", return_value="arm64"),
            mock.patch("core.hardware.detect_processor_name", return_value="Apple M1 Pro"),
            mock.patch("core.hardware._detect_total_memory_bytes", return_value=17179869184),
            mock.patch("core.hardware.os.cpu_count", return_value=10),
        ):
            profile = collect_hardware_profile("10.0.0.5", 52020)

        self.assertEqual(profile.processor, "Apple M1 Pro")
        self.assertEqual(profile.machine, "arm64")
        self.assertEqual(profile.logical_cpu_count, 10)

    def test_extract_device_name_prefers_macos_gpu_name_for_generic_metal_label(self) -> None:
        device_name = _extract_device_name(
            "metal",
            {"trial_notes": ["device=metal"], "notes": []},
            "",
            {"gpus": [{"name": "Apple M1 Pro"}]},
        )

        self.assertEqual(device_name, "Apple M1 Pro")


if __name__ == "__main__":
    unittest.main()
