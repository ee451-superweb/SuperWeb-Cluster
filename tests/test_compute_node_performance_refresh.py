"""Idle-refresh specific tests for compute-node performance refresh helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from compute_node.performance_refresh import (
    _load_best_backend_config,
    validate_idle_refresh_requirements,
)
from common.types import ComputeHardwarePerformance, ComputePerformanceSummary, MethodPerformanceSummary
from compute_node.runtime_services import IdlePerformanceRefreshService, format_compute_performance_summary
from app.constants import METHOD_GEMV, METHOD_CONV2D


class ComputeNodePerformanceRefreshTests(unittest.TestCase):
    """Verify idle refresh can reuse current benchmark schema and retry sanely."""

    def test_load_best_backend_config_accepts_backends_schema(self) -> None:
        backend_name, best_config = _load_best_backend_config(
            {
                "best_backend": "cuda",
                "backends": {
                    "cuda": {
                        "best_config": {
                            "block_size": 512,
                            "tile_size": 16,
                        }
                    }
                },
            }
        )

        self.assertEqual(backend_name, "cuda")
        self.assertEqual(best_config["block_size"], 512)

    def test_idle_refresh_failure_updates_retry_timestamp(self) -> None:
        logger = mock.Mock()
        service = IdlePerformanceRefreshService(
            config=SimpleNamespace(idle_worker_update_interval=60.0),
            logger=logger,
            node_name="worker-a",
        )
        refresh_future = mock.Mock()
        refresh_future.done.return_value = True
        refresh_future.result.side_effect = ValueError("benchmark result is missing backends")

        with mock.patch("compute_node.runtime_services.time.monotonic", return_value=123.0):
            next_future, last_worker_update_at = service.advance(
                session=mock.Mock(),
                assigned_node_id="worker-1",
                refresh_thread_pool=None,
                refresh_future=refresh_future,
                last_worker_update_at=0.0,
                has_active_or_pending_tasks=False,
                refresh_callable=mock.Mock(),
            )

        self.assertIsNone(next_future)
        self.assertEqual(last_worker_update_at, 123.0)
        logger.warning.assert_called_once()

    def test_format_compute_performance_summary_lists_method_gflops(self) -> None:
        summary = ComputePerformanceSummary(
            method_summaries=[
                MethodPerformanceSummary(
                    method=METHOD_GEMV,
                    ranked_hardware=[ComputeHardwarePerformance(hardware_type="cpu", effective_gflops=24.0, rank=1)],
                ),
                MethodPerformanceSummary(
                    method=METHOD_CONV2D,
                    ranked_hardware=[ComputeHardwarePerformance(hardware_type="cuda", effective_gflops=125.0, rank=1)],
                ),
            ]
        )

        formatted = format_compute_performance_summary(summary)

        self.assertIn("gemv=cpu:24.000GFLOPS", formatted)
        self.assertIn("conv2d=cuda:125.000GFLOPS", formatted)

    @mock.patch("compute_node.runtime_services.write_audit_event")
    def test_idle_refresh_success_logs_method_gflops(self, audit_mock: mock.Mock) -> None:
        logger = mock.Mock()
        service = IdlePerformanceRefreshService(
            config=SimpleNamespace(idle_worker_update_interval=60.0),
            logger=logger,
            node_name="worker-a",
        )
        refreshed_performance = ComputePerformanceSummary(
            method_summaries=[
                MethodPerformanceSummary(
                    method=METHOD_GEMV,
                    ranked_hardware=[ComputeHardwarePerformance(hardware_type="cpu", effective_gflops=24.0, rank=1)],
                ),
                MethodPerformanceSummary(
                    method=METHOD_CONV2D,
                    ranked_hardware=[ComputeHardwarePerformance(hardware_type="cuda", effective_gflops=125.0, rank=1)],
                ),
            ]
        )
        refresh_future = mock.Mock()
        refresh_future.done.return_value = True
        refresh_future.result.return_value = refreshed_performance
        session = mock.Mock()

        with mock.patch("compute_node.runtime_services.time.monotonic", return_value=123.0):
            service.advance(
                session=session,
                assigned_node_id="worker-1",
                refresh_thread_pool=None,
                refresh_future=refresh_future,
                last_worker_update_at=0.0,
                has_active_or_pending_tasks=False,
                refresh_callable=mock.Mock(),
            )

        sent_update = session.send.call_args.args[0]
        self.assertEqual(sent_update.kind.name, "WORKER_UPDATE")
        logged = " ".join(str(call.args[0]) for call in audit_mock.call_args_list)
        self.assertIn("gemv=cpu:24.000GFLOPS", logged)
        self.assertIn("conv2d=cuda:125.000GFLOPS", logged)

    def test_validate_idle_refresh_requirements_reports_progress_steps(self) -> None:
        progress_messages: list[tuple[int, int, str]] = []
        payload = {
            "methods": {
                "gemv": {
                    "best_backend": "cuda",
                    "backends": {"cuda": {"best_config": {"block_size": 128}}},
                },
                "conv2d": {
                    "best_backend": "cpu",
                    "backends": {"cpu": {"best_config": {"tile_size": 16}}},
                },
            }
        }

        with (
            mock.patch("compute_node.performance_refresh._load_result_payload", return_value=payload),
            mock.patch("compute_node.performance_refresh._ensure_gemv_refresh_dataset"),
            mock.patch("compute_node.performance_refresh._ensure_gemv_refresh_runner"),
            mock.patch("compute_node.performance_refresh._ensure_conv2d_refresh_dataset"),
            mock.patch("compute_node.performance_refresh._ensure_conv2d_refresh_runner"),
        ):
            validate_idle_refresh_requirements(
                progress_callback=lambda step, total, description: progress_messages.append(
                    (step, total, description)
                )
            )

        self.assertEqual(len(progress_messages), 7)
        self.assertEqual(progress_messages[0][0:2], (1, 7))
        self.assertIn("Loading persisted benchmark result metadata.", progress_messages[0][2])
        self.assertIn("gemv small input matrix", progress_messages[2][2])
        self.assertIn("conv2d runner binary", progress_messages[-1][2])


if __name__ == "__main__":
    unittest.main()
