"""Normalize raw benchmark outputs into one human-friendly schema."""

from __future__ import annotations

import ast
import csv
import re
import shutil
import subprocess
import time
from functools import lru_cache
from typing import Any

from app.constants import METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, METHOD_SPATIAL_CONVOLUTION
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.config import DISPLAY_NAME as FMVM_DISPLAY_NAME
from compute_node.performance_metrics.spatial_convolution.config import DISPLAY_NAME as SPATIAL_DISPLAY_NAME

_DEVICE_NOTE_PATTERN = re.compile(r"device=(.+)")
_QUOTED_DEVICE_PATTERN = re.compile(r"'([^']+)'")
_SEARCH_VALUES_PATTERN = re.compile(r"search order: (?P<values>\[.*\])$")
_SM_NOTE_PATTERN = re.compile(r"\bsm[_=]?(\d+)\b", re.IGNORECASE)
_NVCC_RELEASE_PATTERN = re.compile(r"release\s+(\d+(?:\.\d+)*)", re.IGNORECASE)

_CUDA_ARCHITECTURE_BY_SM = {
    "50": "Maxwell",
    "52": "Maxwell",
    "60": "Pascal",
    "61": "Pascal",
    "70": "Volta",
    "72": "Volta",
    "75": "Turing",
    "80": "Ampere",
    "86": "Ampere",
    "87": "Ampere",
    "88": "Ampere",
    "89": "Ada",
    "90": "Hopper",
    "100": "Blackwell",
    "103": "Blackwell",
    "110": "Blackwell",
    "120": "Blackwell",
    "121": "Blackwell",
}


def _literal_list(text: str) -> list[object]:
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def _compact_note(note: str) -> str | None:
    lowered = note.lower()
    if "cuda backend available" in lowered or "nvcc detected on path" in lowered:
        return None
    if "compiled cuda runner" in lowered or "compiled self-contained windows cuda runner" in lowered:
        return "binary recompiled"
    if "compiled windows dx12 runner" in lowered:
        return "binary recompiled"
    if (
        "cuda" in lowered
        and ("existing binary is older" in lowered or "source or build recipe is newer" in lowered)
        and ("existing binary will be used" in lowered or "existing runner will be used" in lowered)
    ):
        return "binary outdated"
    if (
        "dx12" in lowered
        and ("existing binary is older" in lowered or "source or build recipe is newer" in lowered)
        and ("existing binary will be used" in lowered or "existing runner will be used" in lowered)
    ):
        return "binary outdated"
    if (
        "using prebuilt cuda runner" in lowered
        or "using prebuilt self-contained windows cuda runner" in lowered
        or "using cuda runner at" in lowered
        or ("cuda backend" in lowered and "via prebuilt binary" in lowered)
        or ("cuda backend" in lowered and "via self-contained windows runner" in lowered)
    ):
        return "binary available"
    if "using prebuilt windows dx12 runner" in lowered:
        return "binary available"
    if "fatbin sms:" in lowered or "ptx fallback:" in lowered or "toolkit-limited omissions:" in lowered:
        return None
    replacements = (
        ("using prebuilt", "prebuilt runner"),
        ("cpu runner resolved", "prebuilt runner"),
        ("using cuda runner", "prebuilt runner"),
        ("self-contained", "self-contained binary"),
        ("only available on macos", "macOS only"),
        ("only fp32 accumulation", "fp32 accumulation only"),
        ("autotuned on", "test autotune, runtime measurement"),
        ("autotune on", "test autotune, runtime measurement"),
    )
    for needle, replacement in replacements:
        if needle in lowered:
            return replacement
    if "search order:" in lowered:
        return None
    if "setup_wall_clock_latency_seconds" in lowered or "upload_wall_clock_latency_seconds" in lowered:
        return None
    if "autotune_repeats_per_config" in lowered or "measurement_repeats_for_best_config" in lowered:
        return None
    if lowered.startswith("accumulation_precision="):
        return None
    if lowered.startswith("device=") or lowered.startswith("sm="):
        return None
    trimmed = note.strip()
    return trimmed if len(trimmed) <= 120 else f"{trimmed[:117]}..."


def _compact_notes(*note_lists: list[str]) -> list[str]:
    compacted: list[str] = []
    for notes in note_lists:
        for note in notes:
            compact = _compact_note(str(note))
            if compact and compact not in compacted:
                compacted.append(compact)
    return compacted[:3]


def _cuda_architecture_name(sm_digits: str | None) -> str | None:
    if not sm_digits:
        return None
    if sm_digits in _CUDA_ARCHITECTURE_BY_SM:
        return _CUDA_ARCHITECTURE_BY_SM[sm_digits]
    try:
        numeric = int(sm_digits)
    except ValueError:
        return None
    if numeric >= 100:
        return "Blackwell"
    return None


@lru_cache(maxsize=1)
def _detect_cuda_gpu_inventory() -> list[dict[str, str]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    completed = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []

    rows: list[dict[str, str]] = []
    for row in csv.reader(completed.stdout.splitlines()):
        if not row:
            continue
        name = row[0].strip() if len(row) >= 1 else ""
        compute_cap = row[1].strip() if len(row) >= 2 else ""
        driver_version = row[2].strip() if len(row) >= 3 else ""
        sm_digits = compute_cap.replace(".", "")
        rows.append(
            {
                "name": name,
                "sm_digits": sm_digits if sm_digits.isdigit() else "",
                "driver_version": driver_version,
            }
        )
    return rows


@lru_cache(maxsize=1)
def _detect_nvcc_version() -> str:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return "not detected"

    completed = subprocess.run(
        [nvcc, "--version"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "not detected"

    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    match = _NVCC_RELEASE_PATTERN.search(combined)
    if match:
        return match.group(1)
    return "not detected"


def _extract_sm_digits(raw_backend: dict[str, Any], probe_message: str) -> str | None:
    note_sources = [
        *[str(item) for item in raw_backend.get("trial_notes", [])],
        *[str(item) for item in raw_backend.get("notes", [])],
        probe_message,
    ]
    for note in note_sources:
        match = _SM_NOTE_PATTERN.search(note)
        if match:
            return match.group(1)
    return None


def _cuda_binary_status(raw_backend: dict[str, Any], probe_message: str) -> str:
    note_sources = [probe_message, *[str(item) for item in raw_backend.get("notes", [])]]
    lowered_notes = [note.lower() for note in note_sources]
    if any(
        "compiled cuda runner" in note or "compiled self-contained windows cuda runner" in note
        for note in lowered_notes
    ):
        return "recompiled"
    if any(
        ("existing binary is older" in note or "source or build recipe is newer" in note)
        and ("existing binary will be used" in note or "existing runner will be used" in note)
        for note in lowered_notes
    ):
        return "outdated"
    if any(
        "using prebuilt" in note
        or "using cuda runner at" in note
        or "via prebuilt binary" in note
        or "via self-contained windows runner" in note
        for note in lowered_notes
    ):
        return "available"
    if any("binary is missing" in note for note in lowered_notes):
        return "missing"
    return "not detected"


def _normalize_cuda_environment(
    *,
    device_name: str,
    raw_backend: dict[str, Any],
    probe_message: str,
) -> dict[str, str]:
    gpu_inventory = _detect_cuda_gpu_inventory()
    matched_gpu = next((gpu for gpu in gpu_inventory if gpu.get("name") == device_name), None)
    if matched_gpu is None and gpu_inventory:
        matched_gpu = gpu_inventory[0]

    sm_digits = (
        (matched_gpu or {}).get("sm_digits")
        or _extract_sm_digits(raw_backend, probe_message)
        or ""
    )
    architecture = _cuda_architecture_name(sm_digits)
    return {
        "detected_architecture": architecture or "not detected",
        "sm_version": f"sm_{sm_digits}" if sm_digits else "not detected",
        "driver_version": ((matched_gpu or {}).get("driver_version") or "not detected"),
        "nvcc_version": _detect_nvcc_version(),
        "binary_status": _cuda_binary_status(raw_backend, probe_message),
    }


def _extract_search_space(raw_backend: dict[str, Any], probe_message: str) -> dict[str, list[object]]:
    search_space: dict[str, list[object]] = {}
    for note in [probe_message, *[str(item) for item in raw_backend.get("notes", [])]]:
        lowered = note.lower()
        match = _SEARCH_VALUES_PATTERN.search(note)
        if not match:
            continue
        if "worker search order" in lowered:
            label = "workers"
        elif "tile search order" in lowered or "tile size search order" in lowered:
            label = "tile_size"
        elif "block size search order" in lowered:
            label = "block_size"
        elif "transpose search order" in lowered:
            label = "transpose"
        elif "thread-group-size search order" in lowered:
            label = "thread_group_size"
        elif "rows-per-thread search order" in lowered:
            label = "rows_per_thread"
        else:
            label = "search_space"
        values = _literal_list(match.group("values"))
        if values:
            search_space[label] = values
    return search_space


def _extract_device_name(
    backend_name: str,
    raw_backend: dict[str, Any],
    probe_message: str,
    device_overview: dict[str, Any],
) -> str:
    if backend_name == "cpu":
        return str(((device_overview.get("cpu") or {}).get("name")) or backend_name)

    note_sources = [
        *[str(item) for item in raw_backend.get("trial_notes", [])],
        *[str(item) for item in raw_backend.get("notes", [])],
        probe_message,
    ]
    for note in note_sources:
        match = _DEVICE_NOTE_PATTERN.search(note)
        if match:
            return match.group(1).strip()
        if backend_name in {"cuda", "dx12"} and "adapter" in note.lower():
            quoted = _QUOTED_DEVICE_PATTERN.search(note)
            if quoted:
                return quoted.group(1).strip()
        if backend_name == "cuda" and "available for" in note.lower():
            quoted = _QUOTED_DEVICE_PATTERN.search(note)
            if quoted:
                return quoted.group(1).strip()
    return backend_name


def _normalize_trial(raw_trial: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(raw_trial, dict):
        return None
    return {
        "wall_clock_latency_seconds": float(raw_trial.get("wall_clock_latency_seconds") or 0.0),
        "effective_gflops": float(raw_trial.get("effective_gflops") or 0.0),
        "checksum": str(raw_trial.get("checksum") or ""),
        "score": float(raw_trial.get("score") or 0.0),
    }


def _normalize_backend(
    *,
    backend_name: str,
    raw_backend: dict[str, Any],
    raw_method: dict[str, Any],
    device_overview: dict[str, Any],
    autotune_plan: dict[str, Any],
    measurement_plan: dict[str, Any],
) -> dict[str, Any]:
    probe = raw_method.get("hardware_inventory", {}).get(backend_name, {})
    probe_message = str(probe.get("probe_message") or "")
    best_config = raw_backend.get("best_config") or raw_backend.get("selected_config")
    best_result = raw_backend.get("best_result") or raw_backend.get("best_trial")
    autotune_result = raw_backend.get("autotune_result") or raw_backend.get("autotune_trial")
    device_name = _extract_device_name(backend_name, raw_backend, probe_message, device_overview)
    normalized = {
        "backend": backend_name,
        "available": bool(raw_backend.get("available")),
        "device_name": device_name,
        "rank": raw_backend.get("rank"),
        "autotune_plan": {
            "autotune_repeats": autotune_plan.get("autotune_repeats"),
            "measurement_repeats": measurement_plan.get("measurement_repeats"),
            "trials_run": (best_config or {}).get("trials_run"),
            "search_space": _extract_search_space(raw_backend, probe_message),
        },
        "autotune_result": _normalize_trial(autotune_result),
        "best_config": dict(best_config) if isinstance(best_config, dict) else None,
        "best_result": _normalize_trial(best_result),
        "notes": _compact_notes(
            [probe_message] if probe_message else [],
            [str(item) for item in raw_backend.get("notes", [])],
            [str(item) for item in raw_backend.get("trial_notes", [])],
        ),
    }
    if backend_name == "cuda":
        normalized["cuda_environment"] = _normalize_cuda_environment(
            device_name=device_name,
            raw_backend=raw_backend,
            probe_message=probe_message,
        )
    return normalized


def _normalize_fixed_matrix_vector_dataset(
    raw_method: dict[str, Any], dataset_root: str | None
) -> dict[str, Any]:
    dataset = raw_method.get("dataset", {})
    root_dir = dataset_root or dataset.get("root_dir")
    artifacts = dict(dataset.get("artifacts") or {})
    shape = dict(dataset.get("shape") or {})
    byte_sizes = dict(dataset.get("bytes") or {})
    if not artifacts:
        matrix_name = str(dataset.get("matrix_path") or "").split("/")[-1]
        vector_name = str(dataset.get("vector_path") or "").split("/")[-1]
        artifacts = {
            "autotune_matrix": f"{root_dir}/{matrix_name}" if root_dir and matrix_name else dataset.get("matrix_path"),
            "autotune_vector": f"{root_dir}/{vector_name}" if root_dir and vector_name else dataset.get("vector_path"),
            "measurement_matrix": f"{root_dir}/{matrix_name}" if root_dir and matrix_name else dataset.get("matrix_path"),
            "measurement_vector": f"{root_dir}/{vector_name}" if root_dir and vector_name else dataset.get("vector_path"),
        }
    if not shape:
        shape = {
            "autotune": {
                "rows": dataset.get("rows"),
                "cols": dataset.get("cols"),
            },
            "measurement": {
                "rows": dataset.get("rows"),
                "cols": dataset.get("cols"),
            },
        }
    if not byte_sizes:
        byte_sizes = {
            "autotune_matrix": dataset.get("matrix_bytes"),
            "autotune_vector": dataset.get("vector_bytes"),
            "measurement_matrix": dataset.get("matrix_bytes"),
            "measurement_vector": dataset.get("vector_bytes"),
        }
    if root_dir:
        normalized_artifacts: dict[str, Any] = {}
        for key, value in artifacts.items():
            if value is None:
                normalized_artifacts[key] = None
                continue
            name = str(value).replace("\\", "/").split("/")[-1]
            normalized_artifacts[key] = f"{root_dir}/{name}"
        artifacts = normalized_artifacts
    return {
        "root_dir": root_dir,
        "artifacts": artifacts,
        "shape": shape,
        "bytes": byte_sizes,
        "generated": bool(dataset.get("dataset_was_generated")),
    }


def _normalize_spatial_dataset(raw_method: dict[str, Any], dataset_root: str | None) -> dict[str, Any]:
    workload = raw_method.get("workload", {})
    full_runtime = bool(workload.get("full_runtime_measurement"))
    artifacts = {
        "autotune_input": f"{dataset_root}/test_input.bin" if dataset_root else None,
        "autotune_weight": f"{dataset_root}/test_weight.bin" if dataset_root else None,
    }
    if full_runtime:
        artifacts["measurement_input"] = f"{dataset_root}/runtime_input.bin" if dataset_root else None
        artifacts["measurement_weight"] = f"{dataset_root}/runtime_weight.bin" if dataset_root else None
    else:
        artifacts["measurement_input"] = artifacts["autotune_input"]
        artifacts["measurement_weight"] = artifacts["autotune_weight"]
    return {
        "root_dir": dataset_root,
        "artifacts": artifacts,
    }


def normalize_method_report(
    *,
    method_name: str,
    raw_method: dict[str, Any],
    dataset_root: str | None,
    device_overview: dict[str, Any],
) -> dict[str, Any]:
    if method_name == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION:
        display_name = FMVM_DISPLAY_NAME
        dataset = _normalize_fixed_matrix_vector_dataset(raw_method, dataset_root)
        raw_workload = raw_method.get("workload", {})
        autotune_shape = dict(dataset.get("shape", {}).get("autotune") or {})
        measurement_shape = dict(dataset.get("shape", {}).get("measurement") or {})
        autotune_plan = {
            **dict(raw_workload.get("autotune") or {}),
            "name": raw_workload.get("autotune", {}).get("name")
            or f"fmvm-{autotune_shape.get('rows')}x{autotune_shape.get('cols')}",
            "rows": autotune_shape.get("rows"),
            "cols": autotune_shape.get("cols"),
            "autotune_repeats": raw_workload.get("autotune_repeats"),
            "selection_metric": raw_workload.get("selection_metric"),
            "input_dtype": raw_workload.get("input_dtype"),
            "output_dtype": raw_workload.get("output_dtype"),
            "accumulation_precision": raw_workload.get("accumulation_precision"),
        }
        measurement_plan = {
            **dict(raw_workload.get("measurement") or {}),
            "name": raw_workload.get("measurement", {}).get("name")
            or f"fmvm-{measurement_shape.get('rows')}x{measurement_shape.get('cols')}",
            "rows": measurement_shape.get("rows"),
            "cols": measurement_shape.get("cols"),
            "measurement_repeats": raw_workload.get("measurement_repeats"),
            "reported_metric": raw_workload.get("reported_metric"),
            "cross_backend_validation": raw_workload.get("cross_backend_validation"),
            "full_runtime_measurement": bool(raw_workload.get("full_runtime_measurement")),
        }
    else:
        display_name = SPATIAL_DISPLAY_NAME
        dataset = _normalize_spatial_dataset(raw_method, dataset_root)
        raw_workload = raw_method.get("workload", {})
        autotune_plan = {
            **dict(raw_workload.get("autotune") or {}),
            "autotune_repeats": raw_workload.get("autotune_repeats"),
        }
        measurement_plan = {
            **dict(raw_workload.get("measurement") or {}),
            "measurement_repeats": raw_workload.get("measurement_repeats"),
            "full_runtime_measurement": bool(raw_workload.get("full_runtime_measurement")),
        }

    raw_backends = (
        raw_method.get("backends")
        if isinstance(raw_method.get("backends"), dict)
        else raw_method.get("backend_results", {})
    )
    normalized_backends = {
        backend_name: _normalize_backend(
            backend_name=backend_name,
            raw_backend=raw_backend,
            raw_method=raw_method,
            device_overview=device_overview,
            autotune_plan=autotune_plan,
            measurement_plan=measurement_plan,
        )
        for backend_name, raw_backend in raw_backends.items()
        if isinstance(raw_backend, dict)
    }

    return {
        "method": method_name,
        "display_name": display_name,
        "generated_at_unix": float(raw_method.get("generated_at_unix") or time.time()),
        "benchmark_elapsed_seconds": float(raw_method.get("benchmark_elapsed_seconds") or 0.0),
        "dataset": dataset,
        "workload": {
            "autotune_plan": autotune_plan,
            "measurement_plan": measurement_plan,
        },
        "backends_considered": list(raw_method.get("backends_considered") or raw_backends.keys()),
        "detected_backends": list(raw_method.get("detected_backends") or []),
        "usable_backends": list(raw_method.get("usable_backends") or []),
        "ranking": list(raw_method.get("ranking") or []),
        "best_backend": raw_method.get("best_backend"),
        "backends": normalized_backends,
    }


def build_report(
    *,
    method_reports: dict[str, dict[str, Any]],
    device_overview: dict[str, Any],
    total_elapsed: float,
) -> dict[str, Any]:
    return {
        "schema_version": 5,
        "generated_at_unix": time.time(),
        "benchmark_elapsed_seconds": total_elapsed,
        "device_overview": device_overview,
        "methods": method_reports,
    }
