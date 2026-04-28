"""Cross-cutting helpers shared by supervision, main_node, and compute_node.

Previously split between ``app/`` and ``common/``. Tier 5 of the 2026-04-22
restructure merged both into ``core/`` so callers have one place to find
constants, config, types, tracing, logging setup, hardware detection,
state enums, message helpers, float32 codec, process-exit classification,
venv relaunch, and work-partition utilities.
"""
