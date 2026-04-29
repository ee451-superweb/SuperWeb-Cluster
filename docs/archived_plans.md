# Archived planning documents

This directory used to contain a number of pre-implementation planning
notes, sprint plans, and a hand-maintained file-system tree. All of them
described work that has since shipped (or, in one case, been overcome by
a subsequent restructure), and several already opened with self-warnings
that their file paths and module names had drifted. They were removed
from the working tree on 2026-04-28 to keep `docs/` focused on documents
that still describe the current system or open work. Full text of each
removed document is preserved in git history if a reader needs to
reconstruct the original reasoning.

The summary below records what each removed document covered and where
the resulting capability lives in the codebase today.

| Removed file | What it covered | Where the capability lives now |
|---|---|---|
| `implementation_plan.txt` | Sprint 1 archived implementation outline (OS detection, mDNS, registration, first end-to-end compute path). Self-archived note on top. | Sprint 1 deliverables list in [`technical-detail.md`](technical-detail.md). Code under [`adapters/`](../adapters/), [`discovery/`](../discovery/), [`wire/`](../wire/), [`main_node/`](../main_node/), [`compute_node/`](../compute_node/). |
| `implementation_plan_2026-04-15.txt` | Multi-method integration plan that triggered the `compute_methods/` split and method-aware benchmarking. Self-warning that referenced files like `wire/runtime.py` no longer exist. | Sprint 2 / Sprint 3 deliverables in [`technical-detail.md`](technical-detail.md). Method tree under [`compute_node/compute_methods/`](../compute_node/compute_methods/), benchmark stack under [`compute_node/performance_metrics/`](../compute_node/performance_metrics/). |
| `sprint3_plan_2026-04-15.txt` | Sprint 3 plan: `conv2d` end-to-end, separate data plane, WinUI3 + iOS frontends. Self-warning on stale paths. | Sprint 3 deliverables list in [`technical-detail.md`](technical-detail.md). Conv2d code in [`compute_node/compute_methods/conv2d/`](../compute_node/compute_methods/conv2d/), data plane in [`wire/internal_protocol/`](../wire/internal_protocol/) and [`transport/`](../transport/). |
| `large_data_transfer_plan_2026-04-15.txt` | Motivation and design for splitting a separate TCP data plane out of the control plane. Self-warning on renamed files. | Data-plane port `:52021` is in production. Implementation in [`wire/internal_protocol/`](../wire/internal_protocol/) and [`transport/`](../transport/). Mentioned in the README architecture sketch. |
| `2026-04-18_implementation_plan.txt` | Plan for adding per-slice timing diagnostics (compute / fetch / peripheral milliseconds) to the `CLIENT_RESPONSE` payload. | Shipped in Sprint 4. Listed under "Per-slice timing diagnostics" in the README and in the `CLIENT_RESPONSE` message catalog in [`technical-detail.md`](technical-detail.md). |
| `2026-04-19_data_plane_deliver_protocol.txt` | DELIVER frame format and the switch from a pull-based artifact model to a push-based one for `conv2d` weight upload. | Implemented and in production on the `:52021` data plane. The DELIVER frame, weight upload, and full-output download paths are described in [`technical-detail.md`](technical-detail.md). |
| `2026-04-19_full_protocol_walkthrough.txt` | Three-party (Client ↔ Main ↔ Worker) protocol walkthrough across discovery, control plane, and data plane. | Translated and re-tracked as [`2026-04-19_full_protocol_walkthrough.md`](2026-04-19_full_protocol_walkthrough.md). The `.txt` was replaced rather than removed in spirit. |
| `2026-04-22_implementation_plan.txt` | Directory and naming restructure plan that produced the current `core/`, `supervision/`, `adapters/` split and removed Sprint 1 placeholders such as `main_node/heartbeat.py`. | Restructure shipped before the GEMM Phase 2 work landed on top. Current layout is documented in the README. |
| `file_system_tree.txt` | Hand-maintained file-system tree with one-line descriptions per file. Stale: referenced `app/` and `standalone_model/`, both of which have been renamed or relocated. | Replaced by the more compact "Repository layout" section in the README. |

If a future contributor needs the original plan text, `git log
-- docs/<filename>` followed by `git show <commit>:docs/<filename>` will
recover it. Nothing in the current code paths depends on these files.
