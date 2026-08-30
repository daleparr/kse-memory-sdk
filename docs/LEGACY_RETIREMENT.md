# Legacy Test Retirement Map (T-068, D-16)

D-16: *each FR that lands deletes its mocked counterparts in the same PR;
CI soft-fail lanes flip to hard-fail when Phase 2 completes (T-015).* This
map records what remains, what replaces it, and what triggers each deletion.

Already retired this way: the `ConceptualDimensions` class tests and the
`ConceptualService` integration test died with their subjects; ~70 legacy
tests were healed (not deleted) when the mandatory-LLM-key path was removed.

## Why the lane numbers are what they are

The soft-fail lane's remaining failures are **superseded content, not broken
product code**: fixtures constructing `Product(price=...)` (a retail field
removed by the universal model), configs using the pre-v3 `uri` kwarg shape,
and assertions that the default backend is a cloud service (now wrong by
decision — TC-02 made the default local). Three infrastructure defects that
*masked* this were fixed while drawing this map (see BD6 Session 21-CC):
config no longer demands an LLM key or a Pinecone key on the default path,
and `memory` backends now exist for all three stores.

## The map

| File | Now | Covers | Replaced by | Delete when |
|---|---|---|---|---|
| `tests/test_core.py` | 1F 6P 10E | Mocked KSEMemory CRUD/search; model unit tests | Component pipeline suites; unit model tests | **T-025** (US2 verify). Salvage first: the SearchQuery/Entity model tests are real unit tests — migrate keepers to `tests/unit/` |
| `tests/test_backends.py` | 11F 2P 4E | Mocked backend behaviour; config-shape assertions (old cloud defaults, `uri` kwarg) | Conformance lane (`tests/conformance/`) | Per backend, as each is wired into conformance. The default-backend assertions are already wrong by decision — deletable now |
| `tests/test_integration.py` | 1F 2P 1S 6E | Mocked end-to-end workflows | `tests/integration/` (genuine model, `no_network`) | **T-025** |
| `tests/test_integrations.py` | 1F 4S 3E | LangChain/LlamaIndex adapters (extras-gated) | Extras-gated conformance, with the framework-integration story | That story's TC cycle — not Phase 2 |
| `tests/test_temporal_reasoning.py` | torch-gated skip | v2 temporal subsystem | Its own TC cycle | P3 temporal story |
| `tests/test_federated_learning.py` | torch-gated skip | v2 federated subsystem | Its own TC cycle | P3 federated story |
| `tests/test_comprehensive_benchmark.py` | torch-gated skip | arXiv benchmark harness | `benchmarks/` (US5 — benchmarks are NOT tests) | **US5**, per AR-03 |
| `tests/test_incremental_updates_analysis.py` | torch-gated skip | v2 incremental analysis | FR-01/FR-02 incremental suites (exist) | Deletable at T-025; nothing depends on it |
| `test_universal_model.py` (repo root) | not collected | v2 model smoke script | `tests/unit/` model coverage | **T-025** |

## T-015 hard-fail flip criteria

The concession is temporary by construction. Flip when ALL of:

1. Every file above is deleted or green — the lane reads `0 failed, 0 errors`.
2. `mypy kse_memory` is clean on the public surface — then remove `|| true`
   from the guardrails job's mypy step.
3. Then remove `continue-on-error: true` from the unit job's legacy step.

Gate: criterion 1 held for two consecutive CI runs before flipping 2 and 3.
