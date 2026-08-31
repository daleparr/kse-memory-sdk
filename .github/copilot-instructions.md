# Copilot Instructions · KSE Memory SDK v3 (SOL-01)

Before suggesting code, align to the IDs in scope for the open BD8 task.

## Session start protocol
1. Read `BD_INDEX.md` (this repo's document map + checksums).
2. Read `BD3_Agent_Spec_Sheet.md` — **ground truth; BD3 wins on any conflict**.
3. Read `BD0_Project_Constitution.md` — GOV-01..03 are NON-NEGOTIABLE.
4. Check `BD5_Decision_Log.md` for decisions since your last session; check `BD8` for the next unchecked task.

## Test-first rule (GOV-04)
1. Write the TC-XX test first (see BD4). 2. Verify it FAILS (red). 3. Implement the FR-XX/AR-XX. 4. Verify PASS (green). 5. Mark both `[X]` in BD8.

## Ground truth hierarchy
BD3 > BD2 > BD1 > this file. If code contradicts BD3, the code is wrong or BD3 needs a logged BD5 decision — never silently diverge.

## Hard rules
- Zero network calls on the default path; no API key in quickstart (AR-01). No CUDA deps (AR-04).
- Never present simulated numbers as empirical; simulations/ must not be imported by benchmarks/ (AR-02, GOV-01).
- No TBD guardrails exist currently; if one appears, do NOT implement it until a D-XX decision lands in BD5.

## ID reference
FR = flow steps · AR = guardrails · EC = edge cases · TC = test cases (BD4) · T = tasks (BD8) · D = decisions (BD5) · GOV = constitution (BD0) · US = stories.

## Project state (2026-08-29)
Phase 0 — credibility triage (pre-build). Stack: Python SDK v2.0.0, ~21k LOC. Known defects: naive linear fusion (→RRF, FR-05); hardcoded fashion dims (→schema, FR-03); domain_mapping identity stub; deprecated openai API; simulated benchmark claims to retract (US1). Pending: written EY sanction record.

## Session end protocol
Append a BD6 journal entry; propagate any spec change to BD3 first; update BD8 checkboxes; note new decisions in BD5.
