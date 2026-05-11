# STEP 5 — Validation Report: Tier 4 (Injury impact via top-5 absence)

**Date:** 2026-05-11
**Status:** ⚠️ **NOT PROMOTED — data unavailable**
**Decision:** Tier 4 code shipped on `feat/step5-tier4` but **not activated**. Tier 2 remains the production baseline (Tier 3 was already not promoted in STEP 4).

---

## 1. What was built

The full Tier 4 stack was implemented and unit-tested across 12 tasks:

| Layer | Module | Status |
|---|---|---|
| Tournament metadata + injuries.csv schema | `src/mondiali/data/tm_rosters.py` | ✅ |
| Transfermarkt roster scraper (player-level) | `src/mondiali/data/tm_rosters.py` | ✅ |
| CLI `tm-scrape-rosters` (4 tournaments × 28 nations) | `src/mondiali/cli/main.py` | ✅ |
| Wikipedia withdrawals HTML parser | `src/mondiali/data/injuries_bootstrap.py` | ✅ (code) / ❌ (data — see §3) |
| Bootstrap orchestrator (strict-match + dedup) | `src/mondiali/data/injuries_bootstrap.py` | ✅ |
| CLI `bootstrap-injuries` + Wikipedia fetcher with cache | `src/mondiali/cli/main.py` | ✅ |
| Tier 4 feature engineering (top-5 absent + value-ratio) | `src/mondiali/features/tier4.py` | ✅ |
| Anti-leakage test (strict `< match_date`) | `tests/test_leakage.py` | ✅ |
| `train_tier4_pipeline` with double Optuna apples-to-apples | `src/mondiali/training/train.py` | ✅ |
| CLI `train-tier4` with gate verdict | `src/mondiali/cli/main.py` | ✅ |

Test coverage: 23 new test cases across the four feature/data/training units. All green. Suite stays green except for `test_train_tier3_full_split_passes_gate` (pre-existing, expected to fail — Tier 3 was not promoted).

## 2. Transfermarkt roster scrape — success

```
python -m mondiali.cli.main tm-scrape-rosters
→ 112/112 (nation, tournament) pairs scraped
→ 4715 player rows
→ 2.1% null market_value (101 rows, mostly young players without TM valuations)
```

Coverage by tournament:
- WC2018: 1476 rows
- Euro2020: 899 rows
- WC2022: 1389 rows
- Euro2024: 951 rows

This part of the pipeline is **fully functional and reusable**.

## 3. Injury bootstrap — Wikipedia structural change

```
python -m mondiali.cli.main bootstrap-injuries
→ wc2018:   added=0  skipped_no_match=0  (n_parsed=0)
→ euro2020: added=0  skipped_no_match=0  (n_parsed=0)
→ wc2022:   added=0  skipped_no_match=0  (n_parsed=0)
→ euro2024: added=0  skipped_no_match=0  (n_parsed=0)
```

**Root cause:** Wikipedia's tournament-squads pages no longer have a global `<h2>Withdrawals</h2>` section. Inspecting the cached HTML:

- WC2018 squads page (1.2 MB): zero occurrences of the word "withdrawal".
- WC2022 squads page (1.5 MB): only 5 inline mentions of "withdrew", embedded as free text inside individual nation tables, alongside the player link (e.g. `Sadio Mané withdrew injured on 17 November`).
- MediaWiki version-bump: the `<span class="mw-headline" id="...">` wrapper was removed in MW 1.39+; section ids now live directly on the `<h2>`. **Zero `mw-headline` occurrences** in any of the 4 cached pages.

The original parser assumed both (1) a global Withdrawals section and (2) the legacy `mw-headline` wrapper. Both assumptions are obsolete.

## 4. Why we stopped here

The spec hard floor was bootstrap ≥50% coverage for Tier 4 to be evaluable. We got 0%. The spec's prescribed escalation path applies:

> "escalate at 8h if bootstrap <50% coverage"

We chose the most conservative branch: **close STEP 5 without promoting Tier 4**, keeping the code path latent. Coherent with the baseline-first invariant — no half-built features in production.

## 5. What's left for a future revival

To activate Tier 4 in the future (e.g. closer to WC2026 kickoff, when live injury data becomes available), the work is:

1. Replace `parse_wikipedia_withdrawals` with a parser for inline patterns like `<a href="...">{player}</a> ... withdrew ... [injured|due to ...]` and infer team from the nearest preceding nation `<h3>`. Estimated: 2-3h.
2. OR: switch source. BBC/ESPN/Soccerway tournament squad pages aggregate ritiri in a more parser-friendly form. Estimated: 1-2h to identify + implement.
3. OR: for WC2026 specifically, the team can manually populate `data/manual/injuries.csv` in the weeks leading up to kickoff. The schema is stable; bootstrap-injuries is idempotent so manual entries survive re-runs.

The feature engineering (`add_tier4_features`), training pipeline (`train_tier4_pipeline`), and CLI (`train-tier4`) are **all production-ready**. They are gated only by `injuries.csv` having rows.

## 6. Artifacts on disk

- `data/raw/transfermarkt/rosters.parquet` — 4715 rows, kept for future use.
- `data/raw/wikipedia/squads_cache/{wc2018,euro2020,wc2022,euro2024}.html` — 4 HTML pages cached, total ~5 MB.
- `data/manual/injuries.csv` — header-only, no rows.
- `models/tier4/` — not created (training pipeline never reached the gate step).

## 7. Decision record

- **Tier 2 remains** the production baseline (set in STEP 3).
- **Tier 3 was NOT promoted** in STEP 4 (log-loss regression).
- **Tier 4 is NOT promoted** in STEP 5 (no training data, per this report).
- **Next step:** STEP 6 (model freeze for WC2026), using Tier 2 as the final model.

The freeze invariant (CLAUDE.md §6: "Dal 11 giugno 2026 al 19 luglio 2026: zero modifiche") still applies. Tier 4 code shipping but inactive does not violate this — no model-altering code path is triggered without injuries.csv rows.
