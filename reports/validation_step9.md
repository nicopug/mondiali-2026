# STEP 9 — Talent challenger validation

**Date:** 2026-06-13  
**Challenger:** Tier 2 stack + 2 talent differential features  
**Splits:** train<=2023, val_calib 2023, val_gate 2024-01-01..2024-12-31  

## val_gate 2024 (raw 1X2)

| Model | log-loss | Brier |
|---|---|---|
| v1.4 (XGB-only) | 0.9051 | 0.5328 |
| Challenger (+talent) | 0.9033 | 0.5317 |
| **Delta** | **-0.0018** | -0.0011 |

Gate margin: 0.003. **Verdict: NO DECISION (|delta| < 0.003) — review Brier**

Artefacts: `models/challenger_talent/` (NOT promoted to v1_final).
