# Design: STEP 5 — Tier 4 Injuries

**Data**: 2026-05-10
**Autore**: Nicolò (con brainstorming assistito)
**Stato**: Spec approvata, pronto per writing-plans
**Riferimento**: `2026-04-20-mondiali-prediction-design.md` §5 (Tier 4) e §9 (STEP 5)
**Predecessore**: STEP 4 chiuso (`reports/validation_step4.md`) — Tier 3 NON promosso, baseline produzione = Tier 1+2.

---

## 1. Executive Summary

STEP 5 introduce il **Tier 4 — Infortuni** come ultimo blocco feature di Fase A. L'obiettivo è misurare onestamente, via doppio Optuna apples-to-apples, se l'aggiunta delle 4 feature di assenza top-5 (count + value-ratio per lato) supera il gate di +0.003 log-loss su val_gate WC2022 rispetto a un Tier 1+2 ri-tunato con stesso budget.

**Contesto post-STEP 4**: Tier 3 (market value totale nazione) è stato bocciato come predittore diretto. Tuttavia Tier 4 richiede strutturalmente i player-level market values per identificare i top-5 di ciascuna rosa. Decisione: **Tier 3 sopravvive come *enabler* di Tier 4**, non come feature diretta. Le 6 colonne di Tier 3 NON entrano nel modello (rimangono escluse). Player-level rosters (`rosters.parquet`) servono solo per identificare top-5.

**Bootstrap onesto**: per WC2018, Euro2020, WC2022, Euro2024 scrappiamo le rose torneo da Transfermarkt (player-level) + parsiamo gli articoli Wikipedia "X squads" per estrarre pre-tournament withdrawals. Coverage attesa onesta: 30-50% delle assenze top-5. Pre-2018: feature NaN. Match fuori torneo (qualifiche, amichevoli): NaN.

**Output di gate**:
- Se challenger batte baseline di ≥0.003 log-loss → Tier 4 promosso, tier finali per FREEZE = 1+2+4.
- Altrimenti → Tier 4 NON promosso, tier finali = 1+2 (stato corrente).
- Decisione documentata in `reports/validation_step5.md`.

## 2. Objectives

**IN SCOPE**:
1. Schema injuries.csv esteso (con `tournament` + `source`) e validazione enum.
2. Player-level rosters parquet per 4 tornei storici (~2.5-3k righe) via Transfermarkt.
3. Bootstrap automatico injuries.csv da Wikipedia squads pages (best-effort, no fuzzy match silenzioso).
4. Modulo `features/tier4.py` con 4 colonne, anti-leakage stretto.
5. Comando CLI `mondiali train-tier4` con doppio Optuna study (100 trial cad.).
6. Report `validation_step5.md` con confronto e decisione tier-gate finale.
7. Suite test ≥21 nuovi casi (scraper, bootstrap, feature, leakage, training smoke).

**OUT OF SCOPE (YAGNI)**:
- Fuzzy matching player_name (troppo rischio falsi positivi).
- News scraping per injuries (rinviato a STEP 9 / Tier 5 opzionale).
- Bootstrap di tornei diversi dai 4 elencati (Copa America, AFC Asian Cup, ecc.).
- Fallback automatico Wikipedia → newspaper se Wikipedia parsing fallisce (manuale top-up se serve).
- LSTM o ensemble (Fase B).

## 3. Architettura

### 3.1 Nuovi moduli

```
src/mondiali/
├── data/
│   ├── tm_rosters.py           # NUOVO: scraper rose player-level torneo (riusa cache fast-path)
│   └── injuries_bootstrap.py   # NUOVO: parser Wikipedia tournament squads → injuries.csv
├── features/
│   └── tier4.py                # NUOVO: add_tier4_features (count + value_ratio)
└── cli/main.py                 # +3 comandi: tm-scrape-rosters, bootstrap-injuries, train-tier4
data/
├── raw/
│   ├── transfermarkt/
│   │   ├── rosters/                      # NUOVO: cache HTML rose torneo
│   │   └── rosters.parquet               # NUOVO: player-level
│   └── wikipedia/
│       └── squads_cache/                 # NUOVO: HTML cache Wikipedia squads
├── manual/
│   └── injuries.csv                      # NUOVO: schema spec esteso
reports/
└── validation_step5.md                   # NUOVO: gate report
models/
└── tier4/                                # NUOVO: modelli + params + calibrator
    ├── xgb_poisson.json
    ├── calibrator.json
    ├── baseline_params.json
    └── challenger_params.json
```

### 3.2 Razionale separazione `tm_rosters.py` da `transfermarkt.py`

La rosa torneo (player-level, snapshot puntuale al kickoff) è entità distinta dallo snapshot annuale per nazione (totale aggregato). Mescolare i due rende `transfermarkt.py` non più focalizzato. Riusiamo le primitive comuni (cache fast-path `_scan_cache_for_slug`, `_CACHE_FILE_RE`, helpers BeautifulSoup, rate limiter) via import.

## 4. Schema dati

### 4.1 `data/raw/transfermarkt/rosters.parquet`

| col | type | note |
|---|---|---|
| `nation` | str | normalizzato come in `snapshots.parquet` |
| `tournament` | str | enum: `wc2018` \| `euro2020` \| `wc2022` \| `euro2024` |
| `tournament_start_date` | date | data inizio torneo (used per anti-leakage) |
| `player_name` | str | nome così come riportato da TM |
| `player_url_slug` | str | slug univoco TM (per disambiguare omonimi) |
| `position` | str | `GK` \| `DEF` \| `MID` \| `FWD` |
| `market_value_eur` | int64 (nullable) | null se TM non lo riporta |

Volume: 4 tornei × 24-32 nazioni × 23-26 player ≈ **2.5-3k righe**.

URL pattern TM: `transfermarkt.com/{slug}/kader/verein/{tm_id}/saison_id/{anno-1}/plus/1`
- `anno-1` perché TM usa l'anno di inizio stagione (es. WC2018 → saison 2017/18).

### 4.2 `data/manual/injuries.csv`

Schema:
```csv
date_of_knowledge,team,tournament,player_name,player_url_slug,market_value_eur,status,source
2018-06-12,Spain,wc2018,Dani Carvajal,carvajal,32000000,out,wikipedia_squads
2022-11-19,France,wc2022,Karim Benzema,benzema,25000000,out,wikipedia_squads
2026-06-08,France,wc2026,Kylian Mbappé,mbappe,180000000,out,manual
```

- `status ∈ {out, doubtful, available}`. `available` non viene scritto (assenza implicita = available); il dominio è esplicitato per parser robusto.
- `source ∈ {wikipedia_squads, manual}` per filtrare provenienza in eventuali analisi future.
- `tournament` consente lookup rapido cross-roster.

### 4.3 Tournament metadata (constant)

In `tm_rosters.py`:
```python
TOURNAMENT_META = {
    "wc2018":   {"start": "2018-06-14", "end": "2018-07-15", "saison_id": 2017, "n_participants": 32},
    "euro2020": {"start": "2021-06-11", "end": "2021-07-11", "saison_id": 2020, "n_participants": 24},
    "wc2022":   {"start": "2022-11-20", "end": "2022-12-18", "saison_id": 2022, "n_participants": 32},
    "euro2024": {"start": "2024-06-14", "end": "2024-07-14", "saison_id": 2023, "n_participants": 24},
}

TOURNAMENT_PARTICIPANTS: dict[str, list[str]] = {
    "wc2018":   [...],  # 32 nation strings hardcoded (Russia, Saudi Arabia, Egypt, Uruguay, ...)
    "euro2020": [...],  # 24 nation strings
    "wc2022":   [...],  # 32 nation strings
    "euro2024": [...],  # 24 nation strings
}
```

Le liste hardcoded usano nation strings normalizzate come in `tier3_scope.json`. Totale: 32+24+32+24 = 112 entries da scrappare.

## 5. Bootstrap pipeline

### 5.1 Fase A — Scrape rose torneo (Transfermarkt)

Comando: `mondiali tm-scrape-rosters [--tournament wc2018,...] [--resume]`.

Per ogni `(nation, tournament)` in `TOURNAMENT_META[t]["participants"]`:
1. Costruisci URL TM con `tm_id` da `NATION_TM_IDS` (già verificato in STEP 4).
2. **Cache fast-path**: se `data/raw/transfermarkt/rosters/{slug}__{tournament}.html` esiste → parse, skip network.
3. Altrimenti GET con UA + rate limit (≥1.5s/req, come `transfermarkt.py`). Salva HTML in cache.
4. Parser (BeautifulSoup): tabella `kader` → estrai `(player_name, slug, position, market_value_eur)`.
5. Append a `rosters.parquet` (atomic write via tmpfile + rename).

Idempotente: `--resume` (default ON) skippa entries già in parquet.

**Tempo macchina stimato**: ~20-30 min con cache miss totale (rate limit), <2 min se la cache è già piena.

### 5.2 Fase B — Bootstrap injuries da Wikipedia

Comando: `mondiali bootstrap-injuries [--tournament wc2018,...]`.

Per ogni torneo:
1. URL Wikipedia: `en.wikipedia.org/wiki/{Year}_FIFA_World_Cup_squads` (e analogo Euro).
2. Cache HTML in `data/raw/wikipedia/squads_cache/{tournament}.html`.
3. Parser cerca sezioni note:
   - `## Withdrawals` / `## Replacements` / `## Pre-tournament withdrawals`
   - Pattern frase: `"X was originally selected but was replaced by Y"`, `"X withdrew due to injury"`, etc.
4. Per ogni candidate withdrawal:
   - Match `player_name` esatto (case-insensitive, accent-stripped) contro `rosters.parquet[tournament == t]`.
   - Se no match → log warning `injury_player_no_roster_match`, **skip entry** (no fuzzy).
   - Se match → recupera `player_url_slug` + `market_value_eur` dal roster.
   - `date_of_knowledge` = `tournament_start_date - 1 day`.
   - `status` = `out` (Wikipedia withdrawals sono sempre out, non doubtful).
   - `source` = `wikipedia_squads`.
5. Append a `injuries.csv`. Dedup su `(team, tournament, player_url_slug)` (idempotente).

**Coverage onesta attesa**: 30-50% delle assenze top-5 effettive. Wikipedia non è esaustiva.

### 5.3 Logging strutturato

Stesso pattern di STEP 4 (`structlog`):
- `roster_scrape_complete tournament=wc2018 n_nations=32 n_players=736 cache_hits=...`
- `injuries_bootstrap_complete tournament=wc2022 n_withdrawals_parsed=18 n_matched=11 n_skipped_no_match=7`

## 6. Feature engineering — Tier 4

`src/mondiali/features/tier4.py` → `add_tier4_features(matches, rosters, injuries) -> pd.DataFrame`.

### 6.1 Colonne aggiunte (4)

| col | calcolo |
|---|---|
| `home_top5_absent_count` | quanti dei top-5 valore sono `out` o `doubtful` alla data del match |
| `away_top5_absent_count` | idem |
| `home_value_absent_ratio` | Σ market_value assenti / Σ market_value top-5 totale |
| `away_value_absent_ratio` | idem |

Costanti (in `tier4.py`):
```python
TIER4_TOP_N = 5
TIER4_MIN_YEAR = 2018  # primo torneo bootstrappato
TIER4_TOURNAMENT_GRACE_DAYS = 30  # match dopo end ancora considerati nel torneo
```

### 6.2 Algoritmo per ogni `(match, side)`

1. **Identifica torneo applicabile**: cerca in `rosters` la `(nation, tournament)` con `tournament_start_date ≤ match.date < tournament_end_date + GRACE_DAYS`. Se nessuno → 4 feature `NaN`, return.
2. **Top-5 valore**: `rosters[(nation, tournament)].nlargest(5, "market_value_eur")`. Tie-break: `player_url_slug` ascendente (deterministico).
3. **Filtra injuries**: `injuries[(team==nation) & (tournament==same) & (date_of_knowledge < match.date) & (status in {out, doubtful})]`.
4. **Intersezione**: by `player_url_slug` (univoco). Calcola `count` e `sum(market_value)`.
5. **Output**:
   - `count` = numero entries in intersezione.
   - `value_ratio` = `sum_absent_value / sum_top5_value` se `sum_top5_value > 0` else `NaN`.

### 6.3 Anti-leakage

- `date_of_knowledge < match.date` (strict, no `<=`).
- `tournament_start_date ≤ match.date`: la rosa convocata si conosce ~1-2 settimane prima del kickoff. Lo slack di +14 giorni è hard floor accettato per semplicità.
- Pre-2018: tutte le 4 feature `NaN` (no rose disponibili pre-WC2018).
- Match fuori torneo (qualifiche, amichevoli): `NaN`.

### 6.4 Coverage stimata

Su `matches.parquet` (49 215 righe): WC2018=64 + Euro2020=51 + WC2022=64 + Euro2024=51 = **230 match coperti**. Coverage globale ≈ **0.5%**, ma val_gate (WC2022 = 64 match) ha coverage **100%**. Questo è il punto: il gate si gioca dove le feature esistono.

## 7. Training & gate protocol

### 7.1 Comando CLI

`mondiali train-tier4 [--n-trials 100] [--seed 42]`.

Esegue **doppio Optuna study apples-to-apples** su stesso split temporale:
- Train: `2002-01-01 → 2021-12-31`
- Val_gate: `2022-01-01 → 2022-12-31` (include WC2022, coerente con STEP 4)
- Calibration set: `2021-01-01 → 2021-12-31`

### 7.2 Studies

**Baseline study** (Tier 1+2):
- Stack: XGBoost `count:poisson` + Dixon-Coles + isotonic.
- Search space (vedi §7.3), `random_state=42`, 100 trial.
- Best params → `models/tier4/baseline_params.json`.

**Challenger study** (Tier 1+2+4):
- Stesso stack, stesso split, stesso `random_state`, stesso budget, stessa search space.
- 4 colonne aggiuntive di Tier 4. Tier 3 (6 colonne) escluso.
- Best params → `models/tier4/challenger_params.json`.
- Modello + calibratore salvati in `models/tier4/xgb_poisson.json` + `calibrator.json`.

### 7.3 Search space

| param | range |
|---|---|
| `max_depth` | int [3, 8] |
| `learning_rate` | loguniform [0.01, 0.3] |
| `n_estimators` | int [200, 2000] |
| `min_child_weight` | int [1, 10] |
| `subsample` | uniform [0.6, 1.0] |
| `colsample_bytree` | uniform [0.6, 1.0] |
| `reg_alpha` | loguniform [1e-3, 10] |
| `reg_lambda` | loguniform [1e-3, 10] |

Obiettivo: minimizza log-loss 1X2 sul val_gate (post-Dixon-Coles, post-isotonic fittato su 2021).

### 7.4 Decision gate

| Esito | Δ = challenger - baseline | Azione |
|---|---|---|
| Challenger meglio | `Δ ≤ -0.003` | **Promuovi Tier 4**. Tier finali per FREEZE = 1+2+4. |
| No-decisione | `-0.003 < Δ < 0.003` | **Tie-breaker Brier**. Se nemmeno Brier decide → mantieni baseline (parsimonia, baseline-first). |
| Challenger peggio | `Δ ≥ 0.003` | **NON promuovere**. Tier finali = 1+2. |

Coerenza con STEP 4: stessa logica di gate (`Δ ≥ 0.003` come da Tier System §5).

### 7.5 Report

`reports/validation_step5.md` deve includere:
1. Riepilogo bootstrap (rose scrappate, injuries parsati, coverage).
2. Tabella metriche: baseline (log-loss + Brier), challenger (idem), Δ.
3. Best params di entrambi gli studies.
4. Decisione esplicita + razionale.
5. SHAP top-10 features del challenger (sanity check, no overfitting su 4 nuove colonne).
6. Coerenza con baseline-first invariant.

## 8. Anti-leakage tests

`tests/test_leakage.py` esteso con:
- `test_tier4_strict_pre_match`: per ogni riga non-NaN di Tier 4 features, asserisce `injuries.date_of_knowledge < matches.date` per ogni player contato.
- Suite esistente (5 test) deve continuare a passare.

## 9. Testing strategy

**Test-first** (CLAUDE.md invariant 1+2). 21 nuovi casi.

### `tests/test_tm_rosters.py` (~6)
- `test_parse_roster_html_extracts_players`
- `test_roster_url_pattern_correct` (saison_id = year-1)
- `test_cache_fast_path_skips_network`
- `test_resume_skips_already_done`
- `test_omonimi_disambiguati_via_slug`
- `test_market_value_parser_handles_em_billions` (`€100,00m`, `€1,20bn`)

### `tests/test_injuries_bootstrap.py` (~5)
- `test_parse_wikipedia_withdrawals_section`
- `test_no_match_with_roster_logs_warning_skips`
- `test_idempotent_run_no_duplicates`
- `test_status_enum_validated`
- `test_date_of_knowledge_default_pre_kickoff`

### `tests/test_tier4.py` (~7)
- `test_top5_identification_correct`
- `test_absent_count_excludes_status_available`
- `test_value_ratio_zero_when_no_absences`
- `test_pre_2018_all_nan`
- `test_friendly_match_all_nan`
- `test_strict_pre_match_anti_leakage`
- `test_missing_roster_returns_nan`

### `tests/test_train_tier4.py` (~3, smoke)
- `test_train_tier4_writes_artifacts`
- `test_baseline_and_challenger_share_random_state`
- `test_validation_report_written`

## 10. Risk plan & timeboxing

### 10.1 Rischi specifici

| # | Rischio | Prob | Impatto | Mitigazione |
|---|---|---|---|---|
| S1 | Wikipedia parsing fallisce (HTML cambiato) | Media | Medio | Test HTML offline su 4 fixture diverse. Fallback: skip torneo, log error, accettare coverage ridotta. |
| S2 | Match player_name fallisce per molte entries | Alta | Medio | Log esplicito (`injuries_bootstrap_complete n_skipped_no_match=...`). Top-up manuale possibile post-bootstrap. |
| S3 | Tier 4 non passa il gate | Alta (bayesian prior) | Basso | Esito accettato (baseline-first). Stesso pattern di STEP 4 con Tier 3. |
| S4 | Optuna 2 × 100 trial impiega più di 4h | Media | Basso | Reducibile a 50 trial via `--n-trials`. Budget hard cap 6h totali. |
| S5 | TM rate-limit / 429 sui rosters | Media | Medio | Rate limit ≥1.5s/req già rodato in STEP 4. `--resume` recupera senza ri-querying cache. |

### 10.2 Timeboxing

Hard cap **20h totali** sul lavoro umano (codice + review). Se a 8h il bootstrap (Fase A+B) non ha prodotto almeno 50% di rose scrappate e ≥10 injury entries, escalate: si valuta abort con piano break-glass R5 (skip a STEP 6 con Tier 1+2).

**Tempo macchina puro stimato**: ~4h (Optuna 2×100 trial CPU) + ~30 min (scraping). Eseguibile in background mentre si scrive altro codice/report.

## 11. Decision points & invarianti

1. **Tier 3 NON entra nel modello** (deciso post-STEP 4). Le 6 colonne di Tier 3 dirette restano fuori. `rosters.parquet` (player-level) è separato e usato solo per top-5 identification.
2. **Bootstrap zero su tornei diversi dai 4** (qualifiche, amichevoli, Copa America, AFC, Africa Cup) → feature NaN. Non tentiamo di estendere.
3. **Match player_name esatto, no fuzzy** (case-insensitive + accent strip ok, levenshtein no). Falsi positivi inquinano i dati molto peggio dei falsi negativi.
4. **`random_state=42` ovunque** (CLAUDE.md invariant 4).
5. **XGBoost JSON nativo, mai pickle** (CLAUDE.md invariant 5).
6. **Test-first** (CLAUDE.md invariant 1+2): ogni nuova feature entra con il test che la esercita.
7. Output deterministico: stesso input → stesso parquet/csv (sort + reset_index).

## 12. Acceptance criteria (gate STEP 5)

- [ ] `data/raw/transfermarkt/rosters.parquet` esiste con ≥80% delle 112 (nation, tournament) coppie target popolate.
- [ ] `data/manual/injuries.csv` esiste con ≥10 entries `source=wikipedia_squads` (best-effort, no hard floor superiore).
- [ ] `tests/test_tm_rosters.py`, `tests/test_injuries_bootstrap.py`, `tests/test_tier4.py`, `tests/test_train_tier4.py` esistono e passano (≥21 test).
- [ ] `tests/test_leakage.py::test_tier4_strict_pre_match` passa.
- [ ] `mondiali train-tier4` produce i 4 artefatti in `models/tier4/`.
- [ ] `reports/validation_step5.md` esiste con tabella metriche + decisione esplicita.
- [ ] Decisione tier-gate finale documentata: tier promossi per FREEZE = {1+2} oppure {1+2+4}.
- [ ] Tutti i commit pushati con conventional commits style.

Da qui si passa a **STEP 6 — Final validation + FREEZE v1**.
