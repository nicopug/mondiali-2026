# STEP 4 — Tier 3 Transfermarkt market values — Design

**Data**: 2026-04-29
**Predecessore**: STEP 3 chiuso (`step3-complete`, commit `4a707f0`). Tier 2 raw = 0.8487 vs ELO baseline 0.8525 — gate soft+hard PASS.
**Master plan**: `docs/superpowers/specs/2026-04-20-mondiali-prediction-design.md` §"STEP 4".

## TL;DR

Aggiungere al modello una terza fonte: i valori di mercato delle rose nazionali da Transfermarkt, scaricati via Wayback Machine per ogni `(nation, year)` dal 2014 al 2025. Il feature set XGBoost passa da 18 → 24 colonne (3 nuove × 2 simmetriche: `market_value_total`, `market_value_top11`, `tm_age_days`). Training Tier 3 su matches 2014-01-01+. Gate doppio: funzionale (coverage ≥80%) + metrico (`val_log_loss_raw_tier3 ≤ val_log_loss_raw_tier2 − 0.001` sul val_gate 2022). Calibration e Optuna restano **deferred** rispettivamente a STEP 6 e STEP 5.

## Motivazione

STEP 3 ha esaurito il *headroom* delle feature derivabili dal solo storico match (Elo + form rolling). Per superare ulteriormente la baseline servono *segnali esogeni*. Il valore di mercato dei giocatori è, in letteratura calcistica e in modelli betting professionali, il singolo predittore più forte oltre all'Elo per match top-tier. Lift atteso: 0.02-0.05 nat di log-loss (vs il <0.01 atteso dalle ottimizzazioni interne come Optuna o calibration tuning).

## Scope decisions

Tutte e 7 le decisioni-scope finalizzate durante il brainstorm:

1. **Direzione**: STEP 4 = Tier 3 Transfermarkt (master plan), NOT internal optimizations.
2. **Coverage temporale**: 2014+ only. Pre-2014 = NaN nelle 6 colonne TM (XGBoost gestisce nativamente).
3. **Feature design**: 3 features per lato (`market_value_total`, `market_value_top11`, `tm_age_days`) → +6 colonne, totale 24 features simmetriche.
4. **Scope nazionali**: ~70-80 = "FIFA top-50 storico (2014-2025) ∪ WC2026 qualificate". Lista deterministica generata da `matches.parquet` + lista hardcoded delle 48 qualificate.
5. **Gate**: doppio — funzionale (coverage ≥80% match-team training 2014-2022) + metrico (Δ ≤ −0.001 vs Tier 2 ricomputato apples-to-apples).
6. **Cadenza snapshot**: adaptive 1-snapshot/anno con fallback chain (target 1 luglio anno N, finestra ±60d → ±180d → tutto anno N → anno N-1).
7. **Fallback coverage**: forward-fill + hard floor ≥2 snapshot in 12 anni. Le nazionali sotto-floor escluse dal feature TM (NaN).

## Architettura

Tre nuovi sottosistemi, in ordine di dipendenza:

```
mondiali.data.transfermarkt
        ↓
data/raw/transfermarkt/snapshots.parquet
        ↓
mondiali.features.tier3 (in build_processed_matches)
        ↓
data/processed/matches.parquet (con 6 nuove colonne)
        ↓
mondiali.model.poisson_xgb (SYMMETRIC_FEATURES 18→24)
        ↓
mondiali.training.train.train_tier3_pipeline
        ↓
models/tier3/xgb_poisson.json + reports/validation_step4.md
```

**Filosofia anti-leakage**: lo scraper produce snapshot etichettati per `snapshot_date` reale (timestamp Wayback). Il feature builder al join garantisce `snapshot.date < match.date` strict (no future info). Nuovo test in `test_leakage.py` copre l'invariante.

## Data layer (`mondiali.data.transfermarkt`)

### Endpoint e parsing

Pattern URL Transfermarkt per nazionale: `https://www.transfermarkt.com/{nation-slug}/startseite/verein/{tm-id}`. Esempio: Italy = `italien/startseite/verein/3376`. La mappa `(nation_name → tm_slug, tm_id)` è una costante hardcoded `mondiali.data.transfermarkt.NATION_TM_IDS` (lookup table di ~80 entries, popolata manualmente una volta).

**Wayback CDX API** per trovare lo snapshot più vicino a un target date:
```
GET https://web.archive.org/cdx/search/cdx?
    url=transfermarkt.com/{nation-slug}/startseite/verein/{tm-id}
    &from={YYYYMMDD}&to={YYYYMMDD}&limit=1&output=json
```
Risposta = lista di `(urlkey, timestamp, original_url, mimetype, statuscode, digest, length)`.

URL Wayback per il fetch dell'HTML: `https://web.archive.org/web/{timestamp}/{original_url}`.

**Parser** (BeautifulSoup): il selettore primario è `<table id="kader">` (la tabella rosa). Per ogni `<tr>` riga giocatore, estrai:
- nome dal `<td class="hauptlink">`
- valore dal `<td class="rechts hauptlink">` (formato "€80.00m" / "€500k" / "-")
- posizione (non usata in Tier 3 ma logged per audit)

Funzione `_parse_value_eur(s: str) -> float | None`: gestisce "m" (milioni), "k" (migliaia), "-" (sconosciuto → None). Test sintetici su 3 fixture HTML salvate (formati DOM 2014, 2018, 2022).

`total_value_eur` = somma di tutti i valori parseati (None scartati). `top11_value_eur` = somma top-11 valori (sorted desc). `n_players` per audit.

### Cache, rate limiting, idempotenza

- **Cache**: ogni HTML scaricato salvato in `data/raw/transfermarkt/cache/{nation_slug}__{snapshot_ts}.html`. Riusato se presente.
- **Rate limiter**: hard 1 req / 2s su Wayback. Implementato con `time.sleep(2.0)` tra chiamate (no async/concurrent). Wayback è gentile ma punisce hammering.
- **Retry**: 3 tentativi exp-backoff (2s, 4s, 8s) su HTTP 5xx e timeouts. Fail soft (skip) su 404 → passa al fallback chain.
- **Idempotenza**: rilanciare `mondiali tm-scrape` non rifa chiamate già completate (cache hit).

### Fallback chain per `(nation, year)`

Implementata in `_best_snapshot_for_year(nation: str, year: int) -> Snapshot | None`:

1. CDX query: `from = {year}-05-01`, `to = {year}-09-01`, prendi il più vicino a `{year}-07-01`.
2. Se 0 hit: CDX query `from = {year}-01-01`, `to = {year}-12-31`.
3. Se 0 hit: CDX query `from = {year-1}-07-01`, `to = {year}-06-30` (fallback all'anno precedente, max age 18 mesi).
4. Se 0 hit: ritorna `None` → questo `(nation, year)` resta vuoto in `snapshots.parquet`.

Il `snapshot_date` salvato è il timestamp Wayback reale (es. `2018-08-23`), non il target nominale (`2018-07-01`). È quel timestamp che entra nel calcolo `tm_age_days`.

### Output: `data/raw/transfermarkt/snapshots.parquet`

Schema:
| col | type | descrizione |
|---|---|---|
| nation | str | nome nazionale (uppercase, match con `home_team`/`away_team` in matches.parquet) |
| year | int | anno target (2014-2025) |
| snapshot_date | date | timestamp Wayback effettivo |
| total_value_eur | float | somma valori rosa (EUR) |
| top11_value_eur | float | somma top-11 valori |
| n_players | int | numero giocatori parseati |
| source_url | str | URL Wayback (per audit) |

### Lista nazionali — `mondiali.data.scope.compute_tier3_scope`

Output: lista deterministica scritta in `data/processed/tier3_scope.json`.

Pseudocodice:
```python
WC2026_QUALIFIED: list[str] = [...]  # 48 entries hardcoded, stato 2026-04-29

def compute_tier3_scope(matches: pd.DataFrame) -> list[str]:
    # FIFA top-50 storico per Elo, anno per anno 2014-2025
    df = matches[matches["date"] >= "2014-01-01"].copy()
    df["year"] = df["date"].dt.year
    top50_by_year: set[str] = set()
    for year, grp in df.groupby("year"):
        # Per ogni team, prendi l'Elo massimo osservato nell'anno
        max_elo_home = grp.groupby("home_team")["home_elo_before"].max()
        max_elo_away = grp.groupby("away_team")["away_elo_before"].max()
        team_elo = pd.concat([max_elo_home, max_elo_away]).groupby(level=0).max()
        top50_by_year.update(team_elo.nlargest(50).index.tolist())

    return sorted(set(WC2026_QUALIFIED) | top50_by_year)
```

Atteso ~70-80 entries finali. Output scritto in `data/processed/tier3_scope.json` per audit.

### Coverage gate (warning)

Lo scraper a fine run logga:
```
coverage = {n_filled} / {n_target} = {pct}%
  per_nation_floor_violations: [Eritrea, Anguilla, ...]  # <2 snapshot
```

Se `coverage < 60%`: print warning ma non fail (la pipeline può comunque andare avanti — il gate metrico farà da arbitro finale).

### CLI

```
mondiali tm-scrape \
    --start-year 2014 --end-year 2025 \
    [--nations-file data/processed/tier3_scope.json] \
    [--resume]
```

Default: rilegge `tier3_scope.json` (rigenerato all'avvio se mancante). `--resume` è il default (cache idempotente).

## Feature layer (`mondiali.features.tier3`)

### `add_tier3_features(matches, snapshots) -> pd.DataFrame`

Algoritmo (vettorializzato):

1. **Hard floor check**: per ogni nazionale in `snapshots`, conta n snapshot. Se `<2`, escludi: tutti i suoi match avranno NaN sulle 6 colonne TM.
2. **Per match, per lato (home/away)**:
   - Cerca in `snapshots` la riga con `nation == team` e `snapshot_date < match_date`, ordinata `snapshot_date desc`, prendi prima.
   - Se non esiste (es. nazionale con tutti gli snapshot post-match): NaN per quel lato.
   - `team_tm_age_days = (match_date − snapshot_date).days`. Se `> 540` (1.5 anni): NaN (snapshot troppo stantio anche dopo forward-fill).
3. **Pre-2014 matches**: tutte le 6 colonne TM = NaN.

Output: `matches.parquet` con 6 colonne aggiunte:
- `home_market_value_total`, `away_market_value_total`
- `home_market_value_top11`, `away_market_value_top11`
- `home_tm_age_days`, `away_tm_age_days`

### Integrazione in `build_processed_matches`

Sequenza in `mondiali.data.ingestion.build_processed_matches`:

```python
matches = parse_results_csv(raw_csv)
matches = add_elo_history(matches)
matches = add_tier1_features(matches)   # esistente
matches = add_tier2_features(matches)   # esistente

snapshots_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"
if snapshots_path.exists():
    snapshots = pd.read_parquet(snapshots_path)
    matches = add_tier3_features(matches, snapshots)
else:
    # Pipeline non rompe se TM non ancora scrapato
    for col in TIER3_COLUMNS:
        matches[col] = pd.NA
```

### Test anti-leakage (`tests/test_leakage.py`)

Nuovo test `test_tier3_market_value_strict_pre_match`:
```python
def test_tier3_market_value_strict_pre_match():
    matches = pd.read_parquet(...)  # processed
    snapshots = pd.read_parquet(...)
    rebuilt = add_tier3_features(matches.drop(columns=TIER3_COLUMNS), snapshots)
    for _, row in rebuilt.iterrows():
        if pd.notna(row["home_tm_age_days"]):
            assert row["home_tm_age_days"] >= 0  # snapshot strictly pre-match
        if pd.notna(row["away_tm_age_days"]):
            assert row["away_tm_age_days"] >= 0
```

## Model layer (`mondiali.model.poisson_xgb`)

Estensione di `SYMMETRIC_FEATURES` (18 → 24):

```python
SYMMETRIC_FEATURES += [
    "team_market_value_total",
    "opponent_market_value_total",
    "team_market_value_top11",
    "opponent_market_value_top11",
    "team_tm_age_days",
    "opponent_tm_age_days",
]
```

`build_symmetric_rows` extension lineare: indici 18-23 popolati con lo stesso pattern simmetrico delle Tier 2 (home-perspective in righe pari, away-perspective in dispari, con scambio team/opponent).

**Nessun refactoring** del codice esistente: aggiunta pura. Il pattern Tier 2 è già lì da emulare.

`tm_age_days` agisce come *soft confidence weight* implicito: depth=6 cattura la interaction `tm_age_days × market_value` nello split tree.

## Training pipeline (`mondiali.training.train.train_tier3_pipeline`)

Mirror di `train_tier2_pipeline` con due differenze chiave:

1. **Filtro 2014+**: `train = train[train["date"] >= "2014-01-01"]`. Match pre-2014 esclusi dal Tier 3 training perché TM è NaN per metà del dataset, e il booster pendi sul "missingness pattern" piuttosto che sul vero segnale.

2. **Calibrator informational**: rimane nel return per consistenza ma non blocca il gate (deferred a STEP 6).

### 4-way split adattato (anche per allinearsi a Tier 2 metodologia)

| Split | Date range | n match attesi |
|---|---|---|
| Train | 2014-01-01 → 2019-12-31 | ~6500 |
| Val_ES (early stopping) | 2020-01-01 → 2020-12-31 | ~700 (COVID — accettiamo bias) |
| Val_calib (isotonic fit) | 2021-01-01 → 2021-12-31 | ~900 |
| Val_gate (metrica finale) | 2022-01-01 → 2022-12-31 | ~1100 (include WC2022) |

### Returns dict

```python
{
    "model": PoissonXGBModel,
    "rho": float,
    "calibrator": IsotonicCalibrator1X2,  # informational
    "val_log_loss_raw": float,            # metric ufficiale
    "val_log_loss_calib": float,          # informational
    "brier_before": float,
    "brier_after": float,
    "n_train": int,
    "n_val_es": int,
    "n_val_calib": int,
    "n_val_gate": int,
    "n_train_pre2014_dropped": int,
    "tm_coverage_train": float,           # = sum(home & away TM both non-NaN) / n_train
    "tm_coverage_gate": float,            # idem su val_gate
}
```

### Baseline confronto per il gate metrico

Tier 2 raw da STEP 3 era `0.8487` ma su val_gate 2019-2022. Per questo gate Tier 2 va ricomputato sullo *stesso* val_gate 2022 only (apples-to-apples). Funzione `_recompute_tier2_baseline_for_gate(parquet, val_gate_start, val_gate_end) -> float` invocata all'inizio del pipeline e loggata.

Gate B: `val_log_loss_raw ≤ tier2_baseline_2022 − 0.001`.

## CLI

```
mondiali train-tier3 \
    [--save-model models/tier3/xgb_poisson.json] \
    [--save-calibrator models/tier3/calibrator.json]
```

Pattern identico a `train-tier2`. Tutti i 4 split-args esposti come typer options con i default sopra.

## Tests

Nuovi file:
- `tests/test_transfermarkt.py`: parser HTML su 3 fixture salvate (2014, 2018, 2022); CDX query mockata via `responses` lib; fallback chain; rate-limiter (mock `time.sleep`); idempotenza cache.
- `tests/test_features_tier3.py`: anti-leakage strict-pre, forward-fill behavior, hard floor ≥2, age_days clipping a 540d, NaN propagation pre-2014.
- `tests/test_train_tier3.py`: smoke test rapido + slow test gate-blocking.

Esteso:
- `tests/test_leakage.py`: aggiunge `test_tier3_market_value_strict_pre_match`.

### Slow test gate

```python
@pytest.mark.slow
def test_train_tier3_full_split_passes_gate():
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists() or "home_market_value_total" not in pd.read_parquet(parquet).columns:
        pytest.skip("Tier 3 not yet scraped/built")
    result = train_tier3_pipeline(parquet_path=parquet)
    tier2_baseline = _recompute_tier2_baseline_for_gate(parquet)
    assert result["val_log_loss_raw"] <= tier2_baseline - 0.001
    assert -0.3 <= result["rho"] <= 0.05
    assert result["tm_coverage_gate"] >= 0.80
```

## Gate ufficiale STEP 4

| Gate | Soglia | Tipo | Blocking? |
|---|---|---|---|
| Funzionale | `coverage ≥ 80%` su match-team training 2014-2022 | Functional | ✅ |
| Funzionale | Tutti i test in suite passano (incluso slow `train-tier3`) | Functional | ✅ |
| Metrico | `val_log_loss_raw ≤ tier2_baseline_2022 − 0.001` su val_gate 2022 | Metric | ✅ |
| Anti-leakage | `test_tier3_market_value_strict_pre_match` passa | Safety | ✅ |

### Gate-fail policy

Se gate metrico fallisce dopo che funzionali passano:
- Il report `reports/validation_step4.md` documenta il fallimento con numeri.
- Decisione: "Tier 3 non aggiunge segnale → modello v1 finale userà solo Tier 1+2".
- Codice scraper resta nel repo come legacy (potrebbe essere rivisitato in STEP 5+).
- Tag `step4-no-signal` invece di `step4-complete`.

Se gate funzionale (coverage) fallisce:
- Investigazione obbligatoria: rilancio scraper, analisi delle nazionali con coverage bassa, eventuale fallback manuale per le top-32 (escalation).

## Caveats — cosa NON faccio in STEP 4

1. **Calibration fix**: l'isotonic calibrator dello STEP 3 è broken e resta broken. Cross-fit calibration → STEP 6 (final validation + reliability diagrams).
2. **Optuna**: rinviato a STEP 5 come da master plan (Optuna allargato 100+ trial su Tier 1+2+3+4).
3. **Per-position TM breakdown**: scartato (overfit risk + parsing fragile).
4. **Real-time TM**: niente refresh continuo. Snapshot statici, ricostruito solo se rilanci `tm-scrape`. Per il torneo (STEP 7) si farà uno snapshot dedicato post-annuncio rose ufficiali.
5. **Pre-2014 matches in Tier 3 training**: esclusi (NaN su feature critiche → bias). Il modello Tier 3 ha quindi un training set più piccolo di Tier 2 (~6500 vs 14161). Trade-off accettabile: più segnale per match più recente.
6. **Wayback historical pre-2014**: scartato. Coverage troppo patchy, parsing su DOM legacy fragile, e i match pre-2014 contano poco per WC2026.

## Aperti per STEP 5

Heritage list per il prossimo step:
1. Optuna su 24 features (deferred da STEP 4 e STEP 3).
2. Tier 4 = injuries (master plan).
3. Cross-fit calibration (preview di STEP 6).

## Test suite delta atteso

- 127 test (post STEP 3) → ~150-160 test (post STEP 4):
  - +6-8 test in `test_transfermarkt.py`
  - +5-7 test in `test_features_tier3.py`
  - +3 test in `test_train_tier3.py`
  - +1 test in `test_leakage.py`

## Anti-data-leakage — riepilogo

Invarianti che il design garantisce:
- **Snapshot strict-pre-match**: `snapshot_date < match_date` enforced by `add_tier3_features`. Test in `test_leakage.py`.
- **Age clipping**: `tm_age_days > 540` → NaN. Previene forward-fill abusi.
- **Hard floor coverage**: nazionali con `<2` snapshot in 12 anni escluse interamente (no propagazione di un singolo valore).
- **Pre-2014 NaN**: nessun valore "presente" venga trascinato a un'epoca dove TM non era ancora la fonte affidabile odierna.

## Lift atteso e rischi

- **Lift atteso**: 0.02-0.05 nat di log-loss su val_gate 2022 (vs Tier 2 ricomputato). Letteratura suggerisce TM è il singolo predittore più forte oltre l'Elo per top-tier.
- **Rischio principale**: Wayback coverage <80% → gate funzionale fail → escalation. Mitigazione: scope D (~80 nazionali) + adaptive snapshot + fallback chain a 4 livelli.
- **Rischio secondario**: gate metrico fail con coverage OK → "Tier 3 non aggiunge segnale". Mitigazione: feature `tm_age_days` come soft confidence → il booster decide da solo se fidarsi.
