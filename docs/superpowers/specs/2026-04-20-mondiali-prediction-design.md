# Design: World Cup 2026 Prediction System

**Data**: 2026-04-20
**Autore**: Nicolò (con brainstorming assistito)
**Stato**: Spec approvata, pronto per writing-plans
**Target**: Sistema di predizione per il FIFA World Cup 2026 (kickoff 11 giugno 2026)
**Tempo disponibile**: ~52 giorni dal 2026-04-20 al 2026-06-10

---

## 1. Executive Summary

Sistema Python offline che predice l'esito delle partite del Mondiale 2026 tramite un modello XGBoost con objective Poisson per stimare i gol attesi (λ_home, λ_away) di ciascuna squadra, con correzione Dixon-Coles post-hoc per i punteggi bassi e calibrazione isotonic-regression sulle probabilità 1/X/2. Dalla distribuzione congiunta dei gol si derivano tutti i mercati (1X2, O/U, BTTS, handicap asiatico, risultato esatto).

Il progetto è costruito contro i fallimenti passati dell'autore, dove la complessità introdotta troppo presto (LSTM + ensemble + 50 feature senza baseline) ha impedito qualsiasi misura onesta del valore di ogni aggiunta. La regola centrale è **baseline-first**: ogni complessità deve essere giustificata da un miglioramento misurato di ≥0.003 log-loss su validation temporale.

Il deliverable è un **package Python installabile con CLI** (no UI web, no database). Durante il torneo l'autore aggiorna manualmente un CSV di infortuni pre-match, e lo script `predict` restituisce probabilità multi-mercato per ogni partita.

## 2. Objectives

**Obiettivi (IN SCOPE)**:

1. **Metrica rigorosa** — log-loss 1/X/2 sui 64 match del Mondiale 2026 sotto 0.97, con calibrazione verificata via reliability diagram e Brier score.
2. **Esperimenti "superpoteri"** — integrazione di dati Transfermarkt (valore rosa) e tabella manuale infortuni top-5, ognuno misurato separatamente contro baseline.
3. **Sistema riproducibile** — ogni training è deterministico (`random_state=42`), ogni report archiviato, nessun notebook che modifica stato globale.

**Non-obiettivi (OUT OF SCOPE, YAGNI esplicito)**:

- UI web / frontend React
- Database (SQLite o altro)
- Containerizzazione Docker
- Deploy cloud / CI-CD remoto
- Bivariate Poisson / copulas (complessità non giustificata per v1)
- Reinforcement learning su sequenze di match
- Backtesting di strategie di betting
- Tentativo di "battere i bookmaker" professionali

**Nota legale**: l'autore ha 16 anni, età inferiore al minimo legale italiano per le scommesse sportive (18+). Il modello si progetta come strumento educativo/di ricerca. Le decisioni di uso per scommesse reali sono posticipate a maggiore età.

## 3. Architettura

Cinque moduli disaccoppiati che comunicano tramite file su disco (Parquet/CSV/JSON). Nessun database.

```
┌─────────────────────┐     ┌─────────────────────┐
│  1. data_ingestion  │────▶│  2. feature_build   │
│  (scraper + ETL)    │     │  (team ratings,     │
│                     │     │   form, market val) │
└─────────────────────┘     └──────────┬──────────┘
         │                             │
         ▼                             ▼
┌─────────────────────┐     ┌─────────────────────┐
│  data/              │     │  3. modeling        │
│  ├── matches.parquet│     │  (XGBoost Poisson + │
│  ├── rosters.parquet│     │   Dixon-Coles corr) │
│  ├── market.parquet │     │                     │
│  └── injuries.csv   │◀────│                     │
└─────────────────────┘     └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  4. validation      │
                            │  (temporal CV,      │
                            │   calibration)      │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  5. predict_cli     │
                            │  (CLI + notebook    │
                            │   di predizione)    │
                            └─────────────────────┘
```

**Razionale**: file invece di DB per ridurre superficie (20MB totali, monouso); moduli disaccoppiati per iterazione rapida (ri-esegui solo il pezzo che cambia); package installabile (`pip install -e .`) invece di notebook sparsi per testabilità e riproducibilità.

## 4. Data Pipeline

### 4.1 Sorgenti

| Dato | Sorgente primaria | Fallback |
|---|---|---|
| Risultati match internazionali (~45k dal 1872) | `martj42/international_results` (GitHub CSV) | Scraping Wikipedia |
| FIFA World Rankings | `fifa.com/rankings` scraping | `fifaindex.com` |
| Elo ratings storici | **Calcolo nostro** da international_results | — |
| Market values squadre | `felipeall/transfermarkt-api` wrapper | `requests + BeautifulSoup` diretto |
| Rose ufficiali WC2026 | Transfermarkt + FIFA ufficiale (post ~6 giugno) | Wikipedia "2026 FIFA World Cup squads" |
| Infortuni/squalifiche | **Tabella CSV manuale** mantenuta dall'utente | — |

### 4.2 Schema file

```
data/
├── raw/
│   ├── international_results.csv
│   ├── transfermarkt_raw_YYYYMMDD/
│   └── fifa_rankings_YYYYMMDD.json
├── processed/
│   ├── matches.parquet              # record per match storico, con Elo pre-match
│   ├── team_elo_history.parquet     # Elo di ogni squadra per ogni data
│   ├── market_values.parquet        # valore rosa per squadra per snapshot
│   └── rosters_wc2026.json          # rose congelate pre-torneo
└── manual/
    └── injuries.csv
```

**Schema `matches.parquet`**:
```
match_id, date, competition, stage, home_team, away_team,
home_goals, away_goals,
home_elo_before, away_elo_before,
is_neutral_venue, is_competitive,
days_since_home_last_match, days_since_away_last_match
```

**Schema `injuries.csv`**:
```
date_of_knowledge, team, player_name, market_value_eur, status
2026-06-05, France, Kylian Mbappé, 180000000, out
```
dove `status ∈ {out, doubtful, available}`.

### 4.3 Regola inviolabile: anti-data-leakage

Ogni feature per un match alla data D deve usare **esclusivamente** informazioni disponibili **strettamente prima di D**.

Implementazione:
- `team_elo_history` memorizza l'Elo dopo ogni partita; il lookup per il match alla data D usa il timestamp `D - 1s`
- Le feature di form recente usano solo match con `date < D`
- `market_values` proietta lo snapshot più recente con `snapshot_date ≤ D`
- `tests/test_leakage.py` verifica l'invariante prima di ogni training

### 4.4 Elo custom

Implementazione semplice con K-factor variabile per competizione:
- Mondiale: K=60
- Competizioni continentali (Euro, Copa, AFC, etc.): K=50
- Qualificazioni: K=40
- Amichevoli: K=20

Home advantage: +65 di Elo aggiunto alla squadra di casa nel calcolo delle expected probabilities (standard eloratings.net). **Se `is_neutral_venue=True` (es. tutte le partite del Mondiale su suolo nord-americano per squadre non-USA/MEX/CAN), home_advantage=0** e nessuna delle due squadre è considerata "home" ai fini dell'update Elo.

## 5. Feature Engineering — Tier System

Feature organizzate in 4 tier. Il passaggio al Tier N+1 richiede che la gate di validation sia superata (≥0.003 miglioramento log-loss).

### Tier 0 — Baseline banale
Solo `is_home`, `is_neutral_venue`, prior storico costante. Serve come floor di riferimento. Log-loss atteso ~1.05.

### Tier 1 — Core team-level (MVP)
| Feature | Calcolo |
|---|---|
| `elo_diff` | `home_elo_before - away_elo_before` |
| `elo_home`, `elo_away` | Elo assoluti |
| `is_neutral_venue` | Boolean |
| `competition_importance` | Ordinal: 1=amichevole, 2=qualif, 3=continental, 4=World Cup |
| `days_rest_home`, `days_rest_away`, `days_rest_diff` | Giorni dall'ultimo match |

### Tier 2 — Form recente
| Feature | Calcolo |
|---|---|
| `home_form_5`, `away_form_5` | Punti (W=3, D=1, L=0) ultimi 5 match |
| `home_gd_5`, `away_gd_5` | Goal difference ultimi 5 |
| `home_goals_scored_5`, `home_goals_conceded_5` | Media gol |
| Idem away | |

Tutte calcolate con `rolling_with_cutoff(team, N, cutoff_date=match_date)`.

### Tier 3 — Market value (Transfermarkt)
Solo per match dal 2018 in poi (valori storici inaffidabili prima).
| Feature | Calcolo |
|---|---|
| `home_squad_value_log` | `log(1 + squad_market_value_eur)` |
| `away_squad_value_log` | Idem |
| `value_ratio` | `log(home_value / away_value)` |
| `home_top11_value_log` | Log valore dei top 11 |
| `away_top11_value_log` | Idem |

Per match pre-2018: feature `NaN` (XGBoost gestisce nativamente).

### Tier 4 — Infortuni (manuale)
| Feature | Calcolo |
|---|---|
| `home_top5_absent_count` | Quanti dei top-5 valore sono `out` o `doubtful` |
| `away_top5_absent_count` | Idem |
| `home_value_absent_ratio` | Somma market_value assenti / market_value totale top-5 |
| `away_value_absent_ratio` | Idem |

### Principi trasversali

1. **Temporal safety**: ogni feature passa per `rolling_with_cutoff` o lookup strettamente anteriore a `match_date`.
2. **Missing values espliciti come `NaN`**, mai imputati.
3. **No one-hot encoding delle squadre**: le squadre entrano nel modello solo tramite Elo e (Tier 3) market value. Questo è cruciale per generalizzazione.
4. **No scaling**: gradient boosting non lo richiede.
5. **Feature groups esportati**: `FEATURE_GROUPS = {"tier1": [...], ...}`, trainer accetta `--tiers 1,2` come argomento.

## 6. Model Architecture

### 6.1 Symmetric single-model

Un unico XGBoost allenato con `objective='count:poisson'` predice "gol segnati da una squadra" dato `(features_attaccante, features_difensore, is_home, ...)`. Ogni match genera due righe di training (una da prospettiva home, una da prospettiva away).

Razionale: raddoppia training set (25k match → 50k righe), forza simmetria del concetto "segnare gol", una sola pipeline di hyperparam search, una sola analisi SHAP.

### 6.2 Pipeline di inference

Dato un nuovo match:

1. `λ_home = model.predict(features_home_as_team)`, `λ_away = model.predict(features_away_as_team)`
2. Costruisci matrice congiunta `P(i, j) = Poisson(i; λ_home) * Poisson(j; λ_away)` per `i, j ∈ [0, 10]`
3. Applica correzione Dixon-Coles:
   - `P(0,0) *= 1 - λ_h·λ_a·ρ`
   - `P(0,1) *= 1 + λ_h·ρ`
   - `P(1,0) *= 1 + λ_a·ρ`
   - `P(1,1) *= 1 - ρ`
   con ρ stimato globalmente via MLE (~-0.1 empirico)
4. Rinormalizza la matrice (la correzione rompe la somma a 1)
5. Deriva i mercati:
   - `P(1) = Σ_{i>j} P(i,j)`, `P(X) = Σ_{i=j}`, `P(2) = Σ_{i<j}`
   - `P(Over 2.5) = Σ_{i+j>2.5}`
   - `P(BTTS) = Σ_{i>0 ∧ j>0}`
   - Risultato esatto = cella stessa

### 6.3 Calibrazione post-hoc

XGBoost con objective Poisson non produce probabilità 1X2 ben calibrate. Pipeline:

1. Sul validation set, calcola P(1), P(X), P(2) grezzi
2. Fitta `IsotonicRegression` separatamente per ciascuna delle 3 classi
3. In inference: passa le grezze attraverso i tre isotonic e rinormalizza a somma 1

Diagnostica: reliability diagram + Brier score. La calibrazione è **obbligatoria**, non opzionale.

### 6.4 Iperparametri

- `optuna` con 50-100 trial ottimizzando **validation log-loss** (mai accuracy)
- Early stopping con `patience=50`
- Spazio di ricerca: `max_depth [3,8]`, `learning_rate [0.01,0.2]`, `n_estimators [100,2000]`, `reg_alpha [0,10]`, `reg_lambda [0,10]`, `min_child_weight [1,20]`
- Preferenza a modelli più piccoli a parità di performance (Occam)

### 6.5 Approssimazione accettata

Il modello assume indipendenza tra `home_goals` e `away_goals`. Dixon-Coles corregge solo i punteggi bassi. La letteratura sports-analytics indica che basta per ~95% del valore pratico. Modelli bivariate Poisson / copula sono fuori scope v1.

## 7. Validation Strategy

### 7.1 Split temporale (mai random)

```
2002──────────2018  │  2018──2022  │  2022──2026
  TRAINING          │  VALIDATION  │  TEST (held-out)
  ~18k match        │  ~4k match   │  ~3k match
```

- **Training** (2002-2018): feature engineering, addestramento, tuning preliminare
- **Validation** (2018-2022): optuna, confronto tier, fit isotonic calibrator
- **Test** (2022-oggi, ~3k match): **toccato una sola volta** in STEP 6. Include Euro 2024, Copa América 2024, Nations League, qualifiche WC2026 e soprattutto **WC2022 Qatar (64 match)** che è il benchmark primario per comparabilità con il target WC2026. Il resto del test set serve come robustness check su competizioni non-World-Cup.

### 7.2 Walk-forward CV (dentro training)

Tre fold expanding-window:
- Fold 1: train 2002-2015, val 2015-2016
- Fold 2: train 2002-2016, val 2016-2017
- Fold 3: train 2002-2017, val 2017-2018

### 7.3 Metriche (in ordine di importanza)

1. **Log-loss 1/X/2** — primaria. Baseline di riferimento: prior ~1.05, Elo-only ~0.98, target v1 **su validation** (2018-2022) < 0.95 per Tier finale. Target **su test WC2026** < 0.97 (gap accounts per distribution drift test-vs-validation + varianza su 64 match).
2. **Brier score 1/X/2** — sanity check calibrazione.
3. **Reliability diagram** — plot obbligatorio.
4. **Log-loss sui mercati derivati** (O/U 2.5, BTTS).
5. **Accuracy 1/X/2** — ultima, riportata ma non ottimizzata.

### 7.4 Baseline obbligatori di confronto

| Baseline | Log-loss atteso |
|---|---|
| Random uniforme | ~1.10 |
| Prior storico 45/25/30 | ~1.05 |
| Home-advantage puro | ~1.00 |
| Elo-only logistic | ~0.98 |
| Tier 1 target | < 0.97 |
| Tier 1+2 target | < 0.96 (se giustificato) |
| Tier 1+2+3 target | < 0.955 |
| Tier 1+2+3+4 target | < 0.95 |

### 7.5 Regola tier-gate

Tier N+1 si integra solo se miglioramento log-loss ≥ **0.003** su validation rispetto a Tier N. Delta inferiore = rumore di tuning, non segnale.

### 7.6 Protocollo live (torneo)

1. Aggiornare `injuries.csv` entro 24h pre-match
2. `mondiali predict X Y --date ...` → logging in `predictions_log.csv`
3. Script post-match confronta predizione vs realtà
4. **NO model changes** durante torneo (inquina log-loss finale)
5. Report finale su 64 match del Mondiale 2026

## 8. Project Structure

```
progetto_mondiali/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── .env.example
├── .gitignore
│
├── src/mondiali/
│   ├── config.py                    # paths, K-factor Elo, FEATURE_GROUPS, TEAM_NAME_MAPPING
│   ├── data/
│   │   ├── ingestion.py             # download international_results, FIFA ranking
│   │   ├── transfermarkt.py         # scraper con caching aggressivo
│   │   ├── rosters.py               # loader + lookup per squadra/data
│   │   └── injuries.py              # parser injuries.csv + query assenti
│   ├── features/
│   │   ├── elo.py                   # Elo con K variabile + storia completa
│   │   ├── form.py                  # rolling con cutoff temporale
│   │   ├── market.py                # feature Transfermarkt
│   │   ├── injury.py                # feature injuries
│   │   └── builder.py               # orchestra tier, produce matches.parquet
│   ├── model/
│   │   ├── poisson_xgb.py           # wrapper XGBoost count:poisson
│   │   ├── dixon_coles.py           # correzione + joint distribution
│   │   ├── calibration.py           # isotonic calibrator 1/X/2
│   │   └── markets.py               # derivazione 1X2, O/U, BTTS, handicap
│   ├── training/
│   │   ├── splits.py                # split temporali + walk-forward CV
│   │   ├── train.py                 # loop + optuna
│   │   └── evaluate.py              # log-loss, Brier, reliability
│   └── cli/
│       ├── ingest.py                # `mondiali ingest`
│       ├── build.py                 # `mondiali build --tiers 1,2,3`
│       ├── train.py                 # `mondiali train --tiers 1,2,3`
│       ├── evaluate.py              # `mondiali evaluate --model path`
│       └── predict.py               # `mondiali predict France Italy`
│
├── tests/
│   ├── test_leakage.py              # IL test critico
│   ├── test_elo.py                  # Elo Francia fine 2018 ~2080
│   ├── test_dixon_coles.py          # sum matrice = 1 dopo correzione
│   ├── test_calibration.py          # Brier dopo ≤ Brier prima
│   └── test_markets.py              # invarianti 1X2 + sanity (Francia > San Marino)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_elo_analysis.ipynb
│   ├── 03_feature_importance.ipynb  # SHAP
│   ├── 04_calibration_diagnostics.ipynb
│   └── 05_tournament_results.ipynb
│
├── data/                            # gitignored
│   ├── raw/
│   ├── processed/
│   └── manual/injuries.csv
│
├── models/                          # gitignored
│   └── v1_final/
│       ├── xgb_poisson.json
│       ├── isotonic_{1,X,2}.pkl
│       └── metadata.json
│
└── reports/
    ├── validation_YYYYMMDD.md
    └── tournament_2026/
        ├── predictions_log.csv
        └── live_metrics.md
```

### 8.1 Convenzioni

- Type hints ovunque
- Pydantic per config
- Logging strutturato con `structlog`
- `random_state=42` di default, runs riproducibili
- XGBoost salvato in formato JSON nativo (no pickle)
- Test in parallelo con `pytest-xdist`
- Ruff + mypy come quality gates

### 8.2 Dipendenze (pyproject.toml)

```toml
[project]
dependencies = [
    "xgboost>=2.0",
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "scipy>=1.11",
    "optuna>=3.4",
    "pydantic>=2.0",
    "typer>=0.9",
    "requests",
    "beautifulsoup4",
    "pyarrow",
    "structlog",
    "shap",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-xdist", "ruff", "mypy", "ipython", "jupyter"]
# Solo se STEP 8 (LSTM) viene intrapreso:
lstm = ["torch>=2.0"]
# Solo se STEP 9 (news) viene intrapreso:
news = ["feedparser", "anthropic"]
# Solo se STEP 11 (Telegram) viene intrapreso:
telegram = ["python-telegram-bot>=20"]
```

## 9. Roadmap — STEP-based

Distribuzione non-calendar (l'autore ha disponibilità variabile: 8h+/weekend, ~2h/feriale). Struttura in 3 fasi: **Fase A core blindata** → **Fase B experiments opzionali** → **Fase C torneo live**.

### Fase A — CORE (deve finire prima dell'11 giugno 2026)

#### STEP 1 — Foundation (~10-15h)
- Setup `pyproject.toml`, struttura cartelle, `ruff` + `pytest` + `mypy`
- `data/ingestion.py`: download `international_results` + snapshot FIFA ranking
- `features/elo.py`: K variabile per competizione, storia completa
- `tests/test_leakage.py` framework base
- `tests/test_elo.py`: sanity (Francia fine 2018 Elo ~2080)
- Training **Tier 0** (prior costante) come floor di riferimento
- Report `reports/validation_step1.md`

**Gate**: test verdi, log-loss prior documentato.

#### STEP 2 — Tier 1 model (~15-20h)
- `model/poisson_xgb.py` symmetric single-model
- `model/dixon_coles.py` correzione low-score + stima ρ via MLE
- `model/markets.py` derivazione 1X2, O/U, BTTS
- `training/splits.py` walk-forward CV
- `training/train.py` + optuna (50 trial)
- Training Tier 1, confronto vs Elo-only logistic

**Gate**: Tier 1 batte Elo-only logistic in log-loss sul validation. Se no → stop, debug (probabile bug feature o leakage).

#### STEP 3 — Tier 2 + Calibration (~10-15h)
- `features/form.py` rolling con cutoff
- Estensione `test_leakage.py` per form
- `model/calibration.py` isotonic + reliability diagram
- Training Tier 1+2 calibrato
- **Decisione tier-gate**: Tier 2 passa soglia 0.003?

**Gate**: Tier 2 integrato o rifiutato con documentazione.

#### STEP 4 — Tier 3 Transfermarkt (~15-20h)
- `data/transfermarkt.py` scraper con caching aggressivo
- Snapshot rose + valori per tutte le nazionali top
- Tentativo historical via Wayback Machine
- `features/market.py`
- Training Tier 1+2+3
- **Decisione tier-gate**

**Gate**: scraper works (o fallback snapshot manuale) + Tier 3 decision.

#### STEP 5 — Tier 4 Injuries (~15-20h)
- Schema `injuries.csv` + `data/injuries.py`
- **Bootstrap historical**: best-effort WC2018, WC2022, Euro2020, Euro2024 da news archive + Wikipedia
- `features/injury.py`
- Training Tier 1+2+3+4
- Optuna allargato (100+ trial)
- **Decisione tier-gate finale**

**Gate**: tier finali scelti, iperparametri finalizzati.

#### STEP 6 — Final validation + FREEZE v1 (~8-12h)
- Evaluate modello finale su **Mondiale 2022 (Qatar)** — 64 match mai visti
- Reliability diagrams finali
- SHAP feature importance
- **Freeze** → `models/v1_final/` + metadata
- Polish CLI `predict`

**Gate**: `models/v1_final/` esiste, log-loss WC2022 documentato. **Da qui v1 NON si tocca più.**

#### STEP 7 — Tournament readiness (~8-12h, OBBLIGATORIO entro 10 giugno 2026)
- Post annuncio rose ufficiali (~6 giugno): snapshot TM specifico
- Ricostruzione valori rosa per tutte le 48 nazionali WC2026
- `injuries.csv` inizializzato con infortuni pre-torneo noti
- **Dry run** su tutte le 48 partite della fase a gironi
- Verifica assenza di predizioni assurde

**Gate**: sistema pronto per kickoff, `reports/pre_tournament_readiness.md` scritto.

### Fase B — EXPERIMENTS (opzionali, ognuno con gate isolato)

**Regola invalicabile**: nessuno di questi STEP può ritardare Fase A. Se Fase A sfora, Fase B si salta. `v1_final` al torneo è garantito.

#### STEP 8 — LSTM experiment (v2_lstm) (~20-30h)
Architettura proposta:
- Input: sequenze degli ultimi 10 match di ciascuna squadra (feature + risultato)
- LSTM bidirezionale su ciascuna sequenza → embedding 32-dim per squadra
- Concatenazione embedding + feature match-level attuali
- Head finale che predice (λ_home, λ_away) con Poisson loss
- Stesso split temporale e protocol di Tier 4

**Gate onesto**: v2_lstm batte v1_final di **≥0.005 log-loss** su WC2022 test. Altrimenti archiviato con documentazione del perché.

**Avvertenza**: aspettativa razionale = LSTM perde. Le nazionali giocano pochi match/anno, tree models dominano la letteratura. L'esperimento ha valore anche se conferma la sufficienza di XGBoost.

#### STEP 9 — News scraping + NLP (Tier 5) (~20-30h)
- Scraper RSS/NewsAPI: Gazzetta, ESPN, BBC Sport, The Athletic
- Fetch articoli 48h pre-match menzionanti le due squadre
- Entity extraction via LLM (es. Claude Haiku): infortuni, dubbi, morale
- Feature: `{home,away}_news_negative_count`, `{home,away}_news_sentiment_score`
- Training v2_xgb_with_news (e v2_lstm_with_news se STEP 8 passato)

**Gate**: ≥0.003 miglioramento log-loss su validation.

#### STEP 10 — Ensembling v3 (~8-12h)
Se STEP 8 e/o STEP 9 producono modelli validi:
- Simple averaging: `P_ensemble = weighted(P_xgb, P_lstm)`, pesi calcolati su validation log-loss
- Stacking: meta-modello (logistic regression) sulle probabilità dei base models

**Gate**: v3_ensemble batte il migliore tra modelli individuali di ≥0.003.

#### STEP 11 — Telegram bot (~4-6h)
- Bot con `python-telegram-bot` v20
- Handler `/predict X Y YYYY-MM-DD` → chiama CLI → messaggio formattato
- Schedule opzionale: post automatico 3h pre-match
- Hosting locale o VPS low-cost

**Gate**: nessuno (cosmetica).

### Fase C — TOURNAMENT LIVE (dall'11 giugno 2026)

#### STEP 12 — Execution
- Aggiornamento `injuries.csv` entro 24h pre-match
- `mondiali predict` per ogni partita con logging
- Script post-match confronta predizione vs realtà
- **NO model changes** durante torneo
- Report finale post-Mondiale (15 luglio 2026)

## 10. Risk Register

### 10.1 Matrice

| # | Rischio | Prob | Impatto | Priorità |
|---|---|---|---|---|
| R1 | Transfermarkt scraper rotto | Alta | Medio | 🔴 |
| R2 | Historical injury data non recuperabile | Alta | Medio | 🟠 |
| R3 | LSTM non batte XGBoost | Molto alta | Basso (accettato) | 🟢 |
| R4 | News segnale troppo rumoroso | Alta | Basso | 🟢 |
| R5 | Sfori Fase A, no STEP 6 entro inizio giugno | Media | Catastrofico | 🔴 |
| R6 | Rose cambiano dopo kickoff | Media | Medio | 🟠 |
| R7 | Dry run produce predizioni assurde | Media | Alto | 🟠 |
| R8 | Data leakage scoperto tardi | Bassa | Catastrofico | 🔴 |
| R9 | Nomi squadre inconsistenti tra fonti | Alta | Medio | 🟠 |
| R10 | WC2026 è primo Mondiale a 48 squadre (vs 32 storico) | Certa | Basso | 🟢 |
| R11 | Gol medi anomali nel torneo | Bassa | Medio | 🟡 |
| R12 | GPU/RAM insufficienti per LSTM | Bassa | Basso | 🟢 |

### 10.2 Piani break-glass rossi

**R1 — Transfermarkt rotto**
Fall back a snapshot cachati in `data/raw/transfermarkt_raw_*/`. Se servisse refresh: Playwright one-shot manuale di ~48 nazionali. Peggiore dei casi: Tier 3 congelato all'ultimo snapshot — feature non aggiornata durante torneo ma funzionante.

**R5 — Sfori Fase A**
In ordine: (1) salta STEP 5 e freezzi con Tier 1+2+3. (2) Se peggio, freezzi con Tier 1+2. (3) Regola suprema: STEP 6 deve succedere entro il **7 giugno** con qualsiasi tier. Modello imperfetto > nessun modello al torneo.

**R8 — Data leakage tardivo**
Trigger: log-loss < 0.90 sul validation (sospetto), o test di leakage fallisce. Stop immediato, bisection feature, fix, re-training completo, aggiunta caso a `test_leakage.py` come regressione. Protezione: soglia `log_loss < 0.92` triggera warning nel report.

### 10.3 Piani arancioni

**R2 — No historical injuries**
Bootstrap a zero per tutti i match storici (feature=0 in training). Il Tier 4 impara a non usarla molto in training, ma diventa attiva in live quando popolata dall'utente. Alternativa: Tier 4 escluso dal training e applicato come post-hoc adjustment manuale.

**R6 — Rose post-kickoff**
Update `rosters_wc2026.json` al volo; il market value si ricalcola al prossimo `predict`. Nessun retraining.

**R7 — Predizioni assurde in dry run**
Invarianti in `test_markets.py`:
- Somma P(1)+P(X)+P(2) = 1.000 ± 0.001
- Swap home/away swappa P(1) e P(2) in modo coerente
- P(France vs San Marino win) > 0.85

**R9 — Nomi squadre inconsistenti**
`TEAM_NAME_MAPPING` in `config.py` con alias noti (Czechia↔Czech Republic, Türkiye↔Turkey, ecc.). `validate_team_names()` logga WARN su nomi non mappati al load di ogni sorgente.

### 10.4 Piano B "minimo vitale"

Se tutti gli extra crollano: XGBoost Poisson **Tier 1+2 calibrato** addestrato su 2002-2022, testato su WC2022, CLI `predict` funzionante. Niente Transfermarkt, niente injuries, niente LSTM, niente news, niente Telegram. Ma **progetto finito, onesto, con log-loss reale su 64 match del Mondiale 2026**.

## 11. Success Criteria

### 11.1 KPI primari (misurati a fine torneo, ~15 luglio 2026)

- **Log-loss 1/X/2** sui 64 match WC2026 (target test): **< 0.97**. Sotto 0.95 = eccellente. Sotto 0.93 = straordinario. Nota: target di validation (2018-2022) è più aggressivo (< 0.95) perché il test set WC2026 ha rumore + drift aggiuntivo.
- **Brier score** 1/X/2 calibrato
- **Reliability diagram** che mostri calibrazione decente
- **Log-loss su mercati derivati**: O/U 2.5, BTTS
- **Confronto esplicito con baseline**: prior storico, Elo-only, e (se disponibili) quote bookmaker pubbliche come benchmark esterno

### 11.2 KPI processo

- Freeze di `v1_final` entro il 7 giugno 2026
- `test_leakage.py` verde su ogni training
- Ogni training produce un report archiviato
- `models/v1_final/metadata.json` include commit hash, tier attivi, iperparametri, log-loss train/val/test, data di training

### 11.3 KPI opzionali (Fase B)

- **STEP 8 LSTM**: v2_lstm batte v1_final ≥0.005 log-loss su WC2022 → integrato in ensemble; altrimenti archiviato con documentazione
- **STEP 9 News**: Tier 5 passa soglia 0.003 → integrato; altrimenti archiviato
- **STEP 10 Ensemble**: v3 batte miglior base ≥0.003 → diventa modello live; altrimenti resta v1_final (o il singolo miglior modello)
- **STEP 11 Telegram**: bot online prima del kickoff

## 12. Invarianti che proteggono il progetto

Questi sono i principi non negoziabili su cui tutto il design è costruito:

1. **Baseline-first**: ogni aggiunta di complessità deve battere il baseline precedente di ≥0.003 log-loss. Altrimenti documentata e rimossa.
2. **Temporal split only**: nessuno split random, mai. Training strettamente anteriore a validation strettamente anteriore a test.
3. **Test set toccato una sola volta**: WC2022 si tocca in STEP 6 e basta, il numero che esce è quello del progetto.
4. **No model changes during tournament**: dall'11 giugno al 19 luglio 2026, zero modifiche a modello/iperparametri/feature. Qualsiasi tweak inquina la metrica finale.
5. **No one-hot encoding delle squadre**: entrano solo via Elo e (Tier 3) market value. Protegge la generalizzazione.
6. **Calibrazione obbligatoria**: modelli non calibrati non entrano in `models/v1_final/`.
7. **Fase A blindata**: nessun esperimento di Fase B tocca il codice core prima di `v1_final` freeze.
8. **Determinismo**: `random_state=42` ovunque, ogni run riproducibile.

---

**Fine dello spec.** Prossimo passaggio: implementation plan via `writing-plans` skill.
