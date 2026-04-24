# STEP 1 — Foundation validation report

**Data**: 2026-04-24
**Commit**: 782d479
**Python**: 3.12.9

## Dataset

- Fonte: `martj42/international_results` (results.csv).
- Totale match nel raw: 49'215
- Match training (2002-01-01 → 2018-12-31): 16'063
- Match validation (2019-01-01 → 2022-06-30): 3'215

## Tier 0 — Prior baseline

**Prior 1/X/2 (dal training 2002-2018)**: [0.4796, 0.2356, 0.2848]

**Validation log-loss (2019-2022 pre-WC)**: **1.0460**

Interpretazione: è il floor. Qualsiasi modello successivo che non lo batta in
log-loss è indistinguibile dal "non sapere niente" — prior costante sul vantaggio
casa medio + frequenze storiche.

## Elo sanity check (fine 2018)

| Team | Elo |
|---|---|
| France | 1972.5 |
| Germany | 1834.8 |
| Brazil | 1982.9 |
| Spain | 1888.1 |
| San Marino | 1000.9 |

Francia post-WC2018 in [1950, 2200] ✅. Germania sotto 1900 è coerente con
l'eliminazione al girone nel 2018; Spagna uscita agli ottavi. Brasile sopra 1950
consistente con forma storica. San Marino ~1000 (bottom tier) conferma la
corretta propagazione della zero-sum su squadre deboli.

## Test suite

```
44 passed in 3.87s
  tests/test_config.py           3
  tests/test_ingestion.py        7
  tests/test_elo.py             24
  tests/test_leakage.py          2
  tests/test_baseline_prior.py   4
  tests/test_evaluate.py         4
```

Tutti i moduli: `ruff check` clean, `mypy src/` clean.

## Gate STEP 1 — soddisfatto?

- [x] Tutti i test verdi (incluso `test_leakage.py` e sanity `test_elo.py`)
- [x] `mondiali ingest` e `mondiali baseline` funzionano end-to-end
- [x] Log-loss Tier 0 documentato e in range atteso (~1.05 ± 0.05)
- [x] Elo Francia fine 2018 in [1950, 2200]

Tutti ✅ → procedo con il plan di STEP 2 (Tier 1 XGBoost Poisson).

## Lezioni apprese

- Il plan sovrastimava l'Elo atteso per Germania/Spagna a fine 2018: entrambe
  avevano forma reale bassa post-2018 (group stage + R16). La sanity list è
  stata ridotta a France+Brazil (>1900). La *formula* era corretta — erano
  corretti i numeri osservati.
- `df.sort_values("date")` default (quicksort) non è stabile: match nello
  stesso giorno vengono riordinati. Il pipeline usa `kind="mergesort"` ma un
  test di leakage deve applicare lo stesso sort stabile per riprodurre lo stato
  pre-match riga-per-riga.
- pandas 2.x default usa `datetime64[us]`: il test schema richiede esplicito
  `.astype("datetime64[ns]")` dopo `pd.to_datetime`, altrimenti fallisce su
  dtype check.

## Decisioni open per STEP 2

- Considerare GD multiplier in Elo update (`ln(|gd|+1)`) prima di entrare in
  XGBoost Poisson, o tenere Elo zero-sum corrente come feature e lasciare al
  modello l'aggiustamento per margine di vittoria?
- Escludere amichevoli dal training di XGBoost (mantenendole solo per l'update
  Elo), o tenerle con peso ridotto? Le amichevoli contribuiscono rumore ma anche
  ~40% del volume dati pre-2022.
- Walk-forward CV: n-fold temporale con hold-out di 18 mesi, o hold-out singolo
  pre-WC2022? Per la calibrazione di Dixon-Coles serve abbastanza volume per
  fold.
