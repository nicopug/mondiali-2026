# WC2026 — Baseline Elo-only vs modello (72 partite gironi)

**Generato:** 2026-06-28  
**Train:** 49405 partite internazionali con `date < 2026-06-11` (leak-free)  
**Test:** 72 partite dei gironi WC2026 (20 pareggi, 28%)  
**Baseline:** `EloLogisticBaseline` — feature `[elo_diff, neutral]`, C=1.0, random_state=42  

> Domanda: i tier 2/3 dell'XGBoost aggiungono valore *oltre* al puro Elo? Battere il random e' un'asticella bassa; battere l'Elo nudo, no.

| Modello | log-loss 1X2 | Brier | edge vs random |
|---|---|---|---|
| Random uniforme | 1.0986 | — | +0.0000 |
| Elo-only (ex-ante, frozen) | 0.9033 | 0.5378 | +0.1954 |
| Elo-only (live pre-match) | 0.9075 | 0.5408 | +0.1911 |
| Modello v1_final (congelato) | 1.0011 | 0.5814 | +0.0975 |

**Verdetto (confronto equo, ex-ante vs ex-ante):** Il modello NON batte l'Elo-only ex-ante (lo perde di +0.0979 nats).  

## Significativita' (bootstrap appaiato, modello − Elo)

- Differenza media log-loss per partita: **+0.0979** (>0 = modello peggiore)  
- Bootstrap 95% CI (10k, seed 42): **[+0.0282, +0.1803]**  
- P(modello peggiore dell'Elo) = **99.8%**  

Lo zero e' fuori dall'intervallo: il distacco non e' rumore da campione piccolo.

## Dove perde il modello

Tasso pareggi reale **28%**; P(X) media — modello **19%**, Elo **21%**. Entrambi sottostimano i pari (tratto strutturale del Poisson), il modello di piu'.

| Esito reale | n | log-loss modello | log-loss Elo |
|---|---|---|---|
| Vittoria casa | 33 | 0.769 | 0.619 |
| Pareggio | 20 | 1.789 | 1.622 |
| Vittoria fuori | 19 | 0.575 | 0.640 |

Il modello sanguina su **vittorie casa** e **pareggi**; recupera solo sulle trasferte. I tier 2/3 (forma, valori di mercato) sembrano aggiungere rumore, non segnale, su questo campione.

Note metodologiche:
- *ex-ante (frozen)* = stesso set informativo delle predizioni congelate del modello (Elo pre-torneo, una sola volta). E' il confronto corretto.
- *live* = l'Elo assorbe le giornate gia' giocate; vantaggio informativo non disponibile al modello congelato, quindi solo come riferimento.
- Soglia storica del progetto: l'XGBoost Tier 1 deve battere questo baseline di almeno 0.003 in log-loss, altrimenti STOP/debug.