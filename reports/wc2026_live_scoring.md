# WC2026 — Live scoring (ex-ante vs reale)

**Generato:** 2026-06-13  
**Predizioni:** `reports/wc2026_groups_predictions.csv` (congelate 2026-05-16)  
**Risultati:** `data/raw/results.csv` (martj42)  
**Partite valutate:** 4

> Leak-free: si valutano solo le probabilita' ex-ante, mai ri-predette.

## Metriche aggregate

| Mercato | log-loss modello | log-loss baseline uniforme | edge | Brier |
|---|---|---|---|---|
| 1X2 | 1.0061 | 1.0986 | +0.0925 | 0.6019 |
| Over/Under 2.5 | 0.9498 | 0.6931 | -0.2567 | 0.3561 |
| BTTS | 0.5346 | 0.6931 | +0.1586 | 0.1757 |

`edge` positivo = il modello batte la predizione casuale; negativo = peggio del random.

## Dettaglio partite

| Data | Partita | Risultato | P(H/X/A) | Esito | P(esito) | LL 1X2 |
|---|---|---|---|---|---|---|
| 2026-06-11 | Mexico–South Africa | 2-0 | 0.72/0.16/0.12 | H | 0.72 | 0.327 |
| 2026-06-11 | South Korea–Czech Republic | 2-1 | 0.39/0.24/0.37 | H | 0.39 | 0.935 |
| 2026-06-12 | Canada–Bosnia and Herzegovina | 1-1 | 0.55/0.20/0.25 | D | 0.20 | 1.607 |
| 2026-06-12 | United States–Paraguay | 4-1 | 0.31/0.24/0.45 | H | 0.31 | 1.156 |

**Hit-rate 1X2** (esito reale sopra 1/3 di probabilita'): 50%
