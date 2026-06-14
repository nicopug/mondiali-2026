# WC2026 — Live scoring (ex-ante vs reale)

**Generato:** 2026-06-14  
**Predizioni:** `reports/wc2026_groups_predictions.csv` (congelate 2026-05-16)  
**Risultati:** `data/raw/results.csv` (martj42)  
**Supplemento manuale:** `data/wc2026/manual_results.csv` (4 partite non ancora su martj42)  
**Partite valutate:** 8

> Leak-free: si valutano solo le probabilita' ex-ante, mai ri-predette.

## Metriche aggregate

| Mercato | log-loss modello | log-loss baseline uniforme | edge | Brier |
|---|---|---|---|---|
| 1X2 | 1.2238 | 1.0986 | -0.1252 | 0.7551 |
| Over/Under 2.5 | 1.1871 | 0.6931 | -0.4939 | 0.4625 |
| BTTS | 0.6935 | 0.6931 | -0.0003 | 0.2488 |

`edge` positivo = il modello batte la predizione casuale; negativo = peggio del random.

## Dettaglio partite

| Data | Partita | Risultato | P(H/X/A) | Esito | P(esito) | LL 1X2 |
|---|---|---|---|---|---|---|
| 2026-06-11 | Mexico–South Africa | 2-0 | 0.72/0.16/0.12 | H | 0.72 | 0.327 |
| 2026-06-11 | South Korea–Czech Republic | 2-1 | 0.39/0.24/0.37 | H | 0.39 | 0.935 |
| 2026-06-12 | Canada–Bosnia and Herzegovina | 1-1 | 0.55/0.20/0.25 | D | 0.20 | 1.607 |
| 2026-06-12 | United States–Paraguay | 4-1 | 0.31/0.24/0.45 | H | 0.31 | 1.156 |
| 2026-06-13 | Qatar–Switzerland | 1-1 | 0.05/0.09/0.86 | D | 0.09 | 2.361 |
| 2026-06-13 | Brazil–Morocco | 1-1 | 0.46/0.24/0.30 | D | 0.24 | 1.429 |
| 2026-06-13 | Haiti–Scotland | 0-1 | 0.14/0.17/0.68 | A | 0.68 | 0.380 |
| 2026-06-13 | Australia–Turkey | 2-0 | 0.20/0.18/0.61 | H | 0.20 | 1.596 |

**Hit-rate 1X2** (esito reale sopra 1/3 di probabilita'): 38%
