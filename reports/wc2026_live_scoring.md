# WC2026 — Live scoring (ex-ante vs reale)

**Generato:** 2026-06-21  
**Predizioni:** `reports/wc2026_groups_predictions.csv` (congelate 2026-05-16)  
**Risultati:** `data/raw/results.csv` (martj42)  
**Partite valutate:** 36

> Leak-free: si valutano solo le probabilita' ex-ante, mai ri-predette.

## Metriche aggregate

| Mercato | log-loss modello | log-loss baseline uniforme | edge | Brier |
|---|---|---|---|---|
| 1X2 | 1.1337 | 1.0986 | -0.0351 | 0.6632 |
| Over/Under 2.5 | 0.8848 | 0.6931 | -0.1917 | 0.3207 |
| BTTS | 0.6933 | 0.6931 | -0.0001 | 0.2489 |

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
| 2026-06-14 | Germany–Curaçao | 7-1 | 0.85/0.10/0.05 | H | 0.85 | 0.162 |
| 2026-06-14 | Ivory Coast–Ecuador | 1-0 | 0.14/0.16/0.70 | H | 0.14 | 1.987 |
| 2026-06-14 | Netherlands–Japan | 2-2 | 0.55/0.21/0.23 | D | 0.21 | 1.538 |
| 2026-06-14 | Sweden–Tunisia | 5-1 | 0.40/0.24/0.35 | H | 0.40 | 0.905 |
| 2026-06-15 | Belgium–Egypt | 1-1 | 0.40/0.24/0.36 | D | 0.24 | 1.414 |
| 2026-06-15 | Iran–New Zealand | 2-2 | 0.52/0.21/0.27 | D | 0.21 | 1.563 |
| 2026-06-15 | Spain–Cape Verde | 0-0 | 0.87/0.08/0.05 | D | 0.08 | 2.474 |
| 2026-06-15 | Saudi Arabia–Uruguay | 1-1 | 0.15/0.17/0.68 | D | 0.17 | 1.749 |
| 2026-06-16 | France–Senegal | 3-1 | 0.64/0.19/0.18 | H | 0.64 | 0.452 |
| 2026-06-16 | Iraq–Norway | 1-4 | 0.16/0.18/0.66 | A | 0.66 | 0.414 |
| 2026-06-16 | Argentina–Algeria | 3-0 | 0.73/0.15/0.11 | H | 0.73 | 0.311 |
| 2026-06-16 | Austria–Jordan | 3-1 | 0.46/0.23/0.31 | H | 0.46 | 0.782 |
| 2026-06-17 | Portugal–DR Congo | 1-1 | 0.63/0.20/0.17 | D | 0.20 | 1.605 |
| 2026-06-17 | Uzbekistan–Colombia | 1-3 | 0.17/0.20/0.63 | A | 0.63 | 0.460 |
| 2026-06-17 | England–Croatia | 4-2 | 0.53/0.22/0.25 | H | 0.53 | 0.634 |
| 2026-06-17 | Ghana–Panama | 1-0 | 0.19/0.19/0.62 | H | 0.19 | 1.669 |
| 2026-06-18 | Czech Republic–South Africa | 1-1 | 0.64/0.19/0.16 | D | 0.19 | 1.640 |
| 2026-06-18 | Mexico–South Korea | 1-0 | 0.47/0.24/0.29 | H | 0.47 | 0.759 |
| 2026-06-18 | Switzerland–Bosnia and Herzegovina | 4-1 | 0.67/0.18/0.16 | H | 0.67 | 0.408 |
| 2026-06-18 | Canada–Qatar | 6-0 | 0.74/0.14/0.12 | H | 0.74 | 0.304 |
| 2026-06-19 | Scotland–Morocco | 0-1 | 0.26/0.22/0.52 | A | 0.52 | 0.656 |
| 2026-06-19 | Brazil–Haiti | 3-0 | 0.77/0.13/0.10 | H | 0.77 | 0.262 |
| 2026-06-19 | United States–Australia | 2-0 | 0.31/0.22/0.47 | H | 0.31 | 1.187 |
| 2026-06-19 | Turkey–Paraguay | 0-1 | 0.57/0.20/0.22 | A | 0.22 | 1.501 |
| 2026-06-20 | Germany–Ivory Coast | 2-1 | 0.49/0.22/0.30 | H | 0.49 | 0.722 |
| 2026-06-20 | Ecuador–Curaçao | 0-0 | 0.98/0.01/0.00 | D | 0.01 | 4.350 |
| 2026-06-20 | Netherlands–Sweden | 5-1 | 0.59/0.21/0.21 | H | 0.59 | 0.534 |
| 2026-06-20 | Tunisia–Japan | 0-4 | 0.23/0.21/0.56 | A | 0.56 | 0.581 |

**Hit-rate 1X2** (esito reale sopra 1/3 di probabilita'): 53%
