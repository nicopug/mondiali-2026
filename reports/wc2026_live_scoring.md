# WC2026 — Live scoring (ex-ante vs reale)

**Generato:** 2026-06-28  
**Predizioni:** `reports/wc2026_groups_predictions.csv` (congelate 2026-05-16)  
**Risultati:** `data/raw/results.csv` (martj42)  
**Supplemento manuale:** `data/wc2026/manual_results.csv` (12 partite non ancora su martj42)  
**Partite valutate:** 72

> Leak-free: si valutano solo le probabilita' ex-ante, mai ri-predette.

## Metriche aggregate

| Mercato | log-loss modello | log-loss baseline uniforme | edge | Brier |
|---|---|---|---|---|
| 1X2 | 1.0011 | 1.0986 | +0.0975 | 0.5814 |
| Over/Under 2.5 | 0.7870 | 0.6931 | -0.0938 | 0.2822 |
| BTTS | 0.7135 | 0.6931 | -0.0203 | 0.2587 |

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
| 2026-06-21 | Belgium–Iran | 0-0 | 0.34/0.23/0.44 | D | 0.23 | 1.487 |
| 2026-06-21 | New Zealand–Egypt | 1-3 | 0.29/0.22/0.49 | A | 0.49 | 0.710 |
| 2026-06-21 | Spain–Saudi Arabia | 4-0 | 0.60/0.21/0.19 | H | 0.60 | 0.519 |
| 2026-06-21 | Uruguay–Cape Verde | 2-2 | 0.75/0.14/0.10 | D | 0.14 | 1.945 |
| 2026-06-22 | France–Iraq | 3-0 | 0.77/0.14/0.10 | H | 0.77 | 0.267 |
| 2026-06-22 | Norway–Senegal | 3-2 | 0.48/0.22/0.30 | H | 0.48 | 0.725 |
| 2026-06-22 | Argentina–Austria | 2-0 | 0.64/0.19/0.17 | H | 0.64 | 0.446 |
| 2026-06-22 | Jordan–Algeria | 1-2 | 0.33/0.24/0.43 | A | 0.43 | 0.848 |
| 2026-06-23 | Portugal–Uzbekistan | 5-0 | 0.55/0.21/0.24 | H | 0.55 | 0.596 |
| 2026-06-23 | Colombia–DR Congo | 1-0 | 0.68/0.19/0.13 | H | 0.68 | 0.390 |
| 2026-06-23 | England–Ghana | 0-0 | 0.69/0.17/0.14 | D | 0.17 | 1.775 |
| 2026-06-23 | Panama–Croatia | 0-1 | 0.38/0.23/0.38 | A | 0.38 | 0.959 |
| 2026-06-24 | Mexico–Czech Republic | 3-0 | 0.51/0.22/0.27 | H | 0.51 | 0.678 |
| 2026-06-24 | South Africa–South Korea | 1-0 | 0.13/0.18/0.68 | H | 0.13 | 2.029 |
| 2026-06-24 | Canada–Switzerland | 1-2 | 0.27/0.22/0.51 | A | 0.51 | 0.675 |
| 2026-06-24 | Bosnia and Herzegovina–Qatar | 3-1 | 0.55/0.20/0.25 | H | 0.55 | 0.598 |
| 2026-06-24 | Scotland–Brazil | 0-3 | 0.15/0.18/0.67 | A | 0.67 | 0.400 |
| 2026-06-24 | Morocco–Haiti | 4-2 | 0.66/0.18/0.15 | H | 0.66 | 0.412 |
| 2026-06-25 | United States–Turkey | 2-3 | 0.18/0.19/0.63 | A | 0.63 | 0.464 |
| 2026-06-25 | Paraguay–Australia | 0-0 | 0.34/0.23/0.43 | D | 0.23 | 1.475 |
| 2026-06-25 | Curaçao–Ivory Coast | 0-2 | 0.01/0.04/0.95 | A | 0.95 | 0.048 |
| 2026-06-25 | Ecuador–Germany | 2-1 | 0.51/0.23/0.27 | H | 0.51 | 0.679 |
| 2026-06-25 | Japan–Sweden | 1-1 | 0.37/0.22/0.41 | D | 0.22 | 1.512 |
| 2026-06-25 | Tunisia–Netherlands | 1-3 | 0.14/0.18/0.68 | A | 0.68 | 0.380 |
| 2026-06-26 | Egypt–Iran | 1-1 | 0.22/0.22/0.56 | D | 0.22 | 1.510 |
| 2026-06-26 | New Zealand–Belgium | 1-5 | 0.16/0.18/0.66 | A | 0.66 | 0.417 |
| 2026-06-26 | Cape Verde–Saudi Arabia | 0-0 | 0.28/0.23/0.49 | D | 0.23 | 1.484 |
| 2026-06-26 | Uruguay–Spain | 0-1 | 0.24/0.22/0.53 | A | 0.53 | 0.629 |
| 2026-06-26 | Norway–France | 1-4 | 0.18/0.19/0.63 | A | 0.63 | 0.457 |
| 2026-06-26 | Senegal–Iraq | 5-0 | 0.49/0.22/0.29 | H | 0.49 | 0.713 |
| 2026-06-27 | Jordan–Argentina | 1-3 | 0.14/0.17/0.69 | A | 0.69 | 0.376 |
| 2026-06-27 | Austria–Algeria | 3-3 | 0.53/0.23/0.24 | D | 0.23 | 1.460 |
| 2026-06-27 | Colombia–Portugal | 0-0 | 0.38/0.24/0.37 | D | 0.24 | 1.407 |
| 2026-06-27 | DR Congo–Uzbekistan | 3-1 | 0.22/0.20/0.58 | H | 0.22 | 1.510 |
| 2026-06-27 | Panama–England | 0-2 | 0.23/0.21/0.56 | A | 0.56 | 0.576 |
| 2026-06-27 | Croatia–Ghana | 2-1 | 0.49/0.22/0.29 | H | 0.49 | 0.711 |

**Hit-rate 1X2** (esito reale sopra 1/3 di probabilita'): 61%
