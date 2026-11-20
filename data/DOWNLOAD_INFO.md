# Data Download Record

| Item | Value                                                                                          |
|:-----|:-----------------------------------------------------------------------------------------------|
| Target stock | MSFT (Microsoft Corporation)                                                                   |
| Data source | Yahoo Finance, via `python download_data.py --target MSFT`                                     |
| **Download date** | 17 November 2026                                                                               |
| Downloaded by | Ida Aluso Masheti                                                                              |
| Date range | 2015-01-02 to 2026-11-09                                                                       |
| Trading days | 2,981                                                                                          |
| Predictors | SP500 (^GSPC), TECH (XLK), VIX (^VIX), US10Y (^TNX), USD (DX-Y.NYB), BRENT (BZ=F), GOLD (GC=F) |
| Files created | `data/target_MSFT.csv`, `data/predictors.csv`                                                  |
| `config.py` setting | `TARGET_TICKER = "MSFT"` (`INITIAL_TRAIN` unchanged: history is about 11.9 years)              |

## Data quality report

| Check | Result |
|:------|:-------|
| rows | 2981 |
| first_date | 2015-01-02 |
| last_date | 2026-11-09 |
| duplicate_dates | 0 |
| missing_values | 0 |
| non_positive_prices | 0 |
| days_with_zero_volume | 0 |
| high_below_low | 0 |
| longest_gap_in_days | 4 |
| days_with_move_over_10pct | 6 |

## Moves over 10 percent

| Date            | Move | Likely cause |
|:----------------|-----:|:-------------|
| [date]          | [value] | [to be confirmed by Member 2] |
| 2015-04-24      | 0.104523| |
| 2015-10-23      | 0.100771| |
| 2020-03-13      | 0.142169| |
| 2020-03-16      | -0.147390| |
| 2025-04-09     | 0.101337| |
| 2026-07-30     | 0.155067| |
