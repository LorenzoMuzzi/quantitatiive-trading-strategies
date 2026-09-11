# Quantitative Trading Strategies

This repository contains three quantitative trading and portfolio-selection strategies implemented in Python:

- **ANTICOR**, including an optional long/short configuration;
- **OLMAR** (Online Moving Average Reversion);
- **PCA-based statistical arbitrage**.

The project was developed for educational and research purposes. Its main objective is to translate the mathematical logic of the strategies into working Python implementations, apply them to historical equity data, and compare their behaviour under different parameter and transaction-cost assumptions.

The repository also includes a reduced Dow Jones dataset that can be used to test all three implementations without uploading the much larger original data files.

## Strategies

| File | Strategy | Main idea | Main output |
| --- | --- | --- | --- |
| `Anticor_short.py` | ANTICOR | Reallocates portfolio weights by analysing lagged cross-correlations between asset returns. The implementation can also allow short positions and cap gross leverage. | Wealth series, performance metrics and optional portfolio weights for multiple window sizes. |
| `olmar.py` | OLMAR | Predicts price relatives using moving-average reversion and updates the portfolio through a projection onto the probability simplex. | Wealth series, concentration, turnover and portfolio weights for multiple window sizes. |
| `StatArb.py` | PCA statistical arbitrage | Uses rolling principal component analysis to extract common factors and trades mean-reverting residuals. | A simulated portfolio-equity series and, when the original main block is used, a PDF comparison report. |

The implementations include simplified transaction-cost modelling. ANTICOR also supports a long/short mode and a gross-leverage limit.

## Repository structure

```text
quantitative-trading-strategies/
├── README.md
├── Anticor_short.py
├── olmar.py
├── StatArb.py
└── data/
    └── DJI30_sample.xlsx
```

## Test dataset

`data/DJI30_sample.xlsx` is a reduced version of the supplied Dow Jones workbook. It preserves the original workbook structure while limiting the price history to the most recent **420 trading observations**, from **16 September 2024 to 19 May 2026**.

The sample contains:

- adjusted closing prices for 30 Dow Jones constituents;
- the SPDR Dow Jones Industrial Average ETF (`^DIA`) as the benchmark;
- security names and exchange metadata;
- enough observations to run the 252-day PCA window used by `StatArb.py`.

The workbook contains the following sheets:

| Sheet | Contents |
| --- | --- |
| `Close_ok` | Cleaned price matrix used by the three strategy scripts. The first column is named `data`, and each remaining column is an equity ticker. |
| `Benchmark` | Benchmark prices for `^DIA`, indexed by `data`. |
| `Anagrafe_ok` | Security metadata with dates stored as Excel dates. |
| `AdjClose` | Adjusted closing prices for the equities and the benchmark, with dates stored in `YYYYMMDD` form. |
| `Anagrafe` | Original security metadata and listing dates. |

The order of the sheets is intentional: `Close_ok` is the first sheet because the current scripts load the first worksheet by default.

## Requirements

- Python 3.12 or later
- NumPy
- pandas
- Matplotlib
- openpyxl

Install the dependencies with:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Then install the required packages:

```bash
python -m pip install numpy pandas matplotlib openpyxl
```

## Running the strategies with the sample dataset

The `__main__` sections of the scripts were originally configured for three complete local datasets. The included sample has a different filename, so the most direct way to test it is to import each strategy as shown below.

Run the examples from the repository root.

### ANTICOR

```python
from Anticor_short import run_anticor

wealth, strategy_metrics, weights = run_anticor(
    "data/DJI30_sample.xlsx",
    min_w=3,
    max_w=30,
    transaction_cost_rate=0.0019,
    short=False,
)

print(wealth.tail())
print(strategy_metrics)
```

To test the long/short version with a gross-leverage cap:

```python
from Anticor_short import run_anticor

wealth, strategy_metrics, weights = run_anticor(
    "data/DJI30_sample.xlsx",
    min_w=3,
    max_w=30,
    transaction_cost_rate=0.0019,
    short=True,
    max_gross_leverage=2.0,
    save_weights=True,
)
```

### OLMAR

```python
from olmar import run_olmar

wealth, concentration, turnover, weights = run_olmar(
    "data/DJI30_sample.xlsx",
    transaction_cost_rate=0.0019,
)

print(wealth.tail())
print("Concentration:", concentration)
print("Turnover:", turnover)
```

OLMAR evaluates moving-average windows from 3 to 30 trading days.

### PCA statistical arbitrage

```python
import pandas as pd

from StatArb import run_statarb

prices = pd.read_excel(
    "data/DJI30_sample.xlsx",
    sheet_name="Close_ok",
    index_col="data",
)

equity = run_statarb(
    prices,
    fixed=True,
    transaction_cost=0.0005,
)

print(equity[-10:])
```

The statistical-arbitrage implementation uses a 252-day rolling PCA window and a 60-day residual-estimation window. The sample therefore produces a shorter out-of-sample equity series than the full dataset.

## Input-data format

To use another dataset, create an Excel workbook whose first sheet has the following structure:

| data | TICKER_1 | TICKER_2 | ... |
| --- | ---: | ---: | ---: |
| 2024-01-02 | 100.25 | 54.80 | ... |
| 2024-01-03 | 101.10 | 55.15 | ... |

Requirements:

- the date-index column must be named `data`;
- each other column must contain the price history of one asset;
- values must be numeric and strictly positive when the asset is active;
- dates must be in ascending order;
- the workbook must contain a sheet named `Benchmark`, also indexed by `data`;
- at least 253 observations are required by the current PCA strategy, although a longer history is preferable.

## Outputs

When the functions are imported directly, they return pandas objects or NumPy-compatible sequences that can be analysed, plotted or exported separately.

The original script entry points are additionally configured to create:

- `Report_Anticor_short.xlsx` for ANTICOR;
- `Report_Olmar_epsilon1.xlsx` for OLMAR;
- `StatArb_report.pdf` for PCA statistical arbitrage.

Those entry points expect the complete datasets listed in each script. They will skip or fail on missing files unless the `df_names` lists are updated.

## Validation

The included sample workbook was checked with:

- `pandas.read_excel(..., index_col="data")` on `Close_ok`;
- benchmark loading from the `Benchmark` sheet;
- ANTICOR over test windows 3 to 5;
- all 28 OLMAR windows;
- the PCA statistical-arbitrage function with its 252-day window.

The validation produced aligned price and benchmark matrices with 420 rows and no non-finite price values.

## Methodological limitations

These programs are research prototypes, not production trading systems. In particular:

- historical backtests do not guarantee future performance;
- transaction costs are simplified;
- market impact, bid-ask spreads, liquidity limits, taxes and short-borrow constraints are not fully modelled;
- the constituent universe may introduce survivorship or selection bias;
- parameter choices can materially affect the results;
- no live execution, order management or risk-control infrastructure is included.

The code and data are provided for educational purposes only and do not constitute financial or investment advice.

## References

- A. Borodin, R. El-Yaniv and V. Gogan, [*Can We Learn to Beat the Best Stock?*](https://www.jair.org/index.php/jair/article/view/10380/24854), Journal of Artificial Intelligence Research, 2004.
- B. Li and S. C. H. Hoi, [*On-Line Portfolio Selection with Moving Average Reversion*](https://arxiv.org/abs/1206.4626), 2012.
- M. Avellaneda and J.-H. Lee, [*Statistical Arbitrage in the U.S. Equities Market*](https://math.nyu.edu/~avellane/AvellanedaLeeStatArb20090616.pdf), 2009.

## Author

**Lorenzo Muzzi**  
[GitHub profile](https://github.com/LorenzoMuzzi)
