import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

from matplotlib.backends.backend_pdf import PdfPages

df_names = ["#DJI30_20150101_20260519.xlsx", "#NDX100_20150101_20260519.xlsx", "#SP500_20150101_20260519.xlsx"]
df_titles = ["Dow Jones Industrial Average", "Nasdaq 100", "S&P 500"]
graph_titles = ["", "No transaction costs", "Transaction costs"]

graphs = {}
WINDOW = 60
PCA_WINDOW = 252
T_COST = 0.0005
MIN_K = 256/30
STARTING_EQUITY = 1
LEVERAGE = 2

def run_statarb(prices: pd.DataFrame, fixed: bool = True, explained_var: float = 0.55, transaction_cost: float = 0.0):
    n_eigenvalues = 15
    portfolio_equity = ([0] * PCA_WINDOW) + [STARTING_EQUITY]

    price_matrix = prices.to_numpy()
    transactions = {}

    rend = prices.pct_change()
    rend = rend.replace([np.inf, -np.inf], np.nan)
    rend = rend.fillna(0).to_numpy()[1:, :]
    
    rend_mean = rend.mean(axis=0)
    rend_std = rend.std(axis=0, ddof=1)

    safe_std = np.where(rend_std > 0, rend_std, 1.0)

    standardized = (rend - rend_mean) / safe_std

    days, n_stocks = standardized.shape

    shares_exposure = np.zeros(n_stocks)
    pcas = np.ones(shape=(n_stocks, n_eigenvalues))
    positions = np.zeros(shape=(2, n_stocks))

    # asset_exposure = np.zeros(n_stocks)

    asset_list = list(prices.columns)

    cash = STARTING_EQUITY

    for t in range(days - PCA_WINDOW + 1):
        today_t_cost = 0
        window_matrix = standardized[t: t + PCA_WINDOW]
        short_matrix = rend[t + PCA_WINDOW - WINDOW : t + PCA_WINDOW]

        # CALCOLO EQUITY IN t
        if t != 0:
            equity_t = cash + shares_exposure @ price_today #- T_COST * np.sum(np.abs(operations) * price_today)
            portfolio_equity.append(equity_t)

        else:
            equity_t = portfolio_equity[-1]
        
        if t == days - PCA_WINDOW:
            return portfolio_equity[PCA_WINDOW:] # QUESTO E' IL RETURN FINALE -----------------------------------------------------------------------------------------------------------------
        
        price_today = price_matrix[t + PCA_WINDOW, :]

        # FILTRO TITOLI FUORI DA INDICE

        not_nan = ~ np.any(rend[t: t + PCA_WINDOW] == 0, axis=0)

        window_matrix = window_matrix[:, not_nan]
        short_matrix = short_matrix[:, not_nan]

        #  CALCOLO NUOVI PCAs
        corr_matrix = np.corrcoef(window_matrix, rowvar=False)
        std_matrix = window_matrix.std(axis=0, ddof = 1)

        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if not fixed:
            n_eigenvalues = (np.cumsum(eigenvalues) < explained_var * n_stocks).sum() + 1 # questo perchè la traccia della matrice delle correlazioni == numero titoli

        pcas = eigenvectors[:, :n_eigenvalues]

        eigenportfolios = pcas / std_matrix[:, None]
        eigenportfolio_weights = eigenportfolios / np.abs(eigenportfolios).sum(axis=0)

        factors_hist = short_matrix @ eigenportfolios

        # REGRESSIONE SU PCA
        X_reg = np.column_stack([np.ones(WINDOW), factors_hist])
        coef = np.linalg.inv(X_reg.T @ X_reg) @ (X_reg.T @ short_matrix)

        betas = coef[1:, :]

        eps = short_matrix - (X_reg @ coef)
        
        # REGRESSIONE SU ERRORI
        X = np.cumsum(eps, axis=0)
        
        X_lag = X[:-1, :]
        X_next = X[1:, :]

        lag_mean = X_lag.mean(axis=0)
        next_mean = X_next.mean(axis=0)

        num = ((X_lag - lag_mean) * (X_next - next_mean)).sum(axis=0)
        den = ((X_lag - lag_mean) ** 2).sum(axis=0)

        b = num / den
        a = next_mean - b * lag_mean
        residual = X_next - (a + b * X_lag)

        k = -np.log(b) * 252
        m_raw = a / (1 - b)

        with np.errstate(invalid="ignore", divide="ignore"): # Solo per non vedere un RuntimeWarning nel terminale
            sigma_eq = np.sqrt(residual.var(axis=0, ddof=1) / (1 - b ** 2))

        valid = (np.isfinite(k) & np.isfinite(sigma_eq) & (b > 0) & (b < np.exp(- MIN_K / 252)) & (k > MIN_K) & (sigma_eq > 0))

        score_ok = (np.isfinite(m_raw) & np.isfinite(sigma_eq) & (sigma_eq > 0))
        
        m = m_raw.copy()
        m[score_ok] = m_raw[score_ok] - np.nanmean(m_raw[score_ok]) # media centrata

        s_scores = np.full_like(b, np.nan, dtype=float)
        s_scores[score_ok] = -m[score_ok] / sigma_eq[score_ok]

        not_nan_idx = np.where(not_nan)[0]
        positions_t = positions[:, not_nan_idx]

        pos_today = positions_t.copy()

        flat_t = positions_t.sum(axis=0) == 0

        open_long = flat_t & valid & (s_scores < -1.25)
        open_short = flat_t & valid & (s_scores > 1.25)

        pos_today[0, open_long] = 1
        pos_today[1, open_short] = 1

        close_long = ((positions_t[0, :] == 1) & score_ok & (s_scores > -0.5))
        close_short = ((positions_t[1, :] == 1) & score_ok & (s_scores < 0.75))

        pos_today[0, close_long] = 0
        pos_today[1, close_short] = 0

        operations_t = pos_today - positions_t

        operations = np.zeros_like(positions)
        operations[:, not_nan_idx] = operations_t

        positions = positions + operations

        N_operations = (operations == 1).sum(axis=1)
        N_long_new = N_operations[0]
        N_short_new = N_operations[1]

        N_long_total = positions[0, :].sum()
        N_short_total = positions[1, :].sum()

        if N_long_new != 0 and N_long_total != 0:
            Q_long = equity_t * LEVERAGE / (N_long_new + N_long_total)

        if N_short_new != 0 and N_short_total != 0:
            Q_short = - equity_t * LEVERAGE / (N_short_new + N_short_total)

        # REGISTRA APERTURE
        # long
        mask_new_long = (operations_t[0, :] == 1)
        mask_new_short = (operations_t[1, :] == 1)

        for loc in np.where(mask_new_long)[0]:
            idx = not_nan_idx[loc]
            ticker = asset_list[idx]
            t_id = f"{ticker}_LONG"

            direct_dollar = np.zeros(len(not_nan_idx))
            direct_dollar[loc] = Q_long

            pca_dollar = -(betas[:, loc] * Q_long)
            hedge_dollar = eigenportfolio_weights @ pca_dollar

            total_dollar = direct_dollar + hedge_dollar
            today_t_cost += np.sum(np.abs(total_dollar)) * transaction_cost

            shares_trade = np.zeros(n_stocks)
            shares_trade[not_nan_idx] = total_dollar / price_today[not_nan_idx]

            shares_exposure += shares_trade
            cash -= shares_trade @ price_today

            transactions[t_id] = {"side": "LONG", "ticker": ticker, "open_t": t, "shares": shares_trade}

        # short
        for loc in np.where(mask_new_short)[0]:
            idx = not_nan_idx[loc]
            ticker = asset_list[idx]
            t_id = f"{ticker}_SHORT"

            direct_dollar = np.zeros(len(not_nan_idx))
            direct_dollar[loc] = Q_short

            pca_dollar = -(betas[:, loc] * Q_short)
            hedge_dollar = eigenportfolio_weights @ pca_dollar

            total_dollar = direct_dollar + hedge_dollar
            today_t_cost += np.sum(np.abs(total_dollar)) * transaction_cost

            shares_trade = np.zeros(n_stocks)
            shares_trade[not_nan_idx] = total_dollar / price_today[not_nan_idx]

            shares_exposure += shares_trade
            cash -= shares_trade @ price_today

            transactions[t_id] = {"side": "SHORT", "ticker": ticker, "open_t": t, "shares": shares_trade}

        # CHIUSURE
        # long
        mask_close_long = operations[0, :] == -1

        for idx in np.where(mask_close_long)[0]:
            ticker = asset_list[idx]
            t_id = f"{ticker}_LONG"

            shares_trade = -transactions[t_id]["shares"]

            shares_exposure += shares_trade
            cash -= shares_trade @ price_today

            today_t_cost += np.abs(shares_trade @ price_today) * transaction_cost

            del transactions[t_id]
        
        # short
        for idx in np.where(operations[1, :] == -1)[0]:
            ticker = asset_list[idx]
            t_id = f"{ticker}_SHORT"

            shares_trade = -transactions[t_id]["shares"]

            shares_exposure += shares_trade
            cash -= shares_trade @ price_today

            today_t_cost += np.abs(shares_trade @ price_today) * transaction_cost

            del transactions[t_id]

        # asset_exposure = shares_exposure * price_today
        cash -= today_t_cost

if __name__ == "__main__":
    for dataframe in df_names:
        started = datetime.datetime.now()
        print(f"[{started.strftime('%H:%M:%S.%f')[:-5]}] Processing {dataframe}...\n")
        now = datetime.datetime.now()

        print(f"[{now.strftime('%H:%M:%S.%f')[:-5]}] No transaction costs is running")

        prices = pd.read_excel(dataframe, index_col=0)
        variable_no_trans = run_statarb(prices)
        
        delta = datetime.datetime.now() - now
        now = datetime.datetime.now()
        print(f"[{now.strftime('%H:%M:%S.%f')[:-5]}] No transaction costs finished in {delta.total_seconds():.2f} seconds")
        now = datetime.datetime.now()
        print(f"[{now.strftime('%H:%M:%S.%f')[:-5]}] Transaction costs is running")

        variable_trans = run_statarb(prices, transaction_cost = T_COST)

        delta = datetime.datetime.now() - now
        now = datetime.datetime.now()
        print(f"[{now.strftime('%H:%M:%S.%f')[:-5]}] Transaction costs finished in {delta.total_seconds():.2f} seconds")
        now = datetime.datetime.now()
        print(f"[{now.strftime('%H:%M:%S.%f')[:-5]}] Retrieving benchmark and plotting results...")

        benchmark = pd.read_excel(dataframe, index_col = 0, sheet_name = "Benchmark")
        benchmark = benchmark.iloc[PCA_WINDOW:]

        normalized_benchmark = benchmark.iloc[:, 0] / benchmark.iloc[0, 0] * STARTING_EQUITY
        strategy_index = benchmark.index

        plt.style.use("dark_background")
        fig = plt.figure()
        fig_grid = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(fig_grid[0, :])
        ax2 = fig.add_subplot(fig_grid[1, 0])
        ax3 = fig.add_subplot(fig_grid[1, 1])

        ax1.plot(strategy_index, variable_no_trans, label = "No transaction costs", color = "white", alpha = 0.8, linewidth = 1)
        ax1.plot(strategy_index, variable_trans, label = "Transaction costs", color = "orange", alpha = 0.8, linewidth = 1)
        ax1.plot(strategy_index, normalized_benchmark.values.flatten(), color = "green", label = "Benchmark")

        ax2.plot(strategy_index, np.ones_like(variable_no_trans) * STARTING_EQUITY, color = "red", linestyle = "--", label = "Starting Equity")
        ax2.plot(strategy_index, variable_no_trans, label = "No transaction costs", color = "white", alpha = 0.8, linewidth = 1)
        ax2.plot(strategy_index, normalized_benchmark.values.flatten(), color = "green", label = "Benchmark")

        ax3.plot(strategy_index, np.ones_like(variable_trans) * STARTING_EQUITY, color = "red", linestyle = "--", label = "Starting Equity")
        ax3.plot(strategy_index, variable_trans, label = "Transaction costs", color = "orange", alpha = 0.8, linewidth = 1)
        ax3.plot(strategy_index, normalized_benchmark.values.flatten(), color = "green", label = "Benchmark")

        graph_titles[0] = df_titles[df_names.index(dataframe)]
        for idx, ax in enumerate([ax1, ax2, ax3]):
            ax.legend()
            ax.set_title(graph_titles[idx])

            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

            ax.grid(True, axis = "y", alpha = 0.4)

        plt.tight_layout()
        
        graphs[dataframe] = fig

        now = datetime.datetime.now()
        print(f"[{now.strftime('%H:%M:%S.%f')[:-5]}] plot registered, finished processing {dataframe} - Total time: {(now - started).total_seconds():.2f} seconds")
        print("-" * 200)
    
    now = datetime.datetime.now()
    print(f"[{now.strftime('%H:%M:%S.%f')[:-5]}] Building the report...")
    with PdfPages("StatArb_report.pdf") as pdf:
        for dataframe, fig in graphs.items():
            pdf.savefig(fig, bbox_inches = "tight")
            plt.close(fig)
    
    now = datetime.datetime.now()
    print(f"[{now.strftime('%H:%M:%S.%f')[:-5]}] Report completed!")
    print("=" * 200)