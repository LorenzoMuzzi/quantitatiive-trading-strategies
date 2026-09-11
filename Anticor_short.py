import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

MIN_W = 3
MAX_W = 30

MAX_GROSS_LEVERAGE = 2.0 # per evitare che la leva diventi esageratamente grande e falsi la strategia (sotto un profilo di realisticità e applicazione)

df_names = ["#DJI30_20150101_20260519.xlsx", "#NDX100_20150101_20260519.xlsx", "#SP500_20150101_20260519.xlsx"]


def apply_gross_leverage_cap(weights: np.ndarray, max_gross_leverage: float | None) -> np.ndarray:
    if max_gross_leverage is None:
        return weights

    gross = np.sum(np.abs(weights))

    if gross <= max_gross_leverage:
        return weights

    net = np.sum(weights)

    if gross == 0.0:
        return weights

    positive_mask = weights > 0
    negative_mask = weights < 0

    long_exposure = weights[positive_mask].sum()
    short_exposure = -weights[negative_mask].sum()

    if short_exposure == 0.0:
        return weights

    effective_cap = max(max_gross_leverage, abs(net))

    target_short = max(0.0, (effective_cap - net) / 2.0)
    target_long = net + target_short

    capped = np.zeros_like(weights)

    if long_exposure > 0.0:
        capped[positive_mask] = weights[positive_mask] * (target_long / long_exposure)

    if short_exposure > 0.0:
        capped[negative_mask] = weights[negative_mask] * (target_short / short_exposure)

    return capped


def build_valid_rel_masks(not_nan: np.ndarray, min_w: int, max_w: int) -> dict[int, np.ndarray]:
    n, m = not_nan.shape
    valid_rel_masks = {}

    cumsum_valid = np.cumsum(not_nan.astype(np.int32), axis=0)

    for w in range(min_w, max_w + 1):
        valid = np.zeros_like(not_nan, dtype=bool)

        window_count = cumsum_valid.copy()
        window_count[w:] = cumsum_valid[w:] - cumsum_valid[:-w]

        valid[w - 1:] = window_count[w - 1:] == w
        valid_rel_masks[w] = valid

    return valid_rel_masks


def build_weights_anticor_single_w(w: int, n: int, m: int, rel: np.ndarray, log_rel: np.ndarray, valid_rel: np.ndarray, not_nan: np.ndarray, concentration_list: list,
    turnover_list: list, short: bool = False, max_gross_leverage: float | None = None) -> np.ndarray:

    weights = np.zeros((n, m), dtype=np.float64)

    initial_active = not_nan[1]
    starting_titles = initial_active.sum()

    if starting_titles > 0:
        weights[0, initial_active] = 1.0 / starting_titles
        weights[1, initial_active] = 1.0 / starting_titles

    for t in range(1, n):

        weights[t, ~not_nan[t]] = 0.0

        s_t = weights[t] @ rel[t]

        if s_t == 0.0:
            s_t = 1.0

        if t < n - 1:

            if t < 2 * w:
                weights[t + 1] = weights[t]

            else:
                b_hat_t = weights[t] * rel[t] / s_t

                valid_window = valid_rel[t - 2 * w + 1: t + 1, :].all(axis=0)

                if not np.any(valid_window):
                    weights[t + 1] = b_hat_t
                    weights[t + 1] = apply_gross_leverage_cap(weights[t + 1], max_gross_leverage)
                    continue

                lx_all = log_rel[t - 2 * w + 1: t + 1, :]
                lx_sub = lx_all[:, valid_window]
                cw_sub = b_hat_t[valid_window]

                lx_1 = lx_sub[:w, :]
                lx_2 = lx_sub[w:, :]

                mu_1 = lx_1.mean(axis=0)
                mu_2 = lx_2.mean(axis=0)

                x1 = lx_1 - mu_1
                x2 = lx_2 - mu_2

                cov = (x1.T @ x2) / (w - 1)

                var_1 = np.sum(x1 ** 2, axis=0) / (w - 1)
                var_2 = np.sum(x2 ** 2, axis=0) / (w - 1)

                sigma_1 = np.sqrt(var_1)
                sigma_2 = np.sqrt(var_2)

                s1 = np.where(sigma_1 > 0, sigma_1, 1.0)
                s2 = np.where(sigma_2 > 0, sigma_2, 1.0)

                corr = cov / s1[:, None]
                corr /= s2[None, :]

                corr[(sigma_1 == 0)[:, None] | (sigma_2 == 0)[None, :]] = 0.0

                a = np.abs(np.minimum(0.0, np.diag(corr)))
                claims_raw = corr + a[:, None] + a[None, :]

                claims = np.where((mu_2[:, None] > mu_2[None, :]) & (corr > 0), claims_raw, 0.0)

                row_sum = claims.sum(axis=1)

                if short:
                    total_claims = row_sum.sum()
                    budget = cw_sub.sum()

                    if total_claims > 0:
                        scale = budget / total_claims
                        incoming = claims.sum(axis=0) * scale
                        outgoing = row_sum * scale
                    else:
                        incoming = np.zeros_like(cw_sub)
                        outgoing = np.zeros_like(cw_sub)

                else:
                    coeff = np.divide(
                        cw_sub,
                        row_sum,
                        out=np.zeros_like(cw_sub),
                        where=row_sum != 0
                    )

                    incoming = coeff @ claims
                    outgoing = np.where(row_sum != 0, cw_sub, 0.0)

                new_sub = cw_sub + incoming - outgoing

                weights[t + 1] = b_hat_t
                weights[t + 1, valid_window] = new_sub
                weights[t + 1, ~not_nan[t + 1]] = 0.0

                weights[t + 1] = apply_gross_leverage_cap(weights[t + 1], max_gross_leverage)

        else:
            concentration_list.append(np.sum(weights ** 2, axis=1).mean())

            turnover_list.append((0.5 * np.sum(np.abs(weights[1:] - weights[:-1]), axis=1)).mean())

    return weights


def compute_wealth_from_weights(weights: np.ndarray, rel: np.ndarray, w: int, transaction_cost_rate: float) -> np.ndarray:
    n = weights.shape[0]

    wealth = np.empty(n, dtype=np.float64)
    wealth[0] = 1.0

    for t in range(1, n):

        s_t = weights[t] @ rel[t]

        if (s_t == 0.0) or (t <= 2 * w):
            wealth[t] = wealth[t - 1]
        else:
            tot_weight_chg = np.sum(np.abs(weights[t] - weights[t - 1]))
            transaction_cost = tot_weight_chg * wealth[t - 1] * transaction_cost_rate

            wealth[t] = wealth[t - 1] * s_t - transaction_cost

    return wealth


def run_anticor_from_df(prices_df: pd.DataFrame, min_w: int = MIN_W, max_w: int = MAX_W, transaction_cost_rate: float = 0.0,
    short: bool = False, max_gross_leverage: float | None = None, save_weights: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:

    prices = prices_df.astype(float).to_numpy()

    n, m = prices.shape
    w_values = np.arange(min_w, max_w + 1)

    metrics_df = pd.DataFrame(np.zeros((2, len(w_values))), index=["turnover medio", "concentrazione media"], columns=w_values)

    not_nan = ~np.isnan(prices) & (prices > 0)

    valid_rel_masks = build_valid_rel_masks(not_nan, min_w, max_w)

    prices_shifted = np.zeros_like(prices)
    prices_shifted[1:] = prices[:-1]

    wealth = np.empty((n, len(w_values)), dtype=np.float64)

    turnover_list = []
    concentration_list = []

    weights_list = []

    for k, w in enumerate(w_values):

        valid_rel = valid_rel_masks[w]

        rel = np.ones_like(prices, dtype=np.float64)
        np.divide(prices, prices_shifted, out=rel, where=valid_rel)

        log_rel = np.log(rel)

        weights = build_weights_anticor_single_w(w=w, n=n, m=m, rel=rel, log_rel=log_rel, valid_rel=valid_rel, not_nan=not_nan, concentration_list=concentration_list,
            turnover_list=turnover_list, short=short, max_gross_leverage=max_gross_leverage)

        wealth[:, k] = compute_wealth_from_weights(weights=weights, rel=rel, w=w, transaction_cost_rate=transaction_cost_rate)

        if save_weights:
            weights_list.append(weights)

    metrics_df.loc["turnover medio", :] = turnover_list
    metrics_df.loc["concentrazione media", :] = concentration_list

    wealth_df = pd.DataFrame(
        wealth,
        index=prices_df.index,
        columns=w_values
    )

    return wealth_df, metrics_df, weights_list


def run_anticor(path: str,     min_w: int = MIN_W, max_w: int = MAX_W, transaction_cost_rate: float = 0.0, short: bool = False, max_gross_leverage: float | None = None,
    save_weights: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:

    prices_df = pd.read_excel(path, index_col="data")

    return run_anticor_from_df(prices_df=prices_df, min_w=min_w, max_w=max_w, transaction_cost_rate=transaction_cost_rate, short=short,
        max_gross_leverage=max_gross_leverage, save_weights=save_weights)


def retrieve_benchmark(path: str) -> np.ndarray:
    benchmark = pd.read_excel(path, sheet_name="Benchmark", index_col="data").iloc[:, 0]

    benchmark_rel = (benchmark / benchmark.shift(1)).fillna(1.0).to_numpy(dtype=np.float64)

    return np.ones(MAX_W - MIN_W + 1) * np.prod(benchmark_rel)


def metrics(df_wealth: pd.DataFrame, others_metrics_df: pd.DataFrame, strategy_result_df: pd.DataFrame) -> pd.DataFrame:

    df_wealth_rend = df_wealth.pct_change().fillna(0.0)

    n_days = len(df_wealth)

    wealth_values = df_wealth.to_numpy(dtype=np.float64)
    returns_values = df_wealth_rend.to_numpy(dtype=np.float64)

    cagr = wealth_values[-1, :] ** (252 / n_days) - 1

    vol = np.std(returns_values, axis=0) * np.sqrt(252)

    sharpe_ratio = np.divide(cagr, vol, out=np.zeros_like(cagr), where=vol != 0)

    running_max = np.maximum.accumulate(wealth_values, axis=0)
    drawdown = wealth_values / running_max - 1
    max_drawdown = np.min(drawdown, axis=0)

    df = pd.DataFrame([cagr, vol, sharpe_ratio, max_drawdown], index=["cagr", "vol", "sharpe_ratio", "max_drawdown"], columns=np.arange(MIN_W, MAX_W + 1))

    return pd.concat([df, others_metrics_df, strategy_result_df])


if __name__ == "__main__":

    report_file = "Report_Anticor_short.xlsx"
    excel_start_row = 0

    for i, dataset in enumerate(df_names):

        if not os.path.exists(dataset):
            print(f"File '{dataset}' not found. Please place the Excel files in this directory to run.")
            continue

        print(f"\nProcessing '{dataset}'...")

        prices_df = pd.read_excel(dataset, index_col="data")

        save_weights = dataset == "#DJI30_20150101_20260519.xlsx"

        start_time = time.time()

        No_trans_cost, No_trans_cost_metrics, weights_No_trans = run_anticor_from_df(prices_df=prices_df, transaction_cost_rate=0.0, short=True,
            max_gross_leverage=MAX_GROSS_LEVERAGE, save_weights=save_weights)

        final_wealth_no_cost = pd.DataFrame([No_trans_cost.iloc[-1, :].to_numpy()], index=["final_wealth"], columns=np.arange(MIN_W, MAX_W + 1))

        df_metr = metrics(No_trans_cost, No_trans_cost_metrics, final_wealth_no_cost)

        t_no_cost = time.time() - start_time

        print(
            f" -> [{datetime.now().strftime('%H:%M:%S')}] "
            f"Without transaction costs: simulated 28 windows in {t_no_cost:.2f} seconds!"
        )

        start_time = time.time()

        T_cost, T_cost_metrics, weights_trans = run_anticor_from_df(prices_df=prices_df, transaction_cost_rate=0.0019, short=True, 
            max_gross_leverage=MAX_GROSS_LEVERAGE, save_weights=False)

        final_wealth_t_cost = pd.DataFrame([T_cost.iloc[-1, :].to_numpy()], index=["final_wealth"], columns=np.arange(MIN_W, MAX_W + 1))

        df_metr_T = metrics(T_cost, T_cost_metrics, final_wealth_t_cost)

        t_cost = time.time() - start_time

        print(
            f" -> [{datetime.now().strftime('%H:%M:%S')}] "
            f"With transaction costs: simulated 28 windows in {t_cost:.2f} seconds!"
        )

        benchmark = pd.DataFrame([retrieve_benchmark(dataset)], columns=np.arange(MIN_W, MAX_W + 1))

        sheet_sigle = dataset[dataset.find("#") + 1: dataset.find("_")]
        sheet_price = f"prices_{sheet_sigle}"
        sheet_wealth = f"wealth_{sheet_sigle}"

        mode = "a" if os.path.exists(report_file) else "w"

        writer_kwargs = {
            "path": report_file,
            "engine": "openpyxl",
            "mode": mode
        }

        if mode == "a":
            writer_kwargs["if_sheet_exists"] = "overlay"

        with pd.ExcelWriter(**writer_kwargs) as writer:

            benchmark.to_excel(writer, sheet_name="Elenchi", startrow=(i * 2 + 14), startcol=1, index=False)

            df_metr.to_excel(writer, sheet_name="metrics", startrow=excel_start_row, startcol=1, index=True)

            excel_start_row += 9

            df_metr_T.to_excel(writer, sheet_name="metrics", startrow=excel_start_row, startcol=1, index=True)

            excel_start_row += 9

            prices_df.to_excel(writer, sheet_name=sheet_price, index=True)

            No_trans_cost.to_excel(writer, sheet_name=sheet_wealth, startrow=1, startcol=0, index=True)

            T_cost.to_excel(writer, sheet_name=sheet_wealth, startrow=1, startcol=MAX_W + 2, index=True)

            if dataset == "#DJI30_20150101_20260519.xlsx":

                for idx, w in enumerate(range(MIN_W, MAX_W + 1)):

                    weights_df = pd.DataFrame(weights_No_trans[idx], index=No_trans_cost.index, columns=prices_df.columns)

                    new_sheet = f"weights_{w}"

                    weights_df.to_excel(writer, sheet_name=new_sheet, index=True)

        print(
            f" -> [{datetime.now().strftime('%H:%M:%S')}] "
            f"Exported results safely to Excel '{report_file}'"
        )

        """
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(No_trans_cost.columns, No_trans_cost.iloc[-1, :], marker="o", label="Without transaction costs")
            ax.plot(T_cost.columns, T_cost.iloc[-1, :], marker="s", label="With transaction costs (0.19%)")
            ax.plot(No_trans_cost.columns, benchmark.iloc[0, :], linestyle="--", color="gray", label="Benchmark holding")
            ax.set_title(f"Anticor Strategy Performance - Dataset {dataset.split('_')[0]}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Window Size (w)", fontsize=10)
            ax.set_ylabel("Total Wealth Return", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(frameon=True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f" -> Plotting skipped: {e}")
        """

    print("\n" + "=" * 60)
    print("ALL DATASETS PROCESSED SUCCESSFULLY WITH NO ERROR!")
    print("=" * 60)