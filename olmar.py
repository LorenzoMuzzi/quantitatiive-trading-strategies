import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

epsilon = 1
MAX_W = 30
MIN_W = 3
w_values = np.arange(MIN_W, MAX_W + 1)

df_names = ["DJI30_sample.xlsx"]

# FUNZIONI PER CALCOLI
def predict_price_relatives(w: int, t: int, prices:np.ndarray, active: np.ndarray) -> np.ndarray:
    x_pred = np.ones(prices.shape[1], dtype=float)
    
    current_price = prices[t,:]
    window_prices = prices[t + 1 - w : t + 1, :]

    valid_window = window_prices > 0

    ratios = np.zeros_like(window_prices, dtype=float)
    np.divide(window_prices, current_price, out=ratios, where=valid_window & active)
    
    count_valid = valid_window.sum(axis=0) # che equivale a w per gli elementi validi

    np.divide(ratios.sum(axis=0), count_valid, out=x_pred, where = active & (count_valid > 0))
    x_pred[~active] = 1.0
    return x_pred

def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    n = len(v)

    v_sorted = np.sort(v)[::-1] # utile per capire quali pesi rimarranno positivi dopo averci levato la quota teta
    eccedenza = np.cumsum(v_sorted) - 1

    hp_attivi = np.arange(1, n + 1)
    test = v_sorted - (eccedenza / hp_attivi) > 0
    
    n_attivi = sum(test)
    const = eccedenza[test][-1] / n_attivi

    return np.maximum(v - const, 0)

'''
L'idea è di portare a 0 i pesi negativi e ridurre gli altri di una certa quota costante. Bisogna però stare attenti perchè questa stessa riduzione potrebbe portare certi pesi a divenire negativi
E' proprio per evitare ciò che prima si ordinano gli elementi in ordine decrescente e poi si testa se i vari elementi "resisterebbero" a una riduzione pari alla suddetta quota, la quale dipende proprio dal
numero di elementi attivi (cioè che rimarranno maggiori di 0) e passivi (che verranno invece portati a 0). In questo modo possiamo definire quanti elementi dovranno rimanere attivi e poi basterà
banalmente dividere l'eccedenza per il numero degli attivi e togliere questa quota a ogni peso.
'''

def update_portfolio(rel_pred: np.ndarray, current_portfolio: np.ndarray, epsilon: int, active: np.ndarray) -> np.ndarray:
    new_weights = np.zeros_like(current_portfolio)

    rel_active = rel_pred[active]
    weights_active = current_portfolio[active]
        
    rel_mean = rel_active.mean()
    diff = rel_active - rel_mean
    denom = np.dot(diff, diff)

    lagr = max(0.0, (epsilon - np.dot(weights_active, rel_active)) / denom)
    weights_gross = weights_active + lagr * diff
    new_weights[active] = project_to_simplex(weights_gross)

    return new_weights

def transaction_costs(s_past: float, new_weights: np.ndarray, old_weights: np.ndarray, cost_rate: float = 0) -> float:
    if cost_rate == 0:
        return 0.0
    tot_weight_chg = np.sum(np.abs(new_weights - old_weights))
    return tot_weight_chg * s_past * cost_rate

def run_single_olmar(n: int, m: int, window: int, not_nan: np.ndarray, valid_rel: np.ndarray, relative_df: np.ndarray, prices: np.ndarray, transaction_cost: float = 0.0):
    weights = np.zeros((n + 1, m), dtype=float) # n + 1 per far sì che non si rompa il ciclo quando siamo in t = n
    wealth_col = np.empty(n, dtype=float)
    wealth_col[0] = 1.0
    active_start = not_nan[0, :]
    n_active = active_start.sum()
    weights[0, active_start] = 1.0 / n_active
    weights[1, active_start] = 1.0 / n_active

    for t in range(1, n):
        active_window = valid_rel[t, :]

        price_relative = relative_df[t,:]
        growth_fact = (np.dot(weights[t, :], price_relative)) if t >= window else 1
        wealth_col[t] = wealth_col[t - 1] * growth_fact

        if t + 1 < window:
            active = not_nan[t, :]
            n_active = active.sum()

            weights[t + 1, active] = 1.0 / n_active
            continue
          
        weights[t, ~active_window] = 0.0

        x_pred = predict_price_relatives(window, t, prices, active_window)
        weights[t + 1, :] = update_portfolio(x_pred, weights[t, :], epsilon, active_window) # weights[t, :] = portafoglio deciso dopo aver osservato p_t, quindi usabile dal periodo t+1
        t_cost = transaction_costs(wealth_col[t], weights[t + 1, :], weights[t, :], transaction_cost)

        wealth_col[t] -= t_cost
        
    # concentrazione media
    concentrazione = np.sum(weights**2, axis=1).mean()
            
    # turnover medio
    turnover = (0.5 * np.sum(np.abs(weights[1:] - weights[:-1]), axis=1)).mean()
    
    return wealth_col, concentrazione, turnover, weights

def run_olmar(path: str, transaction_cost_rate: float = 0) -> pd.DataFrame:
    df = pd.read_excel(path, index_col='data')
    prices = df.astype(float).to_numpy()

    n, m = prices.shape
    w_values = np.arange(MIN_W, MAX_W + 1)

    not_nan = ~np.isnan(prices) & (prices > 0)
    
    valid_rel_masks = {}

    for window in range(MIN_W, MAX_W + 1):
        valid_rel_loop = np.zeros_like(not_nan, dtype=bool)

        matrici = [
            not_nan[window - 1 - lag : len(not_nan) - lag] for lag in range(window)
        ]

        valid_rel_loop[window - 1:] = np.logical_and.reduce(matrici)

        valid_rel_masks[window] = valid_rel_loop

    valid_daily_rel = np.zeros_like(not_nan, dtype=bool)
    valid_daily_rel[1:] = not_nan[1:] & not_nan[:-1]

    prices_shifted = np.zeros_like(prices)
    prices_shifted[1:] = prices[:-1]

    rel = np.ones_like(prices)
    np.divide(prices, prices_shifted, out=rel, where=valid_daily_rel)

    wealth = np.empty((n, len(w_values)), dtype=float)

    list_c = []
    list_t = []
    weights_list = []

    for k, window in enumerate(w_values):
        valid_rel = valid_rel_masks[window]

        wealth[:, k], concentrazione, turnover, weights = run_single_olmar(n, m, window, not_nan, valid_rel, rel, prices, transaction_cost_rate)
        list_c.append(concentrazione)
        list_t.append(turnover)

        if path == "#DJI30_20150101_20260519.xlsx":
            weights_list.append(weights)

    return pd.DataFrame(wealth, index=df.index, columns=w_values), list_c, list_t, weights_list

def retrieve_benchmark(path: str) -> np.ndarray:
    benchmark = pd.read_excel(path, sheet_name="Benchmark", index_col="data").iloc[:,0]
    benchmark_rel = (benchmark / benchmark.shift(1)).fillna(1.0).to_numpy(dtype=float)
    return np.ones(len(w_values)) * np.prod(benchmark_rel)

def metrics(df_wealth: pd.DataFrame, others_metrics_df: pd.DataFrame, strategy_result_df: pd.DataFrame) -> pd.DataFrame:
    df_wealth_rend = df_wealth.pct_change().fillna(0.0)
        
    # rend. e volatilità annui
    n_days = len(df_wealth)
    cagr = df_wealth.to_numpy()[-1, :] ** (252 / n_days) - 1

    vol = np.std(df_wealth_rend.to_numpy(), axis=0) * np.sqrt(252)
    sharpe_ratio = np.divide(cagr, vol)

    # max drawdon
    running_max = np.maximum.accumulate(df_wealth, axis=0)
    drawdown = df_wealth / running_max - 1
    max_drawdown = np.min(drawdown, axis=0).to_numpy()

    df = pd.DataFrame([cagr, vol, sharpe_ratio, max_drawdown], index=["cagr", "vol", "sharpe_ratio", "max_drawdown"], columns=np.arange(MIN_W, MAX_W + 1))
    return pd.concat([df, others_metrics_df, strategy_result_df])

if __name__ == "__main__":
    report_file = "Report_Olmar_epsilon1.xlsx"
    excel_start_row = 0

    for i, dataset in enumerate(df_names):
        if not os.path.exists(dataset):
            print(f"File '{dataset}' not found. Please place the Excel files in this directory to run.")
            continue
            
        print(f"\nProcessing '{dataset}'...")

        # NO COSTI DI TRANSAZIONE
        start_time = time.time()

        No_trans_cost, concentration, turnover, weights_No_trans = run_olmar(path=dataset, transaction_cost_rate=0.0)
        final_wealth_no_cost = pd.DataFrame([No_trans_cost.iloc[-1,:].to_numpy()], index=["final_wealth"], columns=np.arange(MIN_W, MAX_W+1))
        No_trans_cost_metrics = pd.DataFrame([concentration, turnover], index = ["concentration", "turnover"], columns=np.arange(MIN_W, MAX_W+1))
        df_metr = metrics(No_trans_cost, No_trans_cost_metrics, final_wealth_no_cost)
        t_no_cost = time.time() - start_time

        print(f" -> [{datetime.now().strftime("%H:%M:%S")}] Without transaction costs: Simulated {MAX_W - MIN_W + 1} windows in {t_no_cost:.2f} seconds!")

        # COSTI DI TRANSAZIONE
        start_time = time.time()

        T_cost, T_concentration, T_turnover, weights_trans = run_olmar(path=dataset, transaction_cost_rate=0.0019)
        T_cost_metrics = pd.DataFrame([T_concentration, T_turnover], index = ["concentration", "turnover"], columns=np.arange(MIN_W, MAX_W+1))
        final_wealth_t_cost = pd.DataFrame([T_cost.iloc[-1,:].to_numpy()], index=["final_wealth"], columns=np.arange(MIN_W, MAX_W+1))
        df_metr_T = metrics(T_cost, T_cost_metrics, final_wealth_t_cost)
        t_cost = time.time() - start_time

        print(f" -> [{datetime.now().strftime("%H:%M:%S")}] With transaction costs: Simulated {MAX_W - MIN_W + 1} windows in {t_cost:.2f} seconds!")

        # ALTRI DATI
        benchmark = pd.DataFrame([retrieve_benchmark(dataset)], columns=np.arange(MIN_W, MAX_W+1))
        prices_df = pd.read_excel(dataset, index_col="data")

        sheet_sigle = dataset[dataset.find("#") + 1 : dataset.find("_")]
        sheet_price = f"prices_{sheet_sigle}"
        sheet_wealth = f"wealth_{sheet_sigle}"

        # EXPORT SU EXCEL
        mode = "a" if os.path.exists(report_file) else "w"

        with pd.ExcelWriter(
            report_file,
            engine="openpyxl",
            mode=mode,
            if_sheet_exists="overlay" if mode == "a" else None
        ) as writer:
            
            benchmark.to_excel(
                writer,
                sheet_name= "Elenchi",
                startrow= (i * 2 + 14),
                startcol= 1,
                index= False
            )

            df_metr.to_excel(
                writer,
                sheet_name= "metrics",
                startrow= excel_start_row,
                startcol= 1,
                index= True
            )
            
            excel_start_row += 9

            df_metr_T.to_excel(
                writer,
                sheet_name= "metrics",
                startrow= excel_start_row,
                startcol= 1,
                index= True
            )

            excel_start_row += 9
            
            prices_df.to_excel(
                writer,
                sheet_name= sheet_price,
                index= True
            )

            No_trans_cost.to_excel(
                writer,
                sheet_name = sheet_wealth,
                startrow= 1,
                startcol= 0,
                index= True
            )

            T_cost.to_excel(
                writer,
                sheet_name= sheet_wealth,
                startrow= 1,
                startcol= MAX_W,
                index= True
            )

            if dataset == "#DJI30_20150101_20260519.xlsx":
                col_weights = 0

                for idx, w in enumerate(range(MIN_W, MAX_W + 1)):
                    weights_df = pd.DataFrame(weights_No_trans[idx][:-1], index=No_trans_cost.index, columns=pd.read_excel(dataset, index_col="data").columns)

                    new_sheet = f"weights_{w}"
                    weights_df.to_excel(
                        writer,
                        sheet_name= new_sheet,
                        index= True
                    )

        print(f" -> [{datetime.now().strftime("%H:%M:%S")}] Exported results safely to Excel '{report_file}'")

        '''fig, ax = plt.subplots()
        ax.plot(No_trans_cost.columns, No_trans_cost.iloc[-1, :], label = "No commission")
        ax.plot(T_cost.columns, T_cost.iloc[-1, :], label = "With commission")
        ax.set_xlabel('finestra w')
        ax.set_ylabel('tot wealth')
        ax.legend()
        plt.show()'''

    print("\n" + "=" * 60)
    print("ALL DATASETS PROCESSED SUCCESSFULLY WITH NO ERROR.")
    print("=" * 60)
