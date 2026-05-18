import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
import config
import data_manager
from rough_spde import RoughSPDE, create_spde_dataset, interpolate_field_to_etfs

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Rough SPDE) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 2:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            ret_win = returns.iloc[-win:]
            u0_list, target_list, x_original, etf_names = create_spde_dataset(ret_win, win, grid_size=config.GRID_SIZE)
            if len(u0_list) < 10:
                continue
            spde = RoughSPDE(grid_size=config.GRID_SIZE, hurst=config.HURST,
                             dt=config.DT, time_steps=config.TIME_STEPS,
                             hidden_channels=config.HIDDEN_CHANNELS,
                             kernel_size=config.KERNEL_SIZE, n_layers=config.N_LAYERS,
                             lr=config.LEARNING_RATE)
            spde.train(u0_list, target_list, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
            # Predict for the most recent initial condition (last day of the window)
            last_u0 = u0_list[-1].to(spde.device)
            pred_field = spde.predict(last_u0)
            etf_predictions = interpolate_field_to_etfs(pred_field, x_original, np.linspace(0,1,config.GRID_SIZE), etf_names)
            window_results[win] = etf_predictions
            for etf, score in etf_predictions.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "score": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by SPDE prediction: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/rough_spde_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Rough SPDE Engine complete ===")

if __name__ == "__main__":
    main()
