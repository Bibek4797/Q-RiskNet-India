"""
Q-RiskNet India — Portfolio Backtesting Engine
Copyright (c) 2026 Bibek Rout
"""
import numpy as np
import pandas as pd

import src.diagnostics.logger as diag
from src.econometrics.tail_risk import compute_historical_var, compute_historical_cvar
from src.portfolio.optimization import (
    optimize_minimum_variance,
    optimize_risk_parity,
    optimize_cvar_portfolio,
    evaluate_portfolio_performance
)


def run_portfolio_backtest(returns_df, max_weight=0.40, min_history_days=126):
    """
    Executes a strict chronological out-of-sample monthly rebalancing portfolio backtest.
    At each month end T, weights are calculated using ONLY data up to T, and held during T+1.
    Computes out-of-sample cumulative equity curves and comparative risk metrics.
    Supports both DatetimeIndex and RangeIndex.
    """
    with diag.DiagnosticTimer(f"Chronological Monthly Rebalancing Backtest (max_w={max_weight})"):
        K = returns_df.shape[1]
        equal_w = pd.Series(np.ones(K) / K, index=returns_df.columns)

        is_datetime = isinstance(returns_df.index, pd.DatetimeIndex)

        if is_datetime:
            periods = returns_df.index.to_period('M').unique()
            start_month_idx = None
            for idx, p in enumerate(periods):
                end_date = returns_df[returns_df.index.to_period('M') <= p].index[-1]
                if len(returns_df.loc[:end_date]) >= min_history_days:
                    start_month_idx = idx
                    break
        else:
            # Fallback for non-datetime index (e.g. RangeIndex in unit tests)
            step = 21  # 21 trading days per month
            periods = list(range(step, len(returns_df), step))
            start_month_idx = 0 if len(returns_df) >= min_history_days else None

        # Fallback to full-sample if dataset is too short for rolling backtest
        if start_month_idx is None or (is_datetime and start_month_idx >= len(periods) - 1):
            diag.log_warning("Dataset too short for monthly out-of-sample rebalancing. Using full-sample fallback.")
            min_var_w = optimize_minimum_variance(returns_df, max_weight=max_weight)
            risk_parity_w = optimize_risk_parity(returns_df, max_weight=max_weight)
            cvar_w = optimize_cvar_portfolio(returns_df, max_weight=max_weight)
            weights_dict = {
                "Equal Weight": equal_w,
                "Minimum Variance (MPT)": min_var_w,
                "Risk Parity (ERC)": risk_parity_w,
                "CVaR Tail-Risk": cvar_w
            }
            perf_rows = []
            equity_curves = {}
            for name, w in weights_dict.items():
                eval_res = evaluate_portfolio_performance(w, returns_df)
                equity_curves[name] = np.cumprod(1.0 + (eval_res["portfolio_returns"] / 100.0)) * 100.0
                row = {
                    "Portfolio": name,
                    "Annualized_Return_Pct": eval_res["Annualized_Return_Pct"],
                    "Annualized_Vol_Pct": eval_res["Annualized_Vol_Pct"],
                    "Sharpe_Ratio": eval_res["Sharpe_Ratio"],
                    "Max_Drawdown_Pct": eval_res["Max_Drawdown_Pct"],
                    "VaR_95_Pct": eval_res["VaR_95_Pct"],
                    "CVaR_95_Pct": eval_res["CVaR_95_Pct"]
                }
                perf_rows.append(row)
            return {
                "summary_df": pd.DataFrame(perf_rows).sort_values(by="Sharpe_Ratio", ascending=False),
                "equity_df": pd.DataFrame(equity_curves),
                "weights_dict": weights_dict
            }

        # Chronological out-of-sample monthly rebalancing loop
        oos_returns = {
            "Equal Weight": [],
            "Minimum Variance (MPT)": [],
            "Risk Parity (ERC)": [],
            "CVaR Tail-Risk": []
        }
        oos_dates = []
        latest_weights = {}

        if is_datetime:
            for i in range(start_month_idx, len(periods) - 1):
                curr_p = periods[i]
                next_p = periods[i + 1]

                hist_end = returns_df[returns_df.index.to_period('M') <= curr_p].index[-1]
                hist_df = returns_df.loc[:hist_end]

                next_ret_df = returns_df[returns_df.index.to_period('M') == next_p]
                if next_ret_df.empty:
                    continue

                w_mv = optimize_minimum_variance(hist_df, max_weight=max_weight)
                w_rp = optimize_risk_parity(hist_df, max_weight=max_weight)
                w_cv = optimize_cvar_portfolio(hist_df, max_weight=max_weight)

                latest_weights = {
                    "Equal Weight": equal_w,
                    "Minimum Variance (MPT)": w_mv,
                    "Risk Parity (ERC)": w_rp,
                    "CVaR Tail-Risk": w_cv
                }

                oos_dates.extend(next_ret_df.index)
                oos_returns["Equal Weight"].extend(np.dot(next_ret_df.values, equal_w.values))
                oos_returns["Minimum Variance (MPT)"].extend(np.dot(next_ret_df.values, w_mv.values))
                oos_returns["Risk Parity (ERC)"].extend(np.dot(next_ret_df.values, w_rp.values))
                oos_returns["CVaR Tail-Risk"].extend(np.dot(next_ret_df.values, w_cv.values))
        else:
            step = 21
            for t_split in range(min_history_days, len(returns_df), step):
                t_end = min(t_split + step, len(returns_df))
                hist_df = returns_df.iloc[:t_split]
                next_ret_df = returns_df.iloc[t_split:t_end]
                if next_ret_df.empty:
                    continue

                w_mv = optimize_minimum_variance(hist_df, max_weight=max_weight)
                w_rp = optimize_risk_parity(hist_df, max_weight=max_weight)
                w_cv = optimize_cvar_portfolio(hist_df, max_weight=max_weight)

                latest_weights = {
                    "Equal Weight": equal_w,
                    "Minimum Variance (MPT)": w_mv,
                    "Risk Parity (ERC)": w_rp,
                    "CVaR Tail-Risk": w_cv
                }

                oos_dates.extend(next_ret_df.index)
                oos_returns["Equal Weight"].extend(np.dot(next_ret_df.values, equal_w.values))
                oos_returns["Minimum Variance (MPT)"].extend(np.dot(next_ret_df.values, w_mv.values))
                oos_returns["Risk Parity (ERC)"].extend(np.dot(next_ret_df.values, w_rp.values))
                oos_returns["CVaR Tail-Risk"].extend(np.dot(next_ret_df.values, w_cv.values))

        # Evaluate performance on concatenated out-of-sample returns
        perf_rows = []
        equity_curves = {}

        for name, ret_list in oos_returns.items():
            s_ret = pd.Series(ret_list, index=oos_dates)
            ann_return = float(np.mean(s_ret) * 252.0)
            ann_vol = float(np.std(s_ret, ddof=1) * np.sqrt(252.0))
            sharpe = float((ann_return - 0.06) / ann_vol) if ann_vol > 0 else 0.0

            cum_ret = np.cumprod(1.0 + (s_ret / 100.0)) * 100.0
            running_max = np.maximum.accumulate(cum_ret)
            drawdowns = (cum_ret - running_max) / running_max * 100.0
            max_dd = float(np.min(drawdowns))

            var_95 = compute_historical_var(s_ret, confidence=0.95)
            cvar_95 = compute_historical_cvar(s_ret, confidence=0.95)

            equity_curves[name] = cum_ret

            row = {
                "Portfolio": name,
                "Annualized_Return_Pct": round(ann_return, 2),
                "Annualized_Vol_Pct": round(ann_vol, 2),
                "Sharpe_Ratio": round(sharpe, 2),
                "Max_Drawdown_Pct": round(max_dd, 2),
                "VaR_95_Pct": var_95,
                "CVaR_95_Pct": cvar_95
            }
            perf_rows.append(row)

        summary_df = pd.DataFrame(perf_rows).sort_values(by="Sharpe_Ratio", ascending=False)
        equity_df = pd.DataFrame(equity_curves)

        diag.log_info(f"Out-of-sample monthly rebalancing completed across {len(oos_dates)} trading days.")

        return {
            "summary_df": summary_df,
            "equity_df": equity_df,
            "weights_dict": latest_weights
        }
