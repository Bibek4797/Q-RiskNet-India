import os
import json
import pandas as pd
import numpy as np

from src.config.settings import PATHS, ROOT_DIR
import src.diagnostics.logger as diag
from src.models.qvar import QVARModel
from src.forecasting.girf import compute_spillover_matrix, calculate_connectedness_metrics

def run_static_connectedness(returns_df, p=2, quantile=0.50, horizon=10):
    """
    Computes static Diebold-Yilmaz connectedness matrix, directional TO/FROM/NET spillovers, and TCI.
    """
    with diag.DiagnosticTimer(f"Static Connectedness Computation (q={quantile}, H={horizon})"):
        model = QVARModel(p=p, quantile=quantile)
        model.fit(returns_df)
        spill_df = compute_spillover_matrix(model, returns_df, horizon=horizon)
        metrics = calculate_connectedness_metrics(spill_df)
        return {
            "spillover_matrix": spill_df,
            "metrics": metrics,
            "model": model
        }

def run_rolling_connectedness(returns_df, window_size=200, step_size=20, p=2, quantile=0.50, horizon=10):
    """
    Computes dynamic time-varying rolling TCI and rolling directional spillovers.
    """
    with diag.DiagnosticTimer(f"Dynamic Rolling Connectedness (Window={window_size}d, Step={step_size}d)"):
        dates = []
        tci_list = []
        to_history = []
        from_history = []
        net_history = []

        total_windows = (len(returns_df) - window_size) // step_size + 1
        for idx, i in enumerate(range(0, len(returns_df) - window_size + 1, step_size)):
            sub_df = returns_df.iloc[i : i + window_size]
            current_date = sub_df.index[-1]

            try:
                roll_model = QVARModel(p=p, quantile=quantile)
                roll_model.fit(sub_df)
                roll_spill = compute_spillover_matrix(roll_model, sub_df, horizon=horizon)
                roll_metrics = calculate_connectedness_metrics(roll_spill)

                dates.append(current_date)
                tci_list.append(roll_metrics["TCI"])
                
                to_row = roll_metrics["TO"].to_dict()
                to_row["Date"] = current_date
                to_history.append(to_row)

                from_row = roll_metrics["FROM"].to_dict()
                from_row["Date"] = current_date
                from_history.append(from_row)

                net_row = roll_metrics["NET"].to_dict()
                net_row["Date"] = current_date
                net_history.append(net_row)

            except Exception as e:
                diag.log_warning(f"Rolling window estimation failed at date {current_date}: {e}")

        tci_df = pd.DataFrame({"Date": dates, "Rolling_TCI_Pct": tci_list}).set_index("Date")
        to_df = pd.DataFrame(to_history).set_index("Date") if to_history else pd.DataFrame()
        from_df = pd.DataFrame(from_history).set_index("Date") if from_history else pd.DataFrame()
        net_df = pd.DataFrame(net_history).set_index("Date") if net_history else pd.DataFrame()

        return {
            "tci_df": tci_df,
            "to_df": to_df,
            "from_df": from_df,
            "net_df": net_df
        }

def run_all_connectedness_reports(returns_df, p=2, quantile=0.50, horizon=10, save_reports=True):
    """
    Master Connectedness Runner generating static & dynamic spillover metrics and exporting reports.
    """
    with diag.DiagnosticTimer("Master Connectedness Analysis Suite"):
        static_res = run_static_connectedness(returns_df, p=p, quantile=quantile, horizon=horizon)
        spill_df = static_res["spillover_matrix"]
        metrics = static_res["metrics"]

        directional_df = pd.DataFrame({
            "TO_OTHERS": metrics["TO"],
            "FROM_OTHERS": metrics["FROM"],
            "NET_SPILLOVER": metrics["NET"]
        })

        rolling_res = run_rolling_connectedness(returns_df, window_size=min(200, len(returns_df)//2), step_size=20, p=p, quantile=quantile, horizon=horizon)
        tci_df = rolling_res["tci_df"]

        reports_dir = os.path.join(ROOT_DIR, PATHS.get("reports_dir", "reports"))
        if save_reports:
            os.makedirs(reports_dir, exist_ok=True)
            spill_df.to_csv(os.path.join(reports_dir, "connectedness_matrix.csv"))
            directional_df.to_csv(os.path.join(reports_dir, "directional_spillovers.csv"))
            tci_df.to_csv(os.path.join(reports_dir, "rolling_tci_history.csv"))

            net_transmitters = directional_df[directional_df["NET_SPILLOVER"] > 0].index.tolist()
            net_receivers = directional_df[directional_df["NET_SPILLOVER"] < 0].index.tolist()

            summary_json = {
                "total_sectors": len(returns_df.columns),
                "total_connectedness_index_pct": round(float(metrics["TCI"]), 2),
                "max_net_transmitter": str(directional_df["NET_SPILLOVER"].idxmax()),
                "max_net_receiver": str(directional_df["NET_SPILLOVER"].idxmin()),
                "net_transmitters_count": len(net_transmitters),
                "net_receivers_count": len(net_receivers),
                "net_transmitters": net_transmitters,
                "net_receivers": net_receivers
            }
            with open(os.path.join(reports_dir, "systemic_risk_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary_json, f, indent=4)

            diag.log_info(f"Saved connectedness reports to {reports_dir}")

        return {
            "static": static_res,
            "directional": directional_df,
            "rolling": rolling_res
        }
