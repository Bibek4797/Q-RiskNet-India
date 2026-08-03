import os
import json
import pandas as pd
from src.config.settings import PATHS, ROOT_DIR
import src.diagnostics.logger as diag
from src.models.qvar import QVARModel, estimate_multi_quantile_qvar, compute_qvar_girf

def run_all_qvar_diagnostics(returns_df, p=2, quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], save_reports=True):
    """
    Master QVAR Modelling & Diagnostic Runner.
    Fits QVAR across all specified quantiles, generates parameter comparison tables,
    computes GIRF curves, and saves reports in reports/.
    """
    with diag.DiagnosticTimer("Master QVAR Analysis Suite"):
        multi_q_res = estimate_multi_quantile_qvar(returns_df, p=p, quantiles=quantiles)
        summary_df = multi_q_res["summary_df"]
        models_dict = multi_q_res["models"]

        # GIRF simulation for first sector
        first_sector = returns_df.columns[0]
        median_model = models_dict.get(0.50, list(models_dict.values())[0])
        girf_df = compute_qvar_girf(median_model, returns_df, shocked_sector=first_sector, shock_size_std=2.0, horizon=10)

        reports_dir = os.path.join(ROOT_DIR, PATHS.get("reports_dir", "reports"))
        if save_reports:
            os.makedirs(reports_dir, exist_ok=True)
            summary_df.to_csv(os.path.join(reports_dir, "qvar_parameter_summary.csv"), index=False)
            
            pivoted_comp = summary_df.pivot_table(
                index=["Target_Sector", "Source_Sector"],
                columns="Quantile",
                values="Coefficient"
            ).reset_index()
            pivoted_comp.to_csv(os.path.join(reports_dir, "qvar_quantile_comparison.csv"), index=False)
            girf_df.to_csv(os.path.join(reports_dir, "qvar_girf_responses.csv"), index=True)

            summary_json = {
                "total_sectors": len(returns_df.columns),
                "lags_p": p,
                "quantiles": quantiles,
                "total_estimated_parameters": len(summary_df),
                "bear_vs_bull_coef_diff_mean": float((summary_df[summary_df["Quantile"] == 0.05]["Coefficient"].values - summary_df[summary_df["Quantile"] == 0.95]["Coefficient"].values).mean())
            }
            with open(os.path.join(reports_dir, "qvar_diagnostics_report.json"), "w", encoding="utf-8") as f:
                json.dump(summary_json, f, indent=4)

            diag.log_info(f"Saved QVAR reports to {reports_dir}")

        return {
            "summary_df": summary_df,
            "models_dict": models_dict,
            "girf_df": girf_df
        }
