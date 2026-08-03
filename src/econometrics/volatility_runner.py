import os
import json
import pandas as pd
from src.config.settings import PATHS, ROOT_DIR
import src.diagnostics.logger as diag
from src.econometrics.volatility import compare_volatility_models_for_sector, generate_multi_step_volatility_forecast

def run_all_volatility_models(returns_df, save_reports=True):
    """
    Master Volatility Modelling Runner.
    Fits ARCH(1), GARCH(1,1), EGARCH(1,1,1), and GJR-GARCH(1,1,1) across all sectors.
    Generates comparison tables, parameter estimates, and multi-step forecasts.
    """
    with diag.DiagnosticTimer("Master Volatility Modelling Suite"):
        all_comparison_rows = []
        forecast_rows = []

        for col in returns_df.columns:
            s = returns_df[col]
            sector_comp_df = compare_volatility_models_for_sector(s)
            
            for _, row in sector_comp_df.iterrows():
                clean_row = {k: v for k, v in row.items() if k != "fit_result"}
                all_comparison_rows.append(clean_row)

                res_obj = row["fit_result"]
                fc_dict = generate_multi_step_volatility_forecast(res_obj, horizons=[1, 5, 20])
                fc_row = {
                    "Sector": col,
                    "Model": row["Model"],
                    "AIC": row["AIC"],
                    **fc_dict
                }
                forecast_rows.append(fc_row)

        comp_df = pd.DataFrame(all_comparison_rows)
        fc_df = pd.DataFrame(forecast_rows)

        reports_dir = os.path.join(ROOT_DIR, PATHS.get("reports_dir", "reports"))
        if save_reports:
            os.makedirs(reports_dir, exist_ok=True)
            comp_df.to_csv(os.path.join(reports_dir, "volatility_model_comparison.csv"), index=False)
            comp_df.to_csv(os.path.join(reports_dir, "volatility_parameter_estimates.csv"), index=False)
            fc_df.to_csv(os.path.join(reports_dir, "volatility_forecasts.csv"), index=False)

            best_models = comp_df.loc[comp_df.groupby("Sector")["AIC"].idxmin()]
            summary_json = {
                "total_sectors": len(returns_df.columns),
                "preferred_model_distribution": best_models["Model"].value_counts().to_dict(),
                "average_persistence": float(comp_df["Persistence"].mean()),
                "average_half_life_days": float(comp_df["Half_Life_Days"].mean())
            }
            with open(os.path.join(reports_dir, "volatility_diagnostics_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary_json, f, indent=4)

            diag.log_info(f"Saved volatility reports to {reports_dir}")

        return {
            "comparison": comp_df,
            "forecasts": fc_df
        }
