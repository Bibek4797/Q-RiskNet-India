import os
import json
import pandas as pd
from src.config.settings import PATHS, ROOT_DIR
import src.diagnostics.logger as diag
from src.econometrics.stationarity import run_full_stationarity_suite
from src.econometrics.autocorr import run_full_autocorrelation_suite
from src.econometrics.hetero import run_full_hetero_suite
from src.econometrics.distribution import run_full_distribution_suite
from src.econometrics.nonlinearity import run_full_nonlinearity_suite
from src.econometrics.structural_breaks import run_full_structural_breaks_suite

def run_all_econometric_diagnostics(returns_df, save_reports=True):
    """
    Master Econometric Diagnostics Suite.
    Executes stationarity, autocorrelation, heteroskedasticity, distribution, non-linearity, and structural break tests.
    Generates structured reports in reports/.
    """
    with diag.DiagnosticTimer("Master Econometric Diagnostics Execution"):
        stationarity_df = run_full_stationarity_suite(returns_df)
        autocorr_df = run_full_autocorrelation_suite(returns_df)
        hetero_df = run_full_hetero_suite(returns_df)
        distribution_df = run_full_distribution_suite(returns_df)
        nonlinearity_df = run_full_nonlinearity_suite(returns_df)
        structural_breaks_df = run_full_structural_breaks_suite(returns_df)

        reports_dir = os.path.join(ROOT_DIR, PATHS.get("reports_dir", "reports"))
        if save_reports:
            os.makedirs(reports_dir, exist_ok=True)
            stationarity_df.to_csv(os.path.join(reports_dir, "stationarity_summary.csv"), index=False)
            autocorr_df.to_csv(os.path.join(reports_dir, "autocorrelation_summary.csv"), index=False)
            hetero_df.to_csv(os.path.join(reports_dir, "arch_lm_summary.csv"), index=False)
            distribution_df.to_csv(os.path.join(reports_dir, "distribution_summary.csv"), index=False)
            nonlinearity_df.to_csv(os.path.join(reports_dir, "nonlinearity_summary.csv"), index=False)
            structural_breaks_df.to_csv(os.path.join(reports_dir, "structural_breaks_summary.csv"), index=False)

            summary_json = {
                "total_sectors": len(returns_df.columns),
                "sectors": list(returns_df.columns),
                "stationary_sectors_count": int((stationarity_df[stationarity_df["Test"] == "ADF"]["Decision"] == "Stationary").sum()),
                "arch_effects_count": int(hetero_df["ARCH_Effects_Present"].sum()),
                "non_normal_sectors_count": int((~distribution_df["Is_Normal"]).sum()),
                "structural_breaks_count": int(structural_breaks_df["Structural_Break_Present"].sum())
            }
            with open(os.path.join(reports_dir, "econometric_diagnostic_report.json"), "w", encoding="utf-8") as f:
                json.dump(summary_json, f, indent=4)

            diag.log_info(f"Saved 7 econometric diagnostic reports to {reports_dir}")

        return {
            "stationarity": stationarity_df,
            "autocorrelation": autocorr_df,
            "heteroskedasticity": hetero_df,
            "distribution": distribution_df,
            "nonlinearity": nonlinearity_df,
            "structural_breaks": structural_breaks_df
        }
