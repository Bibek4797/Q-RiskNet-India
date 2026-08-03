import os
import json
import pandas as pd
from src.config.settings import PATHS, ROOT_DIR
import src.diagnostics.logger as diag

def export_processed_data(feature_dict, suffix=""):
    """
    Exports clean datasets (log returns, volatility, drawdowns) to data/processed/.
    """
    proc_dir = os.path.join(ROOT_DIR, PATHS.get("processed_data", "data/processed"))
    os.makedirs(proc_dir, exist_ok=True)

    exported_files = []
    for key, df in feature_dict.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            filename = f"{key}{suffix}.csv"
            filepath = os.path.join(proc_dir, filename)
            df.to_csv(filepath)
            exported_files.append(filepath)

    diag.log_info(f"Exported {len(exported_files)} processed feature files to {proc_dir}")
    return exported_files

def generate_quality_reports(prices_df, returns_df, validation_report):
    """
    Generates automated data quality reports in reports/:
    - missing_data_summary.csv
    - distribution_summary.csv
    - data_completeness_report.json
    """
    reports_dir = os.path.join(ROOT_DIR, PATHS.get("reports_dir", "reports"))
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Missing data summary
    missing_df = pd.DataFrame({
        "Sector": list(prices_df.columns),
        "Missing_Prices_Count": [prices_df[col].isna().sum() for col in prices_df.columns],
        "Missing_Prices_Pct": [(prices_df[col].isna().sum() / len(prices_df)) * 100 for col in prices_df.columns]
    })
    missing_filepath = os.path.join(reports_dir, "missing_data_summary.csv")
    missing_df.to_csv(missing_filepath, index=False)

    # 2. Distribution summary
    dist_df = pd.DataFrame({
        "Mean": returns_df.mean(),
        "Std": returns_df.std(),
        "Min": returns_df.min(),
        "Max": returns_df.max(),
        "Skewness": returns_df.skew(),
        "Kurtosis": returns_df.kurtosis() + 3
    })
    dist_filepath = os.path.join(reports_dir, "distribution_summary.csv")
    dist_df.to_csv(dist_filepath)

    # 3. Completeness JSON report
    completeness = {
        "start_date": str(prices_df.index[0].date()) if not prices_df.empty else "N/A",
        "end_date": str(prices_df.index[-1].date()) if not prices_df.empty else "N/A",
        "total_observations": len(prices_df),
        "sectors_count": len(prices_df.columns),
        "sectors": list(prices_df.columns),
        "validation": validation_report
    }
    comp_filepath = os.path.join(reports_dir, "data_completeness_report.json")
    with open(comp_filepath, "w", encoding="utf-8") as f:
        json.dump(completeness, f, indent=4)

    diag.log_info(f"Quality reports generated in {reports_dir}")
    return [missing_filepath, dist_filepath, comp_filepath]
