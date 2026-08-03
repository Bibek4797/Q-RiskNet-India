import src.diagnostics.logger as diag
from src.data.download import fetch_raw_market_data
from src.data.validation import validate_dataset
from src.data.preprocessing import compute_comprehensive_features
from src.data.export import export_processed_data, generate_quality_reports

def run_data_pipeline(sectors, start_date, end_date, save_artifacts=True):
    """
    Master Financial Data Engineering Pipeline Orchestrator.
    Downloads, validates, preprocesses, and exports high-quality financial datasets.
    """
    with diag.DiagnosticTimer("Master Financial Data Pipeline Execution"):
        # 1. Download Raw Market Data
        prices_df = fetch_raw_market_data(sectors, start_date, end_date, save_raw=save_artifacts)

        # 2. Validate Dataset Integrity
        val_report = validate_dataset(prices_df)
        if not val_report["is_valid"]:
            diag.log_warning(f"Data validation issues detected: {val_report['warnings']}")

        # 3. Preprocess & Feature Generation
        features = compute_comprehensive_features(prices_df)

        # 4. Export Artifacts & Reports
        if save_artifacts:
            export_processed_data(features)
            generate_quality_reports(prices_df, features["log_returns"], val_report)

        return {
            "prices": features["prices"],
            "returns": features["log_returns"],
            "daily_returns": features["daily_returns"],
            "features": features,
            "validation": val_report
        }
