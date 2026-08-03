import pandas as pd
import numpy as np
import src.diagnostics.logger as diag

def validate_dataset(df):
    """
    Performs comprehensive financial dataset validation.
    Returns a dictionary report with status and issues found.
    """
    with diag.DiagnosticTimer("Dataset Quality & Validation Audit"):
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "duplicate_timestamps": 0,
            "duplicate_rows": 0,
            "negative_prices": 0,
            "invalid_zeros": 0,
            "missing_values_by_column": {},
            "total_missing_values": 0,
            "is_valid": True,
            "warnings": []
        }

        if df.empty:
            report["is_valid"] = False
            report["warnings"].append("Dataset is completely empty.")
            return report

        # 1. Duplicated Timestamps
        if df.index.duplicated().any():
            dup_count = int(df.index.duplicated().sum())
            report["duplicate_timestamps"] = dup_count
            report["is_valid"] = False
            report["warnings"].append(f"Found {dup_count} duplicate timestamps.")

        # 2. Duplicate Rows
        dup_rows = int(df.duplicated().sum())
        report["duplicate_rows"] = dup_rows
        if dup_rows > 0:
            report["warnings"].append(f"Found {dup_rows} duplicate data rows.")

        # 3. Missing Values by Column
        missing_series = df.isna().sum()
        report["missing_values_by_column"] = missing_series.to_dict()
        total_missing = int(missing_series.sum())
        report["total_missing_values"] = total_missing
        if total_missing > 0:
            report["warnings"].append(f"Dataset contains {total_missing} missing (NaN) values.")

        # 4. Invalid / Negative Prices
        numeric_df = df.select_dtypes(include=[np.number])
        neg_count = int((numeric_df < 0).sum().sum())
        report["negative_prices"] = neg_count
        if neg_count > 0:
            report["is_valid"] = False
            report["warnings"].append(f"Found {neg_count} negative price values!")

        zero_count = int((numeric_df == 0).sum().sum())
        report["invalid_zeros"] = zero_count
        if zero_count > 0:
            report["warnings"].append(f"Found {zero_count} zero values in prices.")

        diag.log_info(f"Validation finished. Valid={report['is_valid']}, Warnings={len(report['warnings'])}")
        return report
