"""
Q-RiskNet India — Reports & Export Center Page
Copyright (c) 2026 Bibek Rout
"""
import os
import json
import streamlit as st
import pandas as pd

from src.config.settings import ROOT_DIR, PATHS
from dashboard.components.exports import download_csv, download_json


REPORTS_DIR = os.path.join(ROOT_DIR, PATHS.get("reports_dir", "reports"))


def _load_report_file(filename):
    """Safely loads a report file from the reports/ directory."""
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        return None
    if filename.endswith(".csv"):
        return pd.read_csv(path)
    elif filename.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


REPORT_CATALOG = [
    ("forecast_benchmark_summary.csv", "📊 Forecast Benchmark Summary", "csv"),
    ("forecast_accuracy_comparison.csv", "📈 Forecast Accuracy Comparison", "csv"),
    ("forecast_benchmark_report.json", "📋 Forecast Benchmark Report", "json"),
    ("robustness_window_sensitivity.csv", "🔬 Window Sensitivity", "csv"),
    ("robustness_horizon_sensitivity.csv", "🔬 Horizon Sensitivity", "csv"),
    ("robustness_threshold_sensitivity.csv", "🔬 Threshold Sensitivity", "csv"),
    ("research_validation_report.json", "📋 Validation Summary", "json"),
    ("network_centrality.csv", "🕸️ Network Centrality Rankings", "csv"),
    ("network_global_stats.json", "🕸️ Network Global Statistics", "json"),
    ("volatility_comparison.csv", "📈 Volatility Model Comparison", "csv"),
    ("diagnostics_report.json", "🔬 Diagnostics Report", "json"),
]


def render_page():
    """Renders the Reports & Export Center page."""

    st.header("📋 Reports & Export Center")
    st.caption("Central repository of all generated research reports. "
               "Download individual outputs or browse analysis summaries.")

    if not os.path.isdir(REPORTS_DIR):
        st.warning("⚠️ The `reports/` directory does not exist yet. "
                    "Run analyses from the other pages to generate reports.")
        return

    # ── List Available Reports ────────────────────────────────────────
    available_files = [f for f in os.listdir(REPORTS_DIR)
                       if f.endswith((".csv", ".json"))] if os.path.isdir(REPORTS_DIR) else []

    st.subheader(f"📁 {len(available_files)} Reports Available")

    if not available_files:
        st.info("ℹ️ No report files found. Run analyses to generate outputs.")
        return

    for filename, label, ftype in REPORT_CATALOG:
        if filename in available_files:
            with st.expander(f"{label}  —  `{filename}`"):
                data = _load_report_file(filename)
                if data is None:
                    st.warning(f"Could not load {filename}")
                elif ftype == "csv" and isinstance(data, pd.DataFrame):
                    st.dataframe(data.head(50), use_container_width=True)
                    download_csv(data, filename, key=f"dl_{filename}")
                elif ftype == "json" and isinstance(data, dict):
                    st.json(data)
                    download_json(data, filename, key=f"dl_{filename}")

    # ── Uncatalogued Files ────────────────────────────────────────────
    catalogued = {r[0] for r in REPORT_CATALOG}
    extra = [f for f in available_files if f not in catalogued]
    if extra:
        st.markdown("---")
        st.subheader("📄 Additional Report Files")
        for f in extra:
            data = _load_report_file(f)
            if data is not None:
                with st.expander(f"`{f}`"):
                    if isinstance(data, pd.DataFrame):
                        st.dataframe(data.head(50), use_container_width=True)
                        download_csv(data, f, key=f"dl_extra_{f}")
                    elif isinstance(data, dict):
                        st.json(data)
                        download_json(data, f, key=f"dl_extra_{f}")
