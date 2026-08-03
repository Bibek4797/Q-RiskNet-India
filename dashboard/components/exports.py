"""
Q-RiskNet India — Export & Download Utilities
Copyright (c) 2026 Bibek Rout
"""
import io
import json
import streamlit as st
import pandas as pd


def download_csv(df, filename, label="📥 Download CSV", key=None):
    """Renders a CSV download button for a DataFrame."""
    csv_data = df.to_csv(index=True)
    st.download_button(label=label, data=csv_data, file_name=filename,
                       mime="text/csv", key=key)


def download_json(data, filename, label="📥 Download JSON", key=None):
    """Renders a JSON download button for a dictionary."""
    json_str = json.dumps(data, indent=4, default=str)
    st.download_button(label=label, data=json_str, file_name=filename,
                       mime="application/json", key=key)


def download_chart_html(fig, filename, label="📥 Download Chart (HTML)", key=None):
    """Renders a download button for a Plotly figure as standalone HTML."""
    html_str = fig.to_html(include_plotlyjs="cdn")
    st.download_button(label=label, data=html_str, file_name=filename,
                       mime="text/html", key=key)
