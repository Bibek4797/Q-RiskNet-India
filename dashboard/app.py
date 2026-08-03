import sys
import os
import pandas as pd
import streamlit as st

# Add root directory to python path for modular imports
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import src.data.pipeline as pipe
import src.econometrics.stats as stats
import src.econometrics.garch as garch
import src.models.qvar as qvar
import src.models.quantile_lstm as qlstm
import src.forecasting.girf as girf
import src.network.mst as mst
import src.network.spectral as spectral
import src.diagnostics.logger as diag

from dashboard.utils.theme import setup_page_config, inject_custom_css, render_header
from dashboard.components.sidebar import render_sidebar
from dashboard.components.kpi_cards import render_kpi_cards
from dashboard.components.charts import (
    render_prices_chart,
    render_drawdowns_chart,
    render_rolling_volatility_chart,
    render_correlation_chart,
    render_spillover_charts,
    render_network_graph,
    render_mst_graph,
    render_rolling_tci_chart
)
from dashboard.components.tables import render_descriptive_table, render_spillover_matrix_table

# Initialize page config & theme
setup_page_config()
inject_custom_css()

# Initialize session state
if 'pipeline_output' not in st.session_state:
    st.session_state['pipeline_output'] = None
if 'spillover_df' not in st.session_state:
    st.session_state['spillover_df'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = None

# Render Sidebar Controls
cfg = render_sidebar()

# Render Application Banner Header
render_header()

# Create 4 Core Financial Tabs
tab_data, tab_spillover, tab_network, tab_rolling = st.tabs([
    "📊 Data Center & Pipeline", 
    "📈 Volatility Spillover", 
    "🕸️ Network Topology", 
    "🕒 Dynamic TCI"
])

selected_sectors = cfg["selected_sectors"]
if len(selected_sectors) < 2:
    st.error("⚠️ Please select at least two sectoral indices in the sidebar to perform network analysis.")
    st.stop()

# Run Data Engineering Pipeline
@st.cache_data(show_spinner="Executing Financial Data Engineering Pipeline...")
def execute_pipeline(sectors, start, end):
    return pipe.run_data_pipeline(sectors, start, end, save_artifacts=True)

pipeline_res = None
try:
    pipeline_res = execute_pipeline(tuple(selected_sectors), cfg["start_date"], cfg["end_date"])
    st.session_state['pipeline_output'] = pipeline_res
    prices_df = pipeline_res["prices"]
    returns_df = pipeline_res["returns"]
    features_dict = pipeline_res["features"]
    val_report = pipeline_res["validation"]
    
    if cfg["volatility_proxy"] == "GARCH(1,1) Volatility":
        garch_cols = {}
        for col in returns_df.columns:
            garch_cols[col] = garch.estimate_garch_volatility(returns_df[col])
        model_input = pd.DataFrame(garch_cols, index=returns_df.index).dropna()
    else:
        model_input = returns_df.copy()

except Exception as e:
    diag.log_error("Data engineering pipeline execution failure", e)
    st.error(f"❌ Error in Data Pipeline: {str(e)}")
    st.info("💡 Try selecting different sectors, widening the date range, or picking fewer indices.")
    st.stop()

# TAB 1: DATA CENTER & PIPELINE
with tab_data:
    st.subheader("📊 Enterprise Financial Data Engineering & Quality Center")
    
    # Dataset Metadata KPI Bar
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.metric("Total Observations", val_report["total_rows"])
    with col_d2:
        st.metric("Selected Sectors", len(selected_sectors))
    with col_d3:
        st.metric("Date Coverage", f"{prices_df.index[0].strftime('%Y-%m-%d')} to {prices_df.index[-1].strftime('%Y-%m-%d')}")
    with col_d4:
        val_status = "✅ CLEAN" if val_report["is_valid"] else "⚠️ WARNINGS"
        st.metric("Data Quality Status", val_status)
        
    if val_report["warnings"]:
        with st.expander("🔍 Inspect Data Validation Warnings"):
            for w in val_report["warnings"]:
                st.warning(w)

    st.markdown("---")
    
    view_option = st.radio(
        "Select Feature View", 
        ["Base-100 Prices", "Log Returns", "Historical Drawdowns", "20d Rolling Volatility", "Pearson Correlation", "Descriptive Statistics"],
        horizontal=True
    )
    
    if view_option == "Base-100 Prices":
        render_prices_chart(prices_df)
    elif view_option == "Log Returns":
        fig_ret = px.line(returns_df, x=returns_df.index, y=returns_df.columns, title="Sectoral Percentage Log Returns (%)")
        fig_ret.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_ret, use_container_width=True)
    elif view_option == "Historical Drawdowns":
        render_drawdowns_chart(features_dict["drawdowns"])
    elif view_option == "20d Rolling Volatility":
        render_rolling_volatility_chart(features_dict["volatility_20d"], window_label="20-Day")
    elif view_option == "Pearson Correlation":
        corr_df = returns_df.corr()
        render_correlation_chart(corr_df)
    elif view_option == "Descriptive Statistics":
        desc_stats = stats.get_descriptive_stats(returns_df)
        render_descriptive_table(desc_stats)

# TAB 2: VOLATILITY SPILLOVER
with tab_spillover:
    st.subheader(f"📈 Risk Spillover Analysis ({cfg['model_choice']} at Quantile τ={cfg['quantile']:.2f})")
    
    run_model_btn = st.button("🚀 Calculate Risk Spillovers & Connectedness", type="primary")
    
    if run_model_btn or st.session_state['spillover_df'] is not None:
        if run_model_btn:
            try:
                progress_bar = st.progress(0, text="Fitting Model...")
                def update_progress(curr, total):
                    pct = int((curr / total) * 100)
                    progress_bar.progress(pct, text=f"Training epoch {curr}/{total}...")
                    
                if cfg["model_choice"] == "Quantile VAR (QVAR)":
                    model = qvar.QVARModel(p=cfg["lags"], quantile=cfg["quantile"])
                    model.fit(model_input)
                else:
                    model = qlstm.LSTMQuantileModel(
                        seq_len=cfg["seq_len"], 
                        hidden_dim=cfg["hidden_dim"], 
                        quantile=cfg["quantile"], 
                        epochs=cfg["epochs"], 
                        early_stopping=True, 
                        patience=5
                    )
                    model.fit(model_input, progress_callback=update_progress)
                    
                progress_bar.progress(100, text="Model Training Complete! Simulating GIRF Spillovers...")
                
                spill_df = girf.compute_spillover_matrix(model, model_input, horizon=cfg["forecast_horizon"])
                metrics = girf.calculate_connectedness_metrics(spill_df)
                
                st.session_state['spillover_df'] = spill_df
                st.session_state['metrics'] = metrics
                progress_bar.empty()
            except Exception as e:
                diag.log_error("Model fitting failure", e)
                st.error(f"❌ Error fitting {cfg['model_choice']}: {str(e)}")
                st.stop()
                
        spill_df = st.session_state['spillover_df']
        metrics = st.session_state['metrics']
        
        render_kpi_cards(metrics)
        st.markdown("---")
        render_spillover_charts(metrics)
        st.subheader("Diebold-Yilmaz Spillover Matrix (%)")
        render_spillover_matrix_table(spill_df, metrics)

# TAB 3: NETWORK TOPOLOGY & CLUSTERS
with tab_network:
    if st.session_state['spillover_df'] is None:
        st.info("Please calculate connectedness metrics in the 'Volatility Spillover' tab first.")
    else:
        spill_df = st.session_state['spillover_df']
        
        net_col1, net_col2 = st.columns([3, 1])
        with net_col2:
            st.markdown("### 🕸️ Network Controls")
            min_edge = st.slider("Minimum Edge Threshold (%)", min_value=0.0, max_value=15.0, value=2.0, step=0.5)
            layout_style = st.selectbox("Network Layout", ["circular", "spring"])
            comm_mode = st.radio("Community Detection Mode", ["Auto (Eigengap Heuristic)", "Manual Choice"], horizontal=True)
            
            if comm_mode == "Manual Choice":
                max_communities = max(2, len(selected_sectors) - 1)
                if max_communities > 2:
                    default_comm_val = min(3, max_communities)
                    n_comm = st.slider("Target Communities", min_value=2, max_value=max_communities, value=default_comm_val)
                else:
                    n_comm = 2
                    st.info(f"Target Communities set to 2 for {len(selected_sectors)} selected sectors.")
            else:
                n_comm = "auto"
            
        with net_col1:
            try:
                comms = spectral.detect_communities(spill_df, n_communities=n_comm)
                render_network_graph(spill_df, comms, min_edge, layout_style)
            except Exception as e:
                diag.log_error("Network rendering failure", e)
                st.error(f"Error drawing network graph: {str(e)}")
                
        st.markdown("---")
        st.subheader("🌲 Minimum Spanning Tree (MST) Risk Backbone")
        try:
            dist_matrix = mst.compute_correlation_distance(returns_df)
            mst_graph = mst.construct_mst(dist_matrix)
            render_mst_graph(mst_graph, dist_matrix)
        except Exception as e:
            diag.log_error("MST generation failure", e)
            st.error(f"Error drawing MST: {str(e)}")

# TAB 4: DYNAMIC TCI
with tab_rolling:
    st.subheader("🕒 Dynamic Time-Varying Total Connectedness Index (TCI)")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        window_size = st.slider("Rolling Window Size (Days)", min_value=60, max_value=500, value=200, step=20)
    with col_r2:
        step_size = st.slider("Window Step Size (Days)", min_value=5, max_value=60, value=20, step=5)
        
    run_rolling_btn = st.button("🔄 Compute Dynamic Rolling TCI", type="secondary")
    
    if run_rolling_btn:
        if len(model_input) < window_size:
            st.error(f"Data length ({len(model_input)}) is smaller than window size ({window_size}). Reduce window size.")
        else:
            dates = []
            tci_values = []
            
            total_windows = (len(model_input) - window_size) // step_size + 1
            progress_bar = st.progress(0, text="Calculating Rolling Window TCI...")
            
            for idx, i in enumerate(range(0, len(model_input) - window_size + 1, step_size)):
                sub_df = model_input.iloc[i : i + window_size]
                current_date = sub_df.index[-1]
                
                try:
                    if cfg["model_choice"] == "Quantile VAR (QVAR)":
                        roll_model = qvar.QVARModel(p=cfg["lags"], quantile=cfg["quantile"])
                        roll_model.fit(sub_df)
                    else:
                        roll_model = qlstm.LSTMQuantileModel(
                            seq_len=cfg["seq_len"], 
                            hidden_dim=cfg["hidden_dim"], 
                            quantile=cfg["quantile"], 
                            epochs=min(cfg["epochs"], 20),
                            early_stopping=True,
                            patience=3
                        )
                        roll_model.fit(sub_df)
                        
                    roll_spill = girf.compute_spillover_matrix(roll_model, sub_df, horizon=cfg["forecast_horizon"])
                    roll_metrics = girf.calculate_connectedness_metrics(roll_spill)
                    
                    dates.append(current_date)
                    tci_values.append(roll_metrics['TCI'])
                except Exception as e:
                    diag.log_warning(f"Rolling window at date {current_date} failed: {e}")
                    
                pct = int(((idx + 1) / total_windows) * 100)
                progress_bar.progress(pct, text=f"Processing window {idx+1}/{total_windows}...")
                
            progress_bar.empty()
            
            if dates:
                rolling_tci_df = pd.DataFrame({"Date": dates, "Rolling TCI (%)": tci_values}).set_index("Date")
                render_rolling_tci_chart(rolling_tci_df, window_size, step_size)

# Optional developer log inspector in sidebar expander
with st.sidebar.expander("🛠️ Developer Diagnostics Console", expanded=False):
    st.caption("Backend execution events & performance logs:")
    logs = st.session_state.get('diagnostics_logs', [])
    if not logs:
        st.caption("No log entries recorded.")
    else:
        for entry in reversed(logs[-10:]):
            st.text(f"[{entry['timestamp']}] [{entry['level']}] {entry['message']}")
            if entry.get("traceback"):
                st.code(entry["traceback"], language="python")
