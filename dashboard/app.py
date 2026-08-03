import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add root directory to python path for modular imports
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config.settings import TICKER_MAP
import src.data.data_loader as dl
import src.econometrics.stats as stats
import src.econometrics.garch as garch
import src.models.qvar as qvar
import src.models.quantile_lstm as qlstm
import src.forecasting.girf as girf
import src.network.mst as mst
import src.network.spectral as spectral
import src.visualization.plotly_plots as vis
import src.diagnostics.logger as diag

# Page configuration
st.set_page_config(
    page_title="Q-RiskNet India | Volatility Spillover Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3b82f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        padding-left: 16px;
        padding-right: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'data_downloaded' not in st.session_state:
    st.session_state['data_downloaded'] = False
if 'spillover_df' not in st.session_state:
    st.session_state['spillover_df'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = None

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/line-chart.png", width=60)
st.sidebar.title("📌 Q-RiskNet Controls")

st.sidebar.subheader("📅 Data Configuration")
selected_sectors = st.sidebar.multiselect(
    "Select Sectoral Indices",
    options=list(TICKER_MAP.keys()),
    default=list(TICKER_MAP.keys())[:7]
)

today = datetime.today()
five_years_ago = today - timedelta(days=5 * 365)
start_date = st.sidebar.date_input("Start Date", value=five_years_ago)
end_date = st.sidebar.date_input("End Date", value=today)

st.sidebar.subheader("⚙️ Model Settings")
model_choice = st.sidebar.radio("Model Type", ["Quantile VAR (QVAR)", "Quantile LSTM"])
quantile = st.sidebar.slider("Quantile Value (τ)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)

optim_mode = st.sidebar.radio("Hyperparameter Engine", ["⚡ Auto-Optimized (Recommended)", "⚙️ Custom Parameters"])

lags = 2
seq_len = 5
epochs = 50
hidden_dim = "auto"

if optim_mode == "⚙️ Custom Parameters":
    if model_choice == "Quantile VAR (QVAR)":
        lags = st.sidebar.slider("Autoregressive Lags (p)", min_value=1, max_value=5, value=2)
    else:
        seq_len = st.sidebar.slider("Sequence Length (Lags)", min_value=3, max_value=15, value=5)
        epochs = st.sidebar.slider("LSTM Epochs", min_value=10, max_value=100, value=30, step=5)
        hidden_dim = st.sidebar.slider("Hidden Layer Nodes", min_value=8, max_value=64, value=16, step=8)
else:
    st.sidebar.caption("🤖 Auto-tunes hidden layers, early-stopping convergence, and network topology automatically.")

volatility_proxy = st.sidebar.selectbox("Risk / Volatility Metric", ["Log Returns", "GARCH(1,1) Volatility"])
forecast_horizon = st.sidebar.slider("Forecast Horizon (H)", min_value=5, max_value=30, value=10)

# Main Application Layout
st.markdown("<div class='main-header'>Q-RiskNet-India: Sectoral Spillover Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Quantile-LSTM Deep Learning & Financial Network Topology Analysis for Indian Stock Markets</div>", unsafe_allow_html=True)

tab_data, tab_spillover, tab_network, tab_rolling = st.tabs([
    "📊 Data Center", 
    "📈 Volatility Spillover", 
    "🕸️ Network Topology", 
    "🕒 Dynamic TCI"
])

if len(selected_sectors) < 2:
    st.error("⚠️ Please select at least two sectoral indices in the sidebar to perform network analysis.")
    st.stop()

@st.cache_data(show_spinner="Downloading data from Yahoo Finance...")
def load_data(sectors, start, end):
    return dl.download_data(sectors, start, end)

prices_df = pd.DataFrame()
returns_df = pd.DataFrame()
model_input = pd.DataFrame()

try:
    prices_df = load_data(tuple(selected_sectors), start_date, end_date)
    returns_df = dl.calculate_log_returns(prices_df)
    
    if volatility_proxy == "GARCH(1,1) Volatility":
        garch_cols = {}
        for col in returns_df.columns:
            garch_cols[col] = garch.estimate_garch_volatility(returns_df[col])
        model_input = pd.DataFrame(garch_cols, index=returns_df.index).dropna()
    else:
        model_input = returns_df.copy()
        
    st.session_state['data_downloaded'] = True

except Exception as e:
    diag.log_error("Data loading failure", e)
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("💡 Try selecting different sectors, widening the date range, or picking fewer indices.")
    st.stop()

# TAB 1: DATA CENTER
with tab_data:
    st.subheader("📊 Sectoral Index Prices & Normalized Trends")
    
    norm_prices = (prices_df / prices_df.iloc[0]) * 100
    fig_prices = px.line(
        norm_prices, 
        x=norm_prices.index, 
        y=norm_prices.columns,
        title="Base-100 Normalized Price Trends",
        labels={"value": "Normalized Level (Base=100)", "variable": "Sector"}
    )
    fig_prices.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_prices, use_container_width=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🔥 Pearson Correlation Matrix")
        corr_df = returns_df.corr()
        fig_corr = vis.render_correlation_heatmap(corr_df)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col2:
        st.subheader("📋 Econometric Descriptive Statistics")
        desc_stats = stats.get_descriptive_stats(returns_df)
        st.dataframe(desc_stats, use_container_width=True, height=450)
        st.caption("JB: Jarque-Bera Normality Test (p < 0.05 indicates fat tails). ADF: Augmented Dickey-Fuller Unit Root Test (p < 0.05 indicates stationarity).")

# TAB 2: VOLATILITY SPILLOVER
with tab_spillover:
    st.subheader(f"📈 Risk Spillover Analysis ({model_choice} at Quantile τ={quantile:.2f})")
    
    run_model_btn = st.button("🚀 Calculate Risk Spillovers & Connectedness", type="primary")
    
    if run_model_btn or st.session_state['spillover_df'] is not None:
        if run_model_btn:
            try:
                progress_bar = st.progress(0, text="Fitting Model...")
                def update_progress(curr, total):
                    pct = int((curr / total) * 100)
                    progress_bar.progress(pct, text=f"Training epoch {curr}/{total}...")
                    
                if model_choice == "Quantile VAR (QVAR)":
                    model = qvar.QVARModel(p=lags, quantile=quantile)
                    model.fit(model_input)
                else:
                    model = qlstm.LSTMQuantileModel(
                        seq_len=seq_len, 
                        hidden_dim=hidden_dim, 
                        quantile=quantile, 
                        epochs=epochs, 
                        early_stopping=True, 
                        patience=5
                    )
                    model.fit(model_input, progress_callback=update_progress)
                    
                progress_bar.progress(100, text="Model Training Complete! Simulating GIRF Spillovers...")
                
                spill_df = girf.compute_spillover_matrix(model, model_input, horizon=forecast_horizon)
                metrics = girf.calculate_connectedness_metrics(spill_df)
                
                st.session_state['spillover_df'] = spill_df
                st.session_state['metrics'] = metrics
                progress_bar.empty()
            except Exception as e:
                diag.log_error("Model fitting failure", e)
                st.error(f"❌ Error fitting {model_choice}: {str(e)}")
                st.stop()
                
        spill_df = st.session_state['spillover_df']
        metrics = st.session_state['metrics']
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Connectedness Index (TCI)", f"{metrics['TCI']:.2f}%", help="Systemic risk index of cross-sector connectedness.")
        with col_m2:
            max_transmitter = metrics['NET'].idxmax()
            st.metric("Top Systemic Transmitter", max_transmitter, f"+{metrics['NET'][max_transmitter]:.2f}% Net Outflow")
        with col_m3:
            max_receiver = metrics['NET'].idxmin()
            st.metric("Top Risk Receiver", max_receiver, f"{metrics['NET'][max_receiver]:.2f}% Net Inflow")
            
        st.markdown("---")
        col_chart1, col_chart2 = st.columns([1, 1])
        with col_chart1:
            st.subheader("Net Directional Spillover (TO - FROM)")
            net_series = metrics['NET'].sort_values()
            fig_net = px.bar(
                x=net_series.values, 
                y=net_series.index, 
                orientation='h',
                color=net_series.values,
                color_continuous_scale="RdYlGn_r",
                labels={"x": "Net Spillover (%)", "y": "Sector"}
            )
            fig_net.update_layout(template="plotly_dark", height=400, showlegend=False)
            st.plotly_chart(fig_net, use_container_width=True)
            
        with col_chart2:
            st.subheader("Directional TO & FROM Spillovers")
            to_from_df = pd.DataFrame({"TO OTHERS": metrics['TO'], "FROM OTHERS": metrics['FROM']})
            fig_tf = px.bar(
                to_from_df, 
                barmode='group',
                title="Gross Directional Spillovers"
            )
            fig_tf.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_tf, use_container_width=True)
            
        st.subheader("Diebold-Yilmaz Spillover Matrix (%)")
        display_spill = spill_df.copy()
        display_spill["TO OTHERS"] = metrics["TO"]
        from_row = pd.Series(metrics["FROM"], name="FROM OTHERS")
        display_spill = pd.concat([display_spill, pd.DataFrame([from_row])])
        display_spill.loc["FROM OTHERS", "TO OTHERS"] = metrics["TCI"]
        
        try:
            st.dataframe(display_spill.style.format("{:.2f}%").background_gradient(cmap="Reds", subset=(spill_df.index, spill_df.columns)), use_container_width=True)
        except Exception:
            st.dataframe(display_spill.style.format("{:.2f}%"), use_container_width=True)
        st.caption("Rows represent receiving sectors (FROM); columns represent transmitting sectors (TO). Diagonal represents self-spillover.")

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
                fig_net_graph = vis.render_spillover_network(
                    spill_df, 
                    communities=comms, 
                    min_threshold_pct=min_edge, 
                    layout_type=layout_style
                )
                st.plotly_chart(fig_net_graph, use_container_width=True)
            except Exception as e:
                diag.log_error("Network rendering failure", e)
                st.error(f"Error drawing network graph: {str(e)}")
                
        st.markdown("---")
        st.subheader("🌲 Minimum Spanning Tree (MST) Risk Backbone")
        try:
            dist_matrix = mst.compute_correlation_distance(returns_df)
            mst_graph = mst.construct_mst(dist_matrix)
            fig_mst = vis.render_mst_network(mst_graph, dist_matrix)
            st.plotly_chart(fig_mst, use_container_width=True)
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
                    if model_choice == "Quantile VAR (QVAR)":
                        roll_model = qvar.QVARModel(p=lags, quantile=quantile)
                        roll_model.fit(sub_df)
                    else:
                        roll_model = qlstm.LSTMQuantileModel(
                            seq_len=seq_len, 
                            hidden_dim=hidden_dim, 
                            quantile=quantile, 
                            epochs=min(epochs, 20),
                            early_stopping=True,
                            patience=3
                        )
                        roll_model.fit(sub_df)
                        
                    roll_spill = girf.compute_spillover_matrix(roll_model, sub_df, horizon=forecast_horizon)
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
                fig_rolling = px.line(
                    rolling_tci_df, 
                    y="Rolling TCI (%)", 
                    title=f"Dynamic Rolling Window TCI (Window={window_size}d, Step={step_size}d)",
                    labels={"value": "TCI (%)", "Date": "Time"}
                )
                fig_rolling.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig_rolling, use_container_width=True)

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
