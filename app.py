import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import custom scripts
import data_manager as dm
import models as md
import network_utils as nu
import diagnostics as diag

# =====================================================================
# Page Configurations & Styling
# =====================================================================
st.set_page_config(
    layout="wide",
    page_title="Indian Stock Market Sectoral Risk Connectedness Dashboard",
    page_icon="🕸️"
)

# Custom premium UI style injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0f172a; /* Slate 900 */
        color: #f8fafc;
    }
    
    /* Header Card styling */
    .header-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #312e81;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .header-card h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 10px 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Custom tab container styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px 8px 0px 0px;
        color: #94a3b8;
        padding: 10px 20px;
        font-weight: 600;
        border: 1px solid #334155;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: white !important;
        border-color: #4f46e5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize logs in session state
if 'diagnostics_logs' not in st.session_state:
    st.session_state['diagnostics_logs'] = []
    diag.log_info("Dashboard started. System initialized.")

# =====================================================================
# Sidebar & Configuration Setup
# =====================================================================
st.sidebar.markdown("<h2 style='text-align: center; color: #818cf8;'>NSE India Risk Analytics</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Data selection
st.sidebar.subheader("📅 Data Configuration")
selected_sectors = st.sidebar.multiselect(
    "Select Sectoral Indices",
    options=list(dm.TICKER_MAP.keys()),
    default=list(dm.TICKER_MAP.keys())[:7] # default to first 7 sectors
)

today = datetime.today()
five_years_ago = today - timedelta(days=5 * 365)
start_date = st.sidebar.date_input("Start Date", value=five_years_ago)
end_date = st.sidebar.date_input("End Date", value=today)

# Modeling parameters
st.sidebar.subheader("⚙️ Model Settings")
model_choice = st.sidebar.radio("Model Type", ["Quantile VAR (QVAR)", "Quantile LSTM"])
quantile = st.sidebar.slider("Quantile Value (τ)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)

optim_mode = st.sidebar.radio("Hyperparameter Engine", ["⚡ Auto-Optimized (Recommended)", "⚙️ Custom Parameters"])

# Defaults
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

st.sidebar.markdown("---")
st.sidebar.info(
    "Data is sourced dynamically from Yahoo Finance (EOD close is 100% free and has zero lag)."
)

# =====================================================================
# Main Application Flow
# =====================================================================
# Display Header Card
st.markdown("""
<div class="header-card">
    <h1>Indian Stock Market Risk Connectedness</h1>
    <p>Analyze how risk, volatility, and shocks transmit across the key sectoral indices of the National Stock Exchange (NSE) of India. 
       This dashboard implements the traditional linear <b>QVAR</b> framework alongside a non-linear <b>Quantile LSTM</b> neural network to capture tail-risk connectedness during periods of stress.</p>
</div>
""", unsafe_allow_html=True)

# Create tabs immediately so they are available even on errors
tab_data, tab_spillover, tab_network, tab_rolling, tab_diag = st.tabs([
    "📊 Data Center", 
    "📈 Volatility Spillover", 
    "🕸️ Network Topology", 
    "🕒 Dynamic TCI",
    "🔧 Diagnostics & Logs"
])

# Ensure adequate sectors are selected
if len(selected_sectors) < 2:
    st.error("⚠️ Please select at least two sectoral indices in the sidebar to perform network analysis.")
    st.stop()

# Helper function to cache downloaded price data
@st.cache_data(show_spinner="Downloading data from Yahoo Finance...")
def load_data(sectors, start, end):
    return dm.download_data(sectors, start, end)

# 1. Download & Process Data with Safety Wrapper
prices_df = pd.DataFrame()
returns_df = pd.DataFrame()
model_input = pd.DataFrame()
data_load_error = None

try:
    prices_df = load_data(selected_sectors, start_date, end_date)
    if prices_df.empty:
        raise ValueError("Prices dataframe is empty. No data returned from Yahoo Finance.")
    returns_df = dm.calculate_log_returns(prices_df)
    
    if volatility_proxy == "Log Returns":
        model_input = returns_df
    else:
        garch_cols = {}
        with st.spinner("Extracting conditional volatilities using GARCH(1,1)..."):
            for col in returns_df.columns:
                garch_cols[col] = dm.estimate_garch_volatility(returns_df[col])
        model_input = pd.DataFrame(garch_cols, index=returns_df.index)
except Exception as e:
    data_load_error = e
    diag.log_error("Exception occurred during data downloading or preprocessing", e)

# If data loading fails, show a clean message and direct to Diagnostics tab
if data_load_error is not None:
    st.error(f"❌ **Data Processing Error**: {str(data_load_error)}")
    st.warning("Go to the **Diagnostics & Logs** tab at the top to view the full backend stack trace.")
    st.stop()

# Create Session State to cache model runs
if 'last_run_config' not in st.session_state:
    st.session_state['last_run_config'] = None
if 'spillover_df' not in st.session_state:
    st.session_state['spillover_df'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = None
if 'engine_error' not in st.session_state:
    st.session_state['engine_error'] = None

# =====================================================================
# TAB 1: DATA CENTER
# =====================================================================
with tab_data:
    st.subheader("Explore Sector Performance & Volatility")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Index Price Performance Chart (Normalized to base 100)
        norm_prices = (prices_df / prices_df.iloc[0]) * 100
        fig_prices = px.line(
            norm_prices, 
            title="NSE Sectoral Cumulative Performance (Base 100)",
            labels={"value": "Normalized Index Value", "index": "Date"},
            template="plotly_dark"
        )
        fig_prices.update_layout(
            paper_bgcolor='rgba(15, 23, 42, 1)',
            plot_bgcolor='rgba(15, 23, 42, 1)',
            font_family="Outfit"
        )
        st.plotly_chart(fig_prices, use_container_width=True)
        
    with col2:
        # Pearson Correlation Heatmap
        corr = returns_df.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            hovertemplate='Corr between %{x} and %{y}: <b>%{z:.4f}</b><extra></extra>'
        ))
        fig_corr.update_layout(
            title="Sectoral Correlation Matrix",
            template="plotly_dark",
            paper_bgcolor='rgba(15, 23, 42, 1)',
            plot_bgcolor='rgba(15, 23, 42, 1)',
            font_family="Outfit"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    # Descriptive Statistics Table
    st.subheader("Econometric Summary Statistics")
    with st.spinner("Computing descriptive stats and ADF unit-root tests..."):
        desc_df = dm.get_descriptive_stats(model_input)
    st.dataframe(desc_df.set_index("Sector"), use_container_width=True)
    st.markdown("""
    > [!NOTE]
    > **Jarque-Bera (JB)** tests the null hypothesis of normality (p-val < 0.05 rejects normality). 
    > **Augmented Dickey-Fuller (ADF)** tests for stationarity (p-val < 0.05 indicates the series is stationary, a prerequisite for stable modeling).
    """)

# =====================================================================
# TAB 2: VOLATILITY SPILLOVER ENGINE
# =====================================================================
with tab_spillover:
    st.subheader("Estimate Connectedness Indices")
    
    # Check if run parameters changed
    current_config = {
        "sectors": selected_sectors,
        "start": start_date,
        "end": end_date,
        "model": model_choice,
        "quantile": quantile,
        "proxy": volatility_proxy,
        "horizon": forecast_horizon,
        "lags": lags,
        "seq_len": seq_len,
        "epochs": epochs,
        "hidden": hidden_dim
    }
    
    run_btn = st.button("🔥 Run Connectedness Engine", use_container_width=True)
    
    if run_btn or st.session_state['spillover_df'] is None or st.session_state['last_run_config'] != current_config:
        # Reset error state before running
        st.session_state['engine_error'] = None
        
        try:
            with st.spinner("Training model & generating generalized forecast error variance decompositions..."):
                
                # Setup Model
                if model_choice == "Quantile VAR (QVAR)":
                    model = md.QVARModel(p=lags, quantile=quantile)
                    model.fit(model_input)
                else:
                    model = md.LSTMQuantileModel(
                        seq_len=seq_len, 
                        hidden_dim=hidden_dim, 
                        quantile=quantile, 
                        epochs=epochs
                    )
                    
                    progress_bar = st.progress(0, text="Training Quantile LSTM...")
                    
                    def update_progress(epoch, total_epochs):
                        pct = int((epoch / total_epochs) * 100)
                        progress_bar.progress(pct, text=f"Training Quantile LSTM (Epoch {epoch}/{total_epochs})")
                    
                    model.fit(model_input, progress_callback=update_progress)
                    progress_bar.empty()
                    
                # Compute Spillover Matrix
                spill_df = md.compute_spillover_matrix(model, model_input, horizon=forecast_horizon)
                metrics = md.calculate_connectedness_metrics(spill_df)
                
                # Save to Session State
                st.session_state['spillover_df'] = spill_df
                st.session_state['metrics'] = metrics
                st.session_state['last_run_config'] = current_config
        except Exception as e:
            st.session_state['engine_error'] = e
            st.session_state['spillover_df'] = None
            st.session_state['metrics'] = None
            diag.log_error("Exception occurred during model fitting/connectedness calculation", e)
            
    # Handle error display
    if st.session_state['engine_error'] is not None:
        st.error(f"❌ **Connectedness Engine Error**: {str(st.session_state['engine_error'])}")
        st.warning("Please check the **Diagnostics & Logs** tab at the top for details. Try reducing lags, increasing sequence length, or selecting more data.")
        
    # If we have calculated metrics successfully, display them
    if st.session_state['spillover_df'] is not None:
        spill_df = st.session_state['spillover_df']
        metrics = st.session_state['metrics']
        
        tci_val = metrics["TCI"]
        
        st.markdown("---")
        m_col1, m_col2 = st.columns([1, 3])
        with m_col1:
            st.metric(
                label="Total Connectedness Index (TCI)",
                value=f"{tci_val:.2f}%",
                help="Represents the overall percentage of forecast error variance that is shared across all sectors. High TCI indicates high systemic risk integration."
            )
            if tci_val > 70:
                st.warning("🚨 **Extreme Interconnectedness**: High risk transmission across sectors. Portfolio diversification is heavily compromised.")
            elif tci_val > 40:
                st.info("ℹ️ **Moderate Interconnectedness**: Normal system behavior. Risk transmission is active but regionalized.")
            else:
                st.success("🟢 **Low Interconnectedness**: Excellent diversification opportunities. Risk is mostly localized within sectors.")
                
        with m_col2:
            net_df = pd.DataFrame({
                "Sector": metrics["NET"].index,
                "Net Spillover (%)": metrics["NET"].values
            }).sort_values(by="Net Spillover (%)", ascending=False)
            
            net_df["Role"] = net_df["Net Spillover (%)"].apply(lambda val: "Net Transmitter" if val >= 0 else "Net Receiver")
            
            fig_net = px.bar(
                net_df, 
                x="Sector", 
                y="Net Spillover (%)",
                color="Role",
                color_discrete_map={"Net Transmitter": "#ef4444", "Net Receiver": "#3b82f6"},
                title="Sector Net Volatility Spillovers (TO - FROM)",
                template="plotly_dark"
            )
            fig_net.update_layout(
                paper_bgcolor='rgba(15, 23, 42, 1)',
                plot_bgcolor='rgba(15, 23, 42, 1)',
                font_family="Outfit"
            )
            st.plotly_chart(fig_net, use_container_width=True)
            
        # Full Spillover Table
        st.subheader("Diebold-Yilmaz Spillover Table")
        
        display_spill = spill_df.copy()
        display_spill["TO OTHERS"] = metrics["TO"]
        from_row = pd.Series(metrics["FROM"], name="FROM OTHERS")
        display_spill = pd.concat([display_spill, pd.DataFrame([from_row])])
        display_spill.loc["FROM OTHERS", "TO OTHERS"] = metrics["TCI"]
        
        st.dataframe(display_spill.style.format("{:.2f}%").background_gradient(cmap="Reds", subset=(spill_df.index, spill_df.columns)), use_container_width=True)
        st.caption("Rows represent receiving sectors (FROM); columns represent transmitting sectors (TO). Diagonal represents self-spillover.")

# =====================================================================
# TAB 3: NETWORK TOPOLOGY & CLUSTERS
# =====================================================================
with tab_network:
    if st.session_state['spillover_df'] is None:
        st.info("Please calculate connectedness metrics in the 'Volatility Spillover' tab first.")
    else:
        st.subheader("Visualize Risk Contagion Networks")
        
        spill_df = st.session_state['spillover_df']
        metrics = st.session_state['metrics']
        
        net_col1, net_col2 = st.columns([3, 1])
        
        with net_col2:
            st.markdown("### 🕸️ Network Controls")
            min_edge = st.slider(
                "Minimum Spillover Edge Threshold (%)", 
                min_value=0.0, 
                max_value=15.0, 
                value=2.0, 
                step=0.5,
                help="Filters out weak connections to make the network diagram readable."
            )
            layout_style = st.selectbox("Network Layout", ["circular", "spring"])
            
            comm_mode = st.radio("Community Detection Mode", ["Auto (Eigengap Heuristic)", "Manual Choice"], horizontal=True)
            
            if comm_mode == "Manual Choice":
                max_communities = max(2, len(selected_sectors) - 1)
                default_comm_val = min(3, max_communities)
                n_comm = st.slider(
                    "Target Communities", 
                    min_value=2, 
                    max_value=max_communities, 
                    value=default_comm_val,
                    help="Groups sectors via Spectral Clustering."
                )
            else:
                n_comm = "auto"
            
        with net_col1:
            try:
                # Spectral clustering communities with Eigengap Heuristic support
                comms = nu.detect_communities(spill_df, n_communities=n_comm)
                
                # Interactive Directed Network Diagram
                fig_net = nu.draw_plotly_network(
                    spill_df, 
                    metrics, 
                    communities=comms, 
                    layout_type=layout_style, 
                    min_edge_threshold=min_edge
                )
                st.plotly_chart(fig_net, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate network visualization: {str(e)}")
                diag.log_error("Exception occurred during network layout generation", e)
            
        st.markdown("---")
        st.subheader("🔍 Core Network Backbones & Clustering")
        
        col_mst, col_comm = st.columns([1, 1])
        
        with col_mst:
            try:
                dist_mat = nu.compute_correlation_distance(returns_df.corr())
                mst = nu.build_mst_graph(dist_mat)
                fig_mst = nu.draw_mst_network(mst, returns_df.corr(), layout_type="spring")
                st.plotly_chart(fig_mst, use_container_width=True)
                st.caption("The MST extracts the most significant correlation pathways, mapping the structural backbone of market co-movement.")
            except Exception as e:
                st.error("Could not compute Minimum Spanning Tree.")
                diag.log_error("MST calculation failure", e)
            
        with col_comm:
            try:
                # Group sectors by community
                comm_groups = {}
                for sec, label in comms.items():
                    comm_groups.setdefault(label, []).append(sec)
                    
                st.markdown("### Detected Communities (Spectral Clustering)")
                st.markdown("Community divisions are grouped by average bi-directional volatility transmission:")
                
                colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]
                for label, group in sorted(comm_groups.items()):
                    color = colors[label % len(colors)]
                    sect_list = ", ".join(f"**{s}**" for s in group)
                    st.markdown(f"""
                    <div style='padding: 12px; border-radius: 8px; border-left: 5px solid {color}; background-color: #1e293b; margin-bottom: 10px;'>
                        <h5 style='margin: 0; color: {color};'>Community {label + 1}</h5>
                        <p style='margin: 5px 0 0 0; font-size: 0.95rem; color: #cbd5e1;'>Sectors: {sect_list}</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.caption("Spectral Clustering detects which sectors form transmission feedback loops, acting as tightly integrated risk clusters.")
            except Exception as e:
                st.error("Failed to render community groupings.")

# =====================================================================
# TAB 4: DYNAMIC ROLLINGS (TIME-VARYING TCI)
# =====================================================================
with tab_rolling:
    st.subheader("Time-Varying Connectedness Dynamics")
    
    st.markdown("""
    Risk connectedness changes significantly during global or domestic crises. 
    By running a rolling window analysis, we can track the evolution of the Total Connectedness Index (TCI) over time.
    """)
    
    roll_col1, roll_col2 = st.columns([1, 3])
    
    with roll_col1:
        st.markdown("### 🕒 Rolling Controls")
        window_size = st.slider("Rolling Window (Trading Days)", min_value=100, max_value=500, value=250, step=20)
        roll_step = st.slider("Evaluation Step (Days)", min_value=5, max_value=50, value=15, step=5)
        
        start_roll_btn = st.button("🚀 Calculate Dynamic TCI", use_container_width=True)
        
    with roll_col2:
        if start_roll_btn:
            total_len = len(model_input)
            if total_len <= window_size:
                st.error("Insufficient data for the chosen window size. Please expand date range or decrease window size.")
            else:
                rolling_dates = []
                rolling_tcis = []
                
                # Setup model
                if model_choice == "Quantile VAR (QVAR)":
                    roll_model = md.QVARModel(p=lags, quantile=quantile)
                else:
                    roll_model = md.LSTMQuantileModel(
                        seq_len=seq_len, 
                        hidden_dim=hidden_dim, 
                        quantile=quantile, 
                        epochs=15 # fewer epochs for rolling speed
                    )
                
                # Progress bar
                n_steps = (total_len - window_size) // roll_step
                roll_progress = st.progress(0, text="Calculating rolling spillovers...")
                
                step_count = 0
                error_count = 0
                
                for start_idx in range(0, total_len - window_size, roll_step):
                    end_idx = start_idx + window_size
                    window_df = model_input.iloc[start_idx:end_idx]
                    
                    try:
                        roll_model.fit(window_df)
                        w_spill = md.compute_spillover_matrix(roll_model, window_df, horizon=forecast_horizon)
                        w_metrics = md.calculate_connectedness_metrics(w_spill)
                        
                        rolling_dates.append(model_input.index[end_idx - 1])
                        rolling_tcis.append(w_metrics["TCI"])
                    except Exception as e:
                        error_count += 1
                        diag.log_warning(f"Convergence issue in rolling window step {step_count}: {str(e)}")
                        
                    step_count += 1
                    pct = min(int((step_count / n_steps) * 100), 100)
                    roll_progress.progress(pct, text=f"Computing window {step_count}/{n_steps}")
                    
                roll_progress.empty()
                
                if error_count > 0:
                    diag.log_warning(f"Rolling calculations finished with {error_count} failed steps out of {step_count}.")
                
                if not rolling_tcis:
                    st.error("❌ **All rolling window steps failed to converge**. Please check logs, reduce lags, or select different sectors.")
                else:
                    roll_df = pd.DataFrame({
                        "Date": rolling_dates,
                        "Total Connectedness Index (TCI %)": rolling_tcis
                    })
                    
                    fig_roll = px.line(
                        roll_df, 
                        x="Date", 
                        y="Total Connectedness Index (TCI %)",
                        title="Evolution of Total Risk Connectedness (Rolling TCI)",
                        template="plotly_dark"
                    )
                    fig_roll.update_traces(line_color="#818cf8", line_width=2.5)
                    fig_roll.update_layout(
                        paper_bgcolor='rgba(15, 23, 42, 1)',
                        plot_bgcolor='rgba(15, 23, 42, 1)',
                        font_family="Outfit"
                    )
                    st.plotly_chart(fig_roll, use_container_width=True)
                    st.caption("Rolling TCI peaks capture historical stress events, structural shifts, and heightened systematic risk transmissions.")
        else:
            st.info("Click 'Calculate Dynamic TCI' to run the rolling window analysis. Note: LSTM rolling models may take a few minutes.")

# =====================================================================
# TAB 5: CENTRALIZED DIAGNOSTIC & LOGGING CONSOLE
# =====================================================================
with tab_diag:
    st.subheader("🔧 System Backend & Diagnostics Console")
    st.markdown("""
    This console provides real-time visibility into the backend execution flow, data dimensions, model loss convergence values, 
    and full Python tracebacks in case of any runtime errors.
    """)
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Selected Sectors", len(selected_sectors))
    with col_stat2:
        st.metric("Data Range Points", len(model_input) if not model_input.empty else 0)
    with col_stat3:
        st.metric("Total Log entries", len(st.session_state.get('diagnostics_logs', [])))
        
    log_filter = st.radio("Filter Log Level", ["ALL", "INFO", "WARNING", "ERROR"], horizontal=True)
    
    clear_logs_btn = st.button("🗑️ Clear Diagnostic Logs")
    if clear_logs_btn:
        st.session_state['diagnostics_logs'] = []
        diag.log_info("Logs cleared by user.")
        st.rerun()
        
    st.markdown("### Log Records")
    
    logs = st.session_state.get('diagnostics_logs', [])
    
    if not logs:
        st.info("No log entries recorded.")
    else:
        # Loop backwards to show newest logs first
        for entry in reversed(logs):
            if log_filter != "ALL" and entry["level"] != log_filter:
                continue
                
            color = "#3b82f6" # Info = Blue
            if entry["level"] == "WARNING":
                color = "#f59e0b" # Warning = Orange
            elif entry["level"] == "ERROR":
                color = "#ef4444" # Error = Red
                
            st.markdown(f"""
            <div style='padding: 8px 12px; border-radius: 6px; border-left: 4px solid {color}; background-color: #1e293b; margin-bottom: 8px; font-family: monospace; font-size: 0.9rem;'>
                <span style='color: #64748b;'>[{entry['timestamp']}]</span> 
                <span style='color: {color}; font-weight: bold;'>[{entry['level']}]</span> 
                <span style='color: #f1f5f9;'>{entry['message']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # If log has a traceback, show expandable details
            if entry["traceback"] is not None:
                with st.expander("🔍 Show Full Traceback Details"):
                    st.code(entry["traceback"], language="python")
