import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import src.visualization.plotly_plots as vis

def render_prices_chart(prices_df):
    """
    Renders base-100 normalized price chart.
    """
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

def render_drawdowns_chart(drawdowns_df):
    """
    Renders peak-to-trough percentage drawdown chart.
    """
    fig_dd = px.line(
        drawdowns_df,
        x=drawdowns_df.index,
        y=drawdowns_df.columns,
        title="Sectoral Historical Drawdowns (%)",
        labels={"value": "Drawdown (%)", "variable": "Sector"}
    )
    fig_dd.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_dd, use_container_width=True)

def render_rolling_volatility_chart(vol_df, window_label="20-Day"):
    """
    Renders rolling annualized volatility chart.
    """
    fig_vol = px.line(
        vol_df,
        x=vol_df.index,
        y=vol_df.columns,
        title=f"Annualized Rolling Volatility ({window_label} Window %)",
        labels={"value": "Annualized Volatility (%)", "variable": "Sector"}
    )
    fig_vol.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_vol, use_container_width=True)

def render_correlation_chart(corr_df):
    """
    Renders Pearson correlation matrix heatmap.
    """
    fig_corr = vis.render_correlation_heatmap(corr_df)
    st.plotly_chart(fig_corr, use_container_width=True)

def render_spillover_charts(metrics):
    """
    Renders Net Spillover bar chart and TO/FROM gross spillover bar chart.
    """
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

def render_network_graph(spill_df, comms, min_edge, layout_style):
    """
    Renders Plotly directed risk spillover network.
    """
    fig_net_graph = vis.render_spillover_network(
        spill_df, 
        communities=comms, 
        min_threshold_pct=min_edge, 
        layout_type=layout_style
    )
    st.plotly_chart(fig_net_graph, use_container_width=True)

def render_mst_graph(mst_graph, dist_matrix):
    """
    Renders Minimum Spanning Tree (MST) graph.
    """
    fig_mst = vis.render_mst_network(mst_graph, dist_matrix)
    st.plotly_chart(fig_mst, use_container_width=True)

def render_rolling_tci_chart(rolling_tci_df, window_size, step_size):
    """
    Renders rolling window TCI line chart.
    """
    fig_rolling = px.line(
        rolling_tci_df, 
        y="Rolling TCI (%)", 
        title=f"Dynamic Rolling Window TCI (Window={window_size}d, Step={step_size}d)",
        labels={"value": "TCI (%)", "Date": "Time"}
    )
    fig_rolling.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_rolling, use_container_width=True)
