"""
Q-RiskNet India — Executive Plotly Chart Components
Copyright (c) 2026 Bibek Rout
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import src.visualization.plotly_plots as vis

# Plotly configuration for static, stable charts (no accidental scroll-zoom or lost axes)
CHART_CONFIG = {
    'scrollZoom': False,
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'],
    'doubleClick': 'reset'
}


def _render_plotly(fig, height=None):
    """Internal helper to enforce fixed axes and dark template for static, clean chart rendering."""
    if height is not None:
        fig.update_layout(height=height)
    fig.update_layout(template="plotly_dark")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


def render_prices_chart(prices_df):
    """Renders base-100 normalized price chart."""
    norm_prices = (prices_df / prices_df.iloc[0]) * 100
    fig = px.line(
        norm_prices,
        x=norm_prices.index,
        y=norm_prices.columns,
        title="Base-100 Normalized Price Trends",
        labels={"value": "Normalized Level (Base=100)", "variable": "Sector"}
    )
    _render_plotly(fig, height=450)


def render_drawdowns_chart(drawdowns_df):
    """Renders peak-to-trough percentage drawdown chart."""
    fig = px.line(
        drawdowns_df,
        x=drawdowns_df.index,
        y=drawdowns_df.columns,
        title="Sectoral Historical Drawdowns (%)",
        labels={"value": "Drawdown (%)", "variable": "Sector"}
    )
    _render_plotly(fig, height=400)


def render_rolling_volatility_chart(vol_df, window_label="20-Day"):
    """Renders rolling annualized volatility chart."""
    fig = px.line(
        vol_df,
        x=vol_df.index,
        y=vol_df.columns,
        title=f"Annualized Rolling Volatility ({window_label} Window %)",
        labels={"value": "Annualized Volatility (%)", "variable": "Sector"}
    )
    _render_plotly(fig, height=400)


def render_conditional_volatility_chart(returns_series, cond_vol_series, model_name):
    """Overlays return series with ±2σ conditional volatility bands."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=returns_series.index, y=returns_series.values,
        mode='lines', name='Daily Return (%)',
        line=dict(color='#64748b', width=1), opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=cond_vol_series.index, y=2.0 * cond_vol_series.values,
        mode='lines', name='+2σ Upper Volatility Band',
        line=dict(color='#ef4444', width=1.8, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=cond_vol_series.index, y=-2.0 * cond_vol_series.values,
        mode='lines', name='-2σ Lower Volatility Band',
        line=dict(color='#ef4444', width=1.8, dash='dash')
    ))
    fig.update_layout(
        title=f"Conditional Volatility Envelopes (±2σ) - {model_name} ({returns_series.name})",
        xaxis_title="Date", yaxis_title="Return (%) / Volatility"
    )
    _render_plotly(fig, height=450)


def render_forecast_benchmark_chart(preds_df, target_sector):
    """Renders Out-of-Sample actual returns vs model predictions line plot."""
    fig = px.line(
        preds_df,
        x=preds_df.index,
        y=preds_df.columns,
        title=f"Out-of-Sample Forecast Predictions Benchmark ({target_sector})",
        labels={"value": "Daily Log Return (%)", "variable": "Model / Actual"}
    )
    _render_plotly(fig, height=450)


def render_feature_importance_chart(feat_series, title_str):
    """Renders Feature Importance bar chart."""
    top_feats = feat_series.sort_values(ascending=True).tail(10)
    fig = px.bar(
        x=top_feats.values,
        y=top_feats.index,
        orientation='h',
        title=title_str,
        labels={"x": "Gini Importance", "y": "Lagged Feature"}
    )
    _render_plotly(fig, height=380)


def render_qvar_heatmap(coeff_matrix, quantile_val):
    """Renders QVAR Autoregressive Coefficient Matrix Heatmap."""
    fig = px.imshow(
        coeff_matrix,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=f"QVAR Coefficient Matrix Φ₁(τ={quantile_val:.2f}) [Row=Target, Col=Source]"
    )
    _render_plotly(fig, height=420)


def render_qvar_girf_chart(girf_df, shocked_sector, quantile_val):
    """Renders Generalized Impulse Response Function (GIRF) curves."""
    fig = px.line(
        girf_df,
        x=girf_df.index,
        y=girf_df.columns,
        title=f"QVAR Generalized Impulse Responses (GIRF, τ={quantile_val:.2f}) to +2σ Shock in {shocked_sector}",
        labels={"x": "Horizon (Days)", "value": "Response (%)", "variable": "Sector"}
    )
    _render_plotly(fig, height=420)


def render_acf_pacf_chart(lags, acf_vals, pacf_vals, sector_name):
    """Renders ACF and PACF bar charts."""
    col_a, col_p = st.columns(2)
    with col_a:
        fig_acf = px.bar(
            x=lags, y=acf_vals,
            title=f"Autocorrelation (ACF) - {sector_name}",
            labels={"x": "Lag", "y": "ACF"}
        )
        _render_plotly(fig_acf, height=350)
    with col_p:
        fig_pacf = px.bar(
            x=lags, y=pacf_vals,
            title=f"Partial Autocorrelation (PACF) - {sector_name}",
            labels={"x": "Lag", "y": "PACF"}
        )
        _render_plotly(fig_pacf, height=350)


def render_kde_comparison_chart(series, x_grid, norm_pdf):
    """Renders Empirical Return Histogram with Gaussian PDF Overlay."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=series, histnorm='probability density',
        name='Empirical Returns', marker_color='#3b82f6', opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=x_grid, y=norm_pdf, mode='lines',
        name='Gaussian Normal Fit', line=dict(color='#ef4444', width=2.5, dash='dash')
    ))
    fig.update_layout(
        title=f"Empirical Returns Distribution vs Gaussian Fit ({series.name})",
        xaxis_title="Return (%)", yaxis_title="Density"
    )
    _render_plotly(fig, height=400)


def render_rolling_variance_chart(roll_var, sector_name):
    """Renders 20-Day Rolling Variance plot for volatility clustering."""
    fig = px.line(
        roll_var, x=roll_var.index, y=roll_var.values,
        title=f"Rolling Sample Variance (20-Day Window) - {sector_name}",
        labels={"value": "Variance", "index": "Date"}
    )
    _render_plotly(fig, height=380)


def render_correlation_chart(corr_df):
    """Renders Pearson correlation matrix heatmap."""
    fig = vis.render_correlation_heatmap(corr_df)
    _render_plotly(fig, height=500)


def render_spillover_charts(metrics):
    """Renders Net Spillover bar chart and TO/FROM gross spillover bar chart."""
    col1, col2 = st.columns(2)
    with col1:
        net_series = metrics['NET'].sort_values()
        fig_net = px.bar(
            x=net_series.values, y=net_series.index,
            orientation='h', color=net_series.values,
            color_continuous_scale="RdYlGn_r",
            title="Net Directional Spillover (TO - FROM)",
            labels={"x": "Net Spillover (%)", "y": "Sector"}
        )
        _render_plotly(fig_net, height=400)

    with col2:
        to_from_df = pd.DataFrame({"TO OTHERS": metrics['TO'], "FROM OTHERS": metrics['FROM']})
        fig_tf = px.bar(
            to_from_df, barmode='group',
            title="Gross Directional Spillovers (TO vs FROM)"
        )
        _render_plotly(fig_tf, height=400)


def render_network_graph(spill_df, comms, min_edge, layout_style):
    """Renders Plotly directed risk spillover network graph."""
    fig = vis.render_spillover_network(
        spill_df, communities=comms,
        min_threshold_pct=min_edge, layout_type=layout_style
    )
    _render_plotly(fig, height=600)


def render_mst_graph(mst_graph, dist_matrix):
    """Renders Minimum Spanning Tree (MST) graph."""
    fig = vis.render_mst_network(mst_graph, dist_matrix)
    _render_plotly(fig, height=550)


def render_rolling_tci_chart(rolling_tci_df, window_size, step_size):
    """Renders rolling window TCI line chart."""
    fig = px.line(
        rolling_tci_df, y="Rolling TCI (%)",
        title=f"Dynamic Rolling Window TCI (Window={window_size}d, Step={step_size}d)",
        labels={"value": "TCI (%)", "Date": "Time"}
    )
    _render_plotly(fig, height=450)
