"""
Q-RiskNet India — Financial Network Science Section
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
import pandas as pd

import src.network.mst as mst
import src.network.spectral as spectral
import src.network.network_runner as net_runner
import src.diagnostics.logger as diag
from dashboard.components.charts import render_network_graph, render_mst_graph
from dashboard.components.exports import download_csv


def render_page(returns_df):
    """Renders the Financial Network Science page."""

    st.markdown("### Risk Network & Topology")
    st.caption("Visualises which sectors form the core of systemic risk transmission and how they are connected.")

    spill_df = st.session_state.get("spillover_df")

    if spill_df is None:
        st.info(
            "**No spillover data available.** Navigate to **Connectedness** and click **Run Analysis** first, "
            "or return to **Overview** to auto-compute baseline spillovers."
        )
        return

    # ── Network Graph ──────────────────────────────────────────────────
    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl2:
        st.markdown("**Graph controls**")
        if st.button("↺ Reset view", key="btn_reset_net_graph"):
            st.session_state["key_net_graph"] = st.session_state.get("key_net_graph", 0) + 1
        min_edge = st.slider(
            "Min edge strength (%)",
            0.0, 15.0, 2.0, 0.5,
            help="Hide edges below this spillover threshold to reduce clutter"
        )
        comm_mode = st.radio("Clusters", ["Auto-detect", "Manual"], horizontal=True)
        if comm_mode == "Manual":
            max_c = max(2, len(spill_df.columns) - 1)
            n_comm = st.slider("Number of clusters", 2, max_c, min(3, max_c)) if max_c > 2 else 2
        else:
            n_comm = "auto"

    with ctrl1:
        try:
            comms = spectral.detect_communities(spill_df, n_communities=n_comm)
            net_key = f"net_graph_{st.session_state.get('key_net_graph', 0)}"
            render_network_graph(spill_df, comms, min_edge, layout_style="circular", key=net_key)
        except Exception as e:
            diag.log_error("Network rendering failure", e)
            st.error(f"Network rendering error: {str(e)}")

    st.markdown("---")

    # ── Key Systemic Sectors ───────────────────────────────────────────
    st.markdown("### Key systemic sectors")

    try:
        net_res = net_runner.run_all_network_analysis(
            spill_df, returns_df, threshold_pct=min_edge, save_reports=True
        )
        cent_df = net_res["centrality_df"]
        gs = net_res["global_stats"]

        # 4 summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        top_transmitter = (
            cent_df.sort_values("Out_Degree_Export", ascending=False).iloc[0]["Sector"]
            if not cent_df.empty else "N/A"
        )
        top_receiver = (
            cent_df.sort_values("In_Degree_Import", ascending=False).iloc[0]["Sector"]
            if not cent_df.empty else "N/A"
        )
        top_hub = (
            cent_df.sort_values("PageRank_Centrality", ascending=False).iloc[0]["Sector"]
            if "PageRank_Centrality" in cent_df.columns and not cent_df.empty else "N/A"
        )
        top_bridge = (
            cent_df.sort_values("Betweenness_Centrality", ascending=False).iloc[0]["Sector"]
            if "Betweenness_Centrality" in cent_df.columns and not cent_df.empty else "N/A"
        )

        k1.metric("Top Risk Transmitter", top_transmitter,
                  help="Sector with the highest total outbound risk spillover")
        k2.metric("Top Risk Receiver", top_receiver,
                  help="Sector with the highest total inbound risk spillover")
        k3.metric("Most Systemically Important", top_hub,
                  help="Sector with the highest network influence (PageRank)")
        k4.metric("Key Bridge Sector", top_bridge,
                  help="Sector that most frequently lies on the shortest risk-transmission paths (betweenness)")

        # Simplified ranking table — friendly columns only
        friendly_cols = {
            "Out_Degree_Export": "Risk Transmitted (%)",
            "In_Degree_Import": "Risk Received (%)",
            "Net_Degree_Export": "Net Risk Flow (%)"
        }
        show_cols = ["Sector"] + [v for k, v in friendly_cols.items() if k in cent_df.columns]
        friendly_df = cent_df.rename(columns=friendly_cols)[show_cols]
        friendly_df = friendly_df.sort_values("Risk Transmitted (%)", ascending=False) if "Risk Transmitted (%)" in friendly_df.columns else friendly_df

        st.dataframe(friendly_df, use_container_width=True, hide_index=True)

        with st.expander("Advanced network metrics"):
            st.caption("PageRank = systemic influence. Betweenness = bridge/contagion role. Eigenvector = connection to other important sectors.")
            full_rename = {
                "Out_Degree_Export": "Risk Transmitted (%)",
                "In_Degree_Import": "Risk Received (%)",
                "Net_Degree_Export": "Net Risk Flow (%)",
                "PageRank_Centrality": "Systemic Influence (PageRank)",
                "Eigenvector_Centrality": "Systemic Importance (Eigenvector)",
                "Betweenness_Centrality": "Bridge Role (Betweenness)",
                "Closeness_Centrality": "Closeness"
            }
            st.dataframe(cent_df.rename(columns=full_rename), use_container_width=True, hide_index=True)

            st.markdown(f"""
            **Network summary:**
            - Risk connections above threshold: **{gs.get('Edge_Count', '—')}**
            - Network density: **{gs.get('Network_Density', 0):.3f}**
            """)

            with st.expander("Download network metrics"):
                download_csv(cent_df.rename(columns=full_rename), "network_systemic_rankings.csv", key="dl_cent")

    except Exception as e:
        st.error(f"Centrality calculation error: {str(e)}")

    st.markdown("---")

    # ── Minimum Spanning Tree ─────────────────────────────────────────
    st.markdown("### Risk Backbone (Minimum Spanning Tree — Prim's Algorithm)")
    st.caption(
        "Extracts the acyclic topological backbone connecting all 10 sectoral indices via the strongest risk channels "
        "using Prim's greedy minimum spanning tree algorithm."
    )

    mst_col1, mst_col2 = st.columns([1, 1])
    with mst_col1:
        mst_source = st.selectbox(
            "Backbone Source Space",
            ["Directed Spillover Matrix (Diebold-Yilmaz / GJR-GARCH)", "Pearson Correlation Distance (Mantegna 1999)"],
            index=0,
            help="Extract backbone from directional systemic spillovers or pairwise return correlation distance"
        )
    with mst_col2:
        if "Spillover" in mst_source:
            pairwise_rule = st.selectbox(
                "Pairwise Spillover Rule (Between Sector i and j)",
                ["Sum: Total Bilateral (S_ij + S_ji)", "Max: Peak Contagion (max(S_ij, S_ji))"],
                index=0,
                help=(
                    "Sum: Total 2-way bilateral spillover exchange. "
                    "Max: Peak one-way contagion vulnerability."
                )
            )
        else:
            st.markdown(
                "<div style='font-size:0.75rem; color:#94a3b8; padding-top:28px;'>"
                "Mantegna metric: <code>d_ij = √(2(1 - ρ_ij))</code> (Symmetric: ρ_ij = ρ_ji)"
                "</div>",
                unsafe_allow_html=True
            )
            pairwise_rule = None

    try:
        if "Spillover" in mst_source:
            method_key = "max" if (pairwise_rule and "Max" in pairwise_rule) else "sum"
            dist_matrix, weights_df = mst.compute_spillover_distance(spill_df, method=method_key)
            mst_g = mst.construct_mst(dist_matrix, algorithm="prim", weights_df=weights_df)
            rule_label = "Peak Contagion: max(S_ij, S_ji)" if method_key == "max" else "Total Bilateral: S_ij + S_ji"
            plot_title = f"Systemic Risk Backbone — Prim's MST (Spillover: {rule_label})"
        else:
            dist_matrix = mst.compute_correlation_distance(returns_df)
            weights_df = None
            mst_g = mst.construct_mst(dist_matrix, algorithm="prim")
            plot_title = "Systemic Risk Backbone — Prim's MST (Mantegna Correlation Distance)"

        render_mst_graph(mst_g, dist_matrix, title=plot_title)

        # MST Edges Detail Table
        mst_edges = []
        for u, v, d in mst_g.edges(data=True):
            row = {
                "Sector 1": u,
                "Sector 2": v,
                "MST Distance": round(float(d.get("weight", 0.0)), 4)
            }
            if "bilateral_weight" in d:
                row["Bilateral Spillover (%)"] = round(float(d["bilateral_weight"]), 2)
            elif "Spillover" not in mst_source:
                d_val = float(d.get("weight", 0.0))
                row["Pearson Correlation (r)"] = round(1.0 - 0.5 * (d_val ** 2), 4)
            mst_edges.append(row)

        mst_edges_df = pd.DataFrame(mst_edges)
        if "Bilateral Spillover (%)" in mst_edges_df.columns:
            mst_edges_df = mst_edges_df.sort_values("Bilateral Spillover (%)", ascending=False)
        elif "Pearson Correlation (r)" in mst_edges_df.columns:
            mst_edges_df = mst_edges_df.sort_values("Pearson Correlation (r)", ascending=False)

        with st.expander("View MST Backbone Edges (Prim's Algorithm)", expanded=False):
            st.dataframe(mst_edges_df, use_container_width=True, hide_index=True)
            download_csv(mst_edges_df, "mst_backbone_edges.csv", key="dl_mst_edges")

    except Exception as e:
        st.error(f"MST error: {str(e)}")
