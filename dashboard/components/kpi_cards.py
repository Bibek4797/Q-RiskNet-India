import streamlit as st

def render_kpi_cards(metrics):
    """
    Renders key performance metric cards for TCI, Top Systemic Transmitter, and Top Risk Receiver.
    """
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Total Connectedness Index (TCI)", f"{metrics['TCI']:.2f}%", help="Systemic risk index of cross-sector connectedness.")
    with col_m2:
        max_transmitter = metrics['NET'].idxmax()
        st.metric("Top Systemic Transmitter", max_transmitter, f"+{metrics['NET'][max_transmitter]:.2f}% Net Outflow")
    with col_m3:
        max_receiver = metrics['NET'].idxmin()
        st.metric("Top Risk Receiver", max_receiver, f"{metrics['NET'][max_receiver]:.2f}% Net Inflow")
