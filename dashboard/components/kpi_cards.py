"""
Q-RiskNet India — KPI Cards Component
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st


def render_kpi_cards(metrics):
    """
    Renders clean KPI cards for TCI, Top Risk Transmitter, and Top Risk Receiver.
    Consistent semantic color: red for transmitter (exporting risk), green for receiver (absorbing risk).
    """
    tci_val = metrics['TCI']
    net_series = metrics['NET']
    max_transmitter = net_series.idxmax()
    max_receiver = net_series.idxmin()
    net_out = net_series[max_transmitter]
    net_in = net_series[max_receiver]

    # Semantic TCI color: green < 40%, amber 40–65%, red > 65%
    if tci_val < 40:
        tci_color = "#22c55e"
        tci_label = "Low — sectors are relatively independent"
    elif tci_val < 65:
        tci_color = "#f59e0b"
        tci_label = "Moderate — meaningful cross-sector risk sharing"
    else:
        tci_color = "#ef4444"
        tci_label = "High — shocks spread rapidly across sectors"

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.025); border:1px solid rgba(99,102,241,0.2);
                    border-radius:10px; padding:16px 18px;'>
            <div style='font-size:0.68rem; font-weight:700; color:#475569; text-transform:uppercase;
                        letter-spacing:0.09em; margin-bottom:8px;'>
                Systemic Connectedness (TCI)
            </div>
            <div style='font-family:"Space Grotesk",sans-serif; font-size:2rem; font-weight:700;
                        color:{tci_color}; line-height:1;'>{tci_val:.1f}%</div>
            <div style='font-size:0.75rem; color:{tci_color}; opacity:0.8; margin-top:5px;'>{tci_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background:rgba(239,68,68,0.03); border:1px solid rgba(239,68,68,0.18);
                    border-radius:10px; padding:16px 18px;'>
            <div style='font-size:0.68rem; font-weight:700; color:#475569; text-transform:uppercase;
                        letter-spacing:0.09em; margin-bottom:8px;'>Top Risk Transmitter</div>
            <div style='font-family:"Space Grotesk",sans-serif; font-size:1.5rem; font-weight:700;
                        color:#fca5a5; line-height:1.2;'>{max_transmitter}</div>
            <div style='font-size:0.75rem; color:#f87171; margin-top:5px;'>
                +{net_out:.1f}% net outflow</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background:rgba(34,197,94,0.03); border:1px solid rgba(34,197,94,0.18);
                    border-radius:10px; padding:16px 18px;'>
            <div style='font-size:0.68rem; font-weight:700; color:#475569; text-transform:uppercase;
                        letter-spacing:0.09em; margin-bottom:8px;'>Top Risk Receiver</div>
            <div style='font-family:"Space Grotesk",sans-serif; font-size:1.5rem; font-weight:700;
                        color:#86efac; line-height:1.2;'>{max_receiver}</div>
            <div style='font-size:0.75rem; color:#4ade80; margin-top:5px;'>
                {net_in:.1f}% net inflow</div>
        </div>
        """, unsafe_allow_html=True)
