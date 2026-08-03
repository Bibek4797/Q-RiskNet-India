"""
Q-RiskNet India — Status Indicators & Error Handling
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st


def safe_execute(operation_name, func, *args, **kwargs):
    """Execute a function with graceful Streamlit error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"❌ **{operation_name}** encountered an error: {str(e)}")
        st.info("💡 Try adjusting your parameters, widening the date range, or selecting different sectors.")
        return None


def render_empty_state(message, icon="ℹ️"):
    """Show a friendly message when data is not yet available."""
    st.info(f"{icon} {message}")


def render_phase_badge(phase_num, phase_name, status="complete"):
    """Render a phase completion badge."""
    icons = {"complete": "✅", "in_progress": "🔄", "pending": "⏳"}
    icon = icons.get(status, "❓")
    st.markdown(f"{icon} **Phase {phase_num}** — {phase_name}")
