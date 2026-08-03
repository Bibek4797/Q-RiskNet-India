"""
Unit tests for Q-RiskNet India Dashboard Modular Structure
"""
import pytest


def test_dashboard_components_imports():
    """Verify that all dashboard components and pages can be imported without errors."""
    import dashboard.components.sidebar as sidebar
    import dashboard.components.kpi_cards as kpi_cards
    import dashboard.components.charts as charts
    import dashboard.components.tables as tables
    import dashboard.components.exports as exports
    import dashboard.components.status as status

    assert hasattr(sidebar, "render_sidebar")
    assert hasattr(kpi_cards, "render_kpi_cards")
    assert hasattr(charts, "render_prices_chart")
    assert hasattr(tables, "render_descriptive_table")
    assert hasattr(exports, "download_csv")
    assert hasattr(status, "safe_execute")


def test_dashboard_pages_imports():
    """Verify that all modular dashboard page modules can be imported and have render_page."""
    import dashboard.pages.home as home
    import dashboard.pages.about as about
    import dashboard.pages.data_center as data_center
    import dashboard.pages.diagnostics as diagnostics
    import dashboard.pages.volatility as volatility
    import dashboard.pages.qvar_analysis as qvar_analysis
    import dashboard.pages.connectedness as connectedness
    import dashboard.pages.network as network
    import dashboard.pages.forecasting as forecasting
    import dashboard.pages.validation as validation
    import dashboard.pages.reports as reports

    assert hasattr(home, "render_page")
    assert hasattr(about, "render_page")
    assert hasattr(data_center, "render_page")
    assert hasattr(diagnostics, "render_page")
    assert hasattr(volatility, "render_page")
    assert hasattr(qvar_analysis, "render_page")
    assert hasattr(connectedness, "render_page")
    assert hasattr(network, "render_page")
    assert hasattr(forecasting, "render_page")
    assert hasattr(validation, "render_page")
    assert hasattr(reports, "render_page")


def test_dashboard_utils_theme():
    """Verify theme utilities import and setup."""
    import dashboard.utils.theme as theme
    assert hasattr(theme, "setup_page_config")
    assert hasattr(theme, "inject_custom_css")
    assert hasattr(theme, "render_header")
