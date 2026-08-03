# Data Source Review: NSE/BSE Financial Market Data Ingestion

**Document Version**: 1.0.0  
**Project**: Q-RiskNet India  
**Date**: August 2026  

---

## 1. Overview of Primary Data Provider: Yahoo Finance API (`yfinance`)

The **Q-RiskNet India** platform utilizes Yahoo Finance as its primary data provider for historical prices, volumes, and market index series across National Stock Exchange (NSE) and Bombay Stock Exchange (BSE) indices.

---

## 2. Sectoral Index Ticker Mapping Architecture

The National Stock Exchange of India (NSE) sector indices are identified by specific Yahoo Finance ticker symbols:

| Sector Name | Ticker Symbol | Market Representation | Supported Asset Class |
| :--- | :--- | :--- | :--- |
| **Nifty 50** | `^NSEI` | Indian Benchmark Equity Index | Broad Market |
| **Nifty Bank** | `^NSEBANK` | Banking & Financial Institutions | Sector Index |
| **Nifty IT** | `^CNXIT` | Information Technology Companies | Sector Index |
| **Nifty Pharma** | `^CNXPHARMA` | Pharmaceuticals & Healthcare | Sector Index |
| **Nifty Auto** | `^CNXAUTO` | Automobile & Component Manufacturers | Sector Index |
| **Nifty FMCG** | `^CNXFMCG` | Fast-Moving Consumer Goods | Sector Index |
| **Nifty Metal** | `^CNXMETAL` | Metals, Mining & Materials | Sector Index |
| **Nifty Energy** | `^CNXENERGY` | Oil, Gas, Power & Renewable Energy | Sector Index |
| **Nifty Realty** | `^CNXREALTY` | Real Estate & Construction | Sector Index |
| **Nifty Fin Service**| `NIFTY_FIN_SERVICE.NS` | Broader Financial Services | Sector Index |
| **BSE Sensex** | `^BSESN` | BSE Top 30 Benchmark | Broad Market |

---

## 3. Date Range & Frequency Capabilities

* **Frequency Supported**: Daily (`1d`), Weekly (`1wk`), Monthly (`1mo`). Daily frequency is selected as the standard for financial volatility spillover research.
* **Historical Date Range**: Full history available from ~2007 to Present (15+ years of continuous daily trading data).
* **Data Fields Extracted**: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
* **Standard Return Calculation Field**: `Close` / `Adj Close` used for log return calculation ($r_t = \ln(P_t / P_{t-1}) \times 100$).

---

## 4. Known Technical Limitations & Mitigation Strategies

1. **Timezone Cache Lock (SQLite OperationalError)**:
   * *Limitation*: `yfinance` uses SQLite for local timezone caching. In multi-threaded Streamlit executions, this can throw `OperationalError: database is locked`.
   * *Mitigation*: Programmatically set `yf.set_tz_cache_location(os.path.join(tempfile.gettempdir(), "yf_cache"))` in `src/data/download.py`.

2. **Non-Trading Day & Holiday Discrepancies**:
   * *Limitation*: NSE holidays (e.g. Diwali Muhurat Trading, Independence Day) create non-aligned missing dates across indices.
   * *Mitigation*: The data validation engine performs explicit forward-fill (`ffill`), backward-fill (`bfill`), and inner calendar join to align all sector timestamps.

3. **Corporate Actions (Splits & Dividends)**:
   * *Limitation*: Index series are value-weighted calculations. Individual stock splits do not distort index prices, but `Adj Close` accounts for dividend adjustments.
   * *Mitigation*: Default to split-and-dividend adjusted series for econometric log returns calculations.

4. **Rate Limits & Connection Failures**:
   * *Limitation*: Bulk requests to Yahoo Finance can be throttled or fail due to network timeouts.
   * *Mitigation*: Built-in exponential backoff retries in `src/data/download.py`.
