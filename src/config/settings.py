import os
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")

def load_config(path=CONFIG_PATH):
    """
    Loads YAML configuration dictionary.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

# Loaded configuration object
CFG = load_config()

# Fallback or exported constants
TICKER_MAP = CFG.get("data", {}).get("tickers", {
    "Nifty 50": "^NSEI",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty Auto": "^CNXAUTO",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Energy": "^CNXENERGY",
    "Nifty Realty": "^CNXREALTY",
    "Nifty Financial Services": "NIFTY_FIN_SERVICE.NS"
})

PATHS = CFG.get("paths", {
    "raw_data": "data/raw",
    "processed_data": "data/processed",
    "external_data": "data/external",
    "models_dir": "models",
    "outputs_dir": "outputs",
    "logs_dir": "logs"
})

DASHBOARD_CFG = CFG.get("dashboard", {})
MODEL_CFG = CFG.get("models", {})
GIRF_CFG = CFG.get("girf", {})
