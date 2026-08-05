"""
Q-RiskNet India — Root Streamlit Application Entrypoint
Copyright (c) 2026 Bibek Rout
"""
import os
import sys

# Ensure root directory is on python path for imports across modules
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dashboard.app import main

if __name__ == "__main__":
    main()
else:
    main()
