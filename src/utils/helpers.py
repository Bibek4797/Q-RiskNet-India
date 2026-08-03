import pandas as pd
import numpy as np

def format_percentage(val):
    if isinstance(val, (float, int)):
        return f"{val:.2f}%"
    return str(val)
