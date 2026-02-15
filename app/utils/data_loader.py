# app/utils/data_loader.py
import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_PV_PATH = os.path.join(BASE_DIR, "data", "synthetic_pv_historical.csv")
TOU_PATH = os.path.join(BASE_DIR, "data", "tou_prices.json")

def load_pv_data(path: str = DEFAULT_PV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing PV dataset at {path}. "
            "Expected columns: timestamp,pv_output_kwh"
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "pv_output_kwh" not in df.columns:
        raise ValueError("Missing required column 'pv_output_kwh'")
    return df

def load_tou_prices(path: str = TOU_PATH) -> list[float]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing TOU price file at {path}")
    with open(path) as f:
        data = json.load(f)
    return [float(data[str(h)]) for h in range(24)]
