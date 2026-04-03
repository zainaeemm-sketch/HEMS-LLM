# app/utils/pvgis_preprocess.py
import pandas as pd
import numpy as np
from pathlib import Path

def load_pvgis_for_lstm(csv_path: str | Path):
    csv_path = Path(csv_path)

    with csv_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.lower().startswith("time,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find 'time,' header line in PVGIS file.")

    df = pd.read_csv(csv_path, skiprows=header_idx)
    df = df[df["time"].astype(str).str.match(r"^\d{8}:\d{4}$")]
    if df.empty:
        raise ValueError("After filtering, no valid data rows remain. Check the CSV format.")

    df["time"] = df["time"].astype(str).str.replace(":", "", regex=False)
    df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M", errors="raise")
    df = df.set_index("datetime").sort_index()

    df = df.rename(columns={"P": "P_w", "G(i)": "G_i"})

    if "P_w" not in df.columns:
        raise ValueError("Expected column 'P' (PV power) in PVGIS file.")
    if "G_i" not in df.columns:
        raise ValueError("Expected column 'G(i)' (irradiance on tilted plane) in PVGIS file.")
    if "H_sun" not in df.columns:
        raise ValueError("Expected column 'H_sun' in PVGIS file.")
    if "T2m" not in df.columns:
        raise ValueError("Expected column 'T2m' in PVGIS file.")
    if "WS10m" not in df.columns:
        raise ValueError("Expected column 'WS10m' in PVGIS file.")

    numeric_cols = ["P_w", "G_i", "H_sun", "T2m", "WS10m", "Int"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["P_w", "G_i", "H_sun", "T2m", "WS10m"])

    df["pv_output_kwh"] = df["P_w"] / 1000.0

    df["hour"] = df.index.hour
    df["doy"] = df.index.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

    feature_cols = ["G_i", "H_sun", "T2m", "WS10m", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    X = df[feature_cols].copy()
    y = df["pv_output_kwh"].copy()
    df_full = df.drop(columns=[c for c in ["Int", "time", "hour", "doy"] if c in df.columns])

    return df_full, X, y
