# app/forecasting/multivar_lstm_forecast.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class MultiVarLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

def _create_sequences_1step(X: np.ndarray, y: np.ndarray, seq_len: int = 24):
    X_seq, y_seq = [], []
    N = len(X)
    max_start = N - seq_len - 1
    for start in range(max_start + 1):
        end = start + seq_len
        X_seq.append(X[start:end])
        y_seq.append(y[end])
    return np.array(X_seq), np.array(y_seq)

def forecast_pv_multivariate(
    df_full: pd.DataFrame,
    num_slots: int = 24,
    capacity: float = 1.0,
    seq_len: int = 24,
    max_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 20,
):
    if "pv_output_kwh" not in df_full.columns:
        raise ValueError("DataFrame must contain 'pv_output_kwh'")

    feature_cols = ["G_i", "H_sun", "T2m", "WS10m", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    for c in feature_cols:
        if c not in df_full.columns:
            raise ValueError(f"Missing feature column '{c}'")

    X = df_full[feature_cols].values.astype(np.float32)
    y = df_full["pv_output_kwh"].values.reshape(-1, 1).astype(np.float32)

    if len(X) < seq_len + num_slots:
        raise ValueError(f"Not enough data points ({len(X)}) for seq_len={seq_len}, num_slots={num_slots}")

    feat_scaler = MinMaxScaler()
    targ_scaler = MinMaxScaler()

    Xs = feat_scaler.fit_transform(X)
    ys = targ_scaler.fit_transform(y)

    X_seq, y_seq = _create_sequences_1step(Xs, ys, seq_len=seq_len)
    if len(X_seq) < 10:
        raise ValueError("Too few training samples")

    split = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiVarLSTM(input_size=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        Xb = X_train_t[perm]
        yb = y_train_t[perm]

        for i in range(0, len(Xb), batch_size):
            xb = Xb[i : i + batch_size]
            tb = yb[i : i + batch_size]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, tb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = model(X_val_t)
            vloss = loss_fn(vp, y_val_t).item()

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    metrics = {}
    try:
        model.eval()
        with torch.no_grad():
            val_preds_scaled = model(X_val_t).cpu().numpy()
        y_true = targ_scaler.inverse_transform(y_val_t.cpu().numpy()).reshape(-1)
        y_pred = targ_scaler.inverse_transform(val_preds_scaled).reshape(-1)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        rng = float(np.max(y_true) - np.min(y_true)) + 1e-9
        peak = float(np.max(y_true)) + 1e-9

        metrics = {
            "mae_kwh": float(mae),
            "rmse_kwh": float(rmse),
            "nmae_range": float(mae / rng),
            "nrmse_range": float(rmse / rng),
            "nmae_peak": float(mae / peak),
            "nrmse_peak": float(rmse / peak),
            "r2": float(r2),
            "val_samples": int(len(y_true)),
        }
    except Exception as e:
        metrics = {"error": str(e)}

    # multi-step inference (feature hold except cyclical time updates)
    X_hist = Xs[-seq_len:].copy()
    preds_kwh = []

    model.eval()
    with torch.no_grad():
        for step in range(num_slots):
            x_in = torch.tensor(X_hist[np.newaxis, :, :], dtype=torch.float32).to(device)
            y_next_scaled = model(x_in).cpu().numpy()
            y_next_kwh = targ_scaler.inverse_transform(y_next_scaled)[0, 0]
            preds_kwh.append(y_next_kwh)

            if isinstance(df_full.index, pd.DatetimeIndex):
                last_time = df_full.index[-1] + pd.Timedelta(hours=step + 1)
                hour = last_time.hour
                doy = last_time.dayofyear
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                doy_sin = np.sin(2 * np.pi * doy / 365)
                doy_cos = np.cos(2 * np.pi * doy / 365)
            else:
                hour_sin = X_hist[-1, feature_cols.index("hour_sin")]
                hour_cos = X_hist[-1, feature_cols.index("hour_cos")]
                doy_sin = X_hist[-1, feature_cols.index("doy_sin")]
                doy_cos = X_hist[-1, feature_cols.index("doy_cos")]

            last_feat = X_hist[-1].copy()
            last_feat[feature_cols.index("hour_sin")] = hour_sin
            last_feat[feature_cols.index("hour_cos")] = hour_cos
            last_feat[feature_cols.index("doy_sin")] = doy_sin
            last_feat[feature_cols.index("doy_cos")] = doy_cos

            X_hist = np.vstack([X_hist[1:], last_feat])

    preds_kwh = np.clip(np.array(preds_kwh, dtype=np.float32) * float(capacity), 0.0, None)
    eps = max(0.01, 0.02 * float(preds_kwh.max() if preds_kwh.size else 1.0))
    preds_kwh[preds_kwh < eps] = 0.0

    return preds_kwh.tolist(), metrics
