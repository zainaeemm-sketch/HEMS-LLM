# app/forecasting/lstm_forecast.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

def forecast_pv(historical_data: pd.DataFrame, num_slots: int = 24, capacity: float = 1.0):
    if "pv_output_kwh" not in historical_data.columns:
        raise ValueError("Missing 'pv_output_kwh' column")

    y = historical_data["pv_output_kwh"].values.reshape(-1, 1)
    if len(y) < 48:
        raise ValueError("Need at least 48 samples")

    scaler = MinMaxScaler()
    y_s = scaler.fit_transform(y)

    seq_len = min(24, len(y_s) // 2)
    X, T = [], []
    for i in range(len(y_s) - seq_len):
        X.append(y_s[i : i + seq_len])
        T.append(y_s[i + seq_len])
    X = np.array(X, dtype=np.float32)      # (N, seq, 1)
    T = np.array(T, dtype=np.float32)      # (N, 1)

    X_t = torch.tensor(X)
    T_t = torch.tensor(T)

    model = LSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(25):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, T_t)
        loss.backward()
        opt.step()

    last = y_s[-seq_len:].astype(np.float32).reshape(1, seq_len, 1)
    cur = torch.tensor(last)

    preds = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_slots):
            p = model(cur).cpu().numpy()[0, 0]
            preds.append(p)
            nxt = np.append(cur.cpu().numpy()[0, 1:, 0], p).astype(np.float32).reshape(1, seq_len, 1)
            cur = torch.tensor(nxt)

    preds_kwh = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    preds_kwh = np.clip(preds_kwh * float(capacity), 0.0, None)
    return preds_kwh.tolist()
