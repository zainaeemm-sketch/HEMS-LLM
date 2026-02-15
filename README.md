# HEMS Project (Form + SQLite + PV Forecasting)

A Home Energy Management System (HEMS) application built with **Streamlit**, featuring:
- Form-based configuration
- Persistent storage using **SQLite**
- **PV forecasting** using historical PVGIS data and real-time OpenWeather data
- Cost and comfort **optimization**
- An **LLM assistant** (VectorEngine / GPT‑5‑mini) for explanations only

---

## ⚙️ Requirements

- **Python version (required):**
  ```
  Python 3.13.3
  ```

- All required Python libraries are already frozen in:
  ```
  requirements.txt
  ```

---

## 🐍 Virtual Environment Setup

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the **project root directory**:

```env
VECTORENGINE_API_KEY=your_vectorengine_key_here
OPENWEATHER_API_KEY=your_openweather_key_here
```

Notes:
- `VECTORENGINE_API_KEY` is used only for the **Assistant** page
- `OPENWEATHER_API_KEY` is used for **real-time PV forecasting**
- API keys are **never stored** in SQLite

---

## ▶️ Run the Application

From the project root directory:

```bash
streamlit run app/main.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 🧭 Application Pages Overview

### 🧩 Setup (Form)
- Collects user inputs using a Streamlit form
- City, date range, appliances, comfort bounds, PV capacity
- Validated with a schema and stored in **SQLite**
- Replaces all LLM-based configuration extraction

---

### 🔮 Forecast PV

Two forecasting modes are supported:

**1. PVGIS (Historical CSV)**
- Upload PVGIS hourly CSV files
- Uses a **multivariate LSTM**
- Best for historical and baseline PV behavior

**2. OpenWeather (Real-time)**
- Uses OpenWeather API 
- Forecasts next 24h PV generation using:
  - solar geometry
  - cloud cover
  - temperature derating
- Results are saved automatically

---

### ⚙️ Optimize Schedule
- Minimizes electricity cost subject to:
  - Time-of-use pricing
  - PV availability
  - Appliance constraints
  - Thermal comfort bounds
- Supports:
  - grid import vs export
  - feed-in tariff
  - external temperature from weather data

---

### 📊 View Results
- Visualizations for:
  - appliance schedules
  - indoor temperature profile
  - grid import/export
  - PV generation
- Displays total cost and key metrics

---

### 🤖 Assistant
- Powered by **VectorEngine GPT‑5‑mini**
- Explains forecasts and optimization results
- Provides insights and recommendations
- Read-only: never modifies configuration or results

---

## 🗂️ Data & Storage

- All runs are stored in:
  ```
  hems.db (SQLite)
  ```
- Each setup, forecast, and optimization run is versioned


---
