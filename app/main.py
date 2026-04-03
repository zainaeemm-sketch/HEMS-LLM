import os
import json
import altair as alt
import streamlit as st
import pandas as pd
from datetime import date
from pathlib import Path

from utils.env import load_env, get_env
from utils.schema_validator import HEMSParameters, ValidationError
from utils.parameter_store import (
    init_db,
    load_latest_parameters,
    overwrite_latest_parameters,
)
from utils.constants import TIME_SLOTS, TOU_PRICES
from utils.data_loader import load_tou_prices
from utils.pvgis_preprocess import load_pvgis_for_lstm
from forecasting.multivar_lstm_forecast import forecast_pv_multivariate
from forecasting.openweather_pv_forecast import (
    OpenWeatherError,
    geocode_city,
    fetch_forecast25_hourly,
    estimate_pv_kwh_24h_from_weather,
)
from optimization.hems_optimizer import optimize_schedule
from utils.llm_agent import chat_with_vectorengine

load_env()

st.set_page_config(page_title="LLM-HEMS (Form + SQLite)", layout="wide")


# ---------- Helpers ----------
def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_appliance_rows(appliances: list[dict]) -> pd.DataFrame:
    if not appliances:
        appliances = []
    return pd.DataFrame(
        appliances,
        columns=["name", "start_time", "end_time", "can_shift"],
    ).fillna("")


def _df_to_appliances(df: pd.DataFrame) -> list[dict]:
    rows = []
    if df is None or df.empty:
        return rows

    for _, r in df.iterrows():
        name = str(r.get("name", "")).strip()
        if not name:
            continue

        start_time = str(r.get("start_time", "")).strip() or None
        end_time = str(r.get("end_time", "")).strip() or None

        # Normalize midnight start to 00:00
        if start_time == "24:00":
            start_time = "00:00"

        can_shift = r.get("can_shift", False)
        if isinstance(can_shift, str):
            can_shift = can_shift.lower() in ("true", "1", "yes", "y")

        rows.append(
            {
                "name": name,
                "start_time": start_time,
                "end_time": end_time,
                "can_shift": bool(can_shift),
            }
        )

    return rows


def style_chart(chart):
    return (
        chart.properties(
            padding={"left": 8, "right": 8, "top": 10, "bottom": 8}
        )
        .configure_view(strokeWidth=0)
        .configure_axis(
            grid=True,
            gridColor="#e5e7eb",
            gridOpacity=0.7,
            domainColor="#cbd5e1",
            tickColor="#cbd5e1",
            labelColor="#334155",
            titleColor="#0f172a",
            labelFontSize=12,
            titleFontSize=13,
        )
        .configure_title(
            anchor="start",
            color="#0f172a",
            fontSize=16,
            fontWeight="bold",
        )
        .configure_legend(
            orient="top",
            labelColor="#334155",
            titleColor="#0f172a",
            cornerRadius=8,
            padding=8,
        )
    )


def _hour_labels(n=24):
    return [f"{h:02d}:00" for h in range(n)]


def build_pv_forecast_chart(pv_forecast: list[float]):
    pv = (pv_forecast or [])[:24]
    time_labels = _hour_labels(len(pv))
    df = pd.DataFrame(
        {
            "Time": time_labels,
            "PV Forecast (kWh)": [round(float(x or 0.0), 3) for x in pv],
        }
    )

    chart = (
        alt.Chart(df, title="24-Hour PV Forecast")
        .mark_area(line=True, opacity=0.22)
        .encode(
            x=alt.X("Time:N", sort=time_labels, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("PV Forecast (kWh):Q", title="PV Forecast (kWh)"),
            tooltip=[
                alt.Tooltip("Time:N"),
                alt.Tooltip("PV Forecast (kWh):Q", format=".3f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    return style_chart(chart)


def build_schedule_heatmap(schedule: dict):
    time_labels = _hour_labels()
    rows = []

    for appliance, values in (schedule or {}).items():
        vals = (values or [])[:24]
        for h, raw in enumerate(vals):
            level = float(raw or 0.0)
            rows.append(
                {
                    "Appliance": appliance,
                    "Time": time_labels[h],
                    "Hour": h,
                    "Level": round(level, 2),
                    "Running": "On" if level > 0.01 else "Off",
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        return alt.Chart(
            pd.DataFrame({"Message": ["No schedule data available"]})
        ).mark_text(size=14).encode(text="Message:N")

    max_level = max(df["Level"].max(), 1.0)

    heatmap = (
        alt.Chart(df, title="Optimized Appliance Operating Schedule")
        .mark_rect(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X(
                "Time:N",
                sort=time_labels,
                title=None,
                axis=alt.Axis(labelAngle=0, labelOverlap=False),
            ),
            y=alt.Y("Appliance:N", title=None, sort=list((schedule or {}).keys())),
            color=alt.Color(
                "Level:Q",
                title="Usage level",
                scale=alt.Scale(
                    domain=[0, max_level],
                    range=["#eff6ff", "#bfdbfe", "#60a5fa", "#2563eb", "#1d4ed8"],
                ),
            ),
            opacity=alt.condition(
                alt.datum.Level > 0.01,
                alt.value(1.0),
                alt.value(0.35),
            ),
            tooltip=[
                alt.Tooltip("Appliance:N"),
                alt.Tooltip("Time:N"),
                alt.Tooltip("Level:Q", format=".2f"),
                alt.Tooltip("Running:N"),
            ],
        )
        .properties(height=max(260, 42 * len((schedule or {}).keys())))
    )

    return style_chart(heatmap)


def build_temperature_chart(results: dict, params: dict):
    time_labels = _hour_labels()
    indoor = (results.get("temps") or [0] * 24)[:24]
    outdoor = (results.get("T_ext") or [None] * 24)[:24]
    tmin = float(params.get("Tmin", 18.0))
    tmax = float(params.get("Tmax", 22.0))
    target = round((tmin + tmax) / 2, 2)

    df = pd.DataFrame(
        {
            "Time": time_labels,
            "Indoor": [round(float(x or 0.0), 2) for x in indoor],
            "Outdoor": [None if x is None else round(float(x), 2) for x in outdoor],
            "Tmin": [tmin] * 24,
            "Tmax": [tmax] * 24,
            "Target": [target] * 24,
        }
    )
    df["Violation"] = (df["Indoor"] < df["Tmin"]) | (df["Indoor"] > df["Tmax"])

    comfort_band = (
        alt.Chart(df)
        .mark_area(opacity=0.18, color="#10b981")
        .encode(
            x=alt.X("Time:N", sort=time_labels, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Tmin:Q", title="Temperature (°C)"),
            y2="Tmax:Q",
            tooltip=[
                alt.Tooltip("Time:N"),
                alt.Tooltip("Tmin:Q", title="Comfort min", format=".1f"),
                alt.Tooltip("Tmax:Q", title="Comfort max", format=".1f"),
            ],
        )
    )

    target_line = (
        alt.Chart(df)
        .mark_line(strokeDash=[5, 5], strokeWidth=1.8, color="#64748b")
        .encode(
            x=alt.X("Time:N", sort=time_labels, title=None),
            y=alt.Y("Target:Q", title="Temperature (°C)"),
            tooltip=[alt.Tooltip("Target:Q", title="Target midpoint", format=".1f")],
        )
    )

    indoor_line = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=3, color="#2563eb")
        .encode(
            x=alt.X("Time:N", sort=time_labels, title=None),
            y=alt.Y("Indoor:Q", title="Temperature (°C)"),
            tooltip=[
                alt.Tooltip("Time:N"),
                alt.Tooltip("Indoor:Q", title="Indoor temp", format=".2f"),
            ],
        )
    )

    layers = [comfort_band, target_line, indoor_line]

    if df["Outdoor"].notna().any():
        outdoor_line = (
            alt.Chart(df)
            .mark_line(strokeDash=[6, 4], strokeWidth=2, color="#f59e0b")
            .encode(
                x=alt.X("Time:N", sort=time_labels, title=None),
                y=alt.Y("Outdoor:Q", title="Temperature (°C)"),
                tooltip=[
                    alt.Tooltip("Time:N"),
                    alt.Tooltip("Outdoor:Q", title="Outdoor temp", format=".2f"),
                ],
            )
        )
        layers.append(outdoor_line)

    if df["Violation"].any():
        violations = (
            alt.Chart(df[df["Violation"]])
            .mark_point(size=100, filled=True, color="#dc2626")
            .encode(
                x=alt.X("Time:N", sort=time_labels),
                y=alt.Y("Indoor:Q"),
                tooltip=[
                    alt.Tooltip("Time:N"),
                    alt.Tooltip("Indoor:Q", title="Indoor temp", format=".2f"),
                ],
            )
        )
        layers.append(violations)

    return style_chart(
        alt.layer(
            *layers,
            title="Indoor Temperature, Comfort Band, and Outdoor Temperature",
        )
        .properties(height=360)
        .interactive()
    )


def build_energy_panel(results: dict, params: dict, tou_prices: list[float]):
    time_labels = _hour_labels()

    grid_import = [
        round(float(x or 0.0), 3)
        for x in (results.get("grid_import") or [0] * 24)[:24]
    ]
    grid_export = [
        round(float(x or 0.0), 3)
        for x in (results.get("grid_export") or [0] * 24)[:24]
    ]
    pv = [
        round(float(x or 0.0), 3)
        for x in (params.get("pv_forecast") or [0] * 24)[:24]
    ]
    tou = [round(float(x or 0.0), 3) for x in (tou_prices or [0] * 24)[:24]]

    flow_rows = []
    for i, t in enumerate(time_labels):
        flow_rows.append({"Time": t, "Series": "Grid Import", "Value": grid_import[i]})
        flow_rows.append({"Time": t, "Series": "Grid Export", "Value": -grid_export[i]})

    flow_df = pd.DataFrame(flow_rows)
    pv_df = pd.DataFrame({"Time": time_labels, "PV": pv})
    tou_df = pd.DataFrame({"Time": time_labels, "TOU Price": tou})

    zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="#94a3b8", strokeDash=[4, 4])
        .encode(y="y:Q")
    )

    bars = (
        alt.Chart(flow_df)
        .mark_bar(size=18)
        .encode(
            x=alt.X("Time:N", sort=time_labels, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Value:Q", title="Energy / power per hour"),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["Grid Import", "Grid Export"],
                    range=["#2563eb", "#f97316"],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("Time:N"),
                alt.Tooltip("Series:N"),
                alt.Tooltip("Value:Q", format=".3f"),
            ],
        )
    )

    pv_area = (
        alt.Chart(pv_df)
        .mark_area(color="#22c55e", opacity=0.18)
        .encode(
            x=alt.X("Time:N", sort=time_labels, title=None),
            y=alt.Y("PV:Q", title="Energy / power per hour"),
            tooltip=[
                alt.Tooltip("Time:N"),
                alt.Tooltip("PV:Q", title="PV forecast", format=".3f"),
            ],
        )
    )

    pv_line = (
        alt.Chart(pv_df)
        .mark_line(color="#16a34a", strokeWidth=3, point=True)
        .encode(
            x=alt.X("Time:N", sort=time_labels, title=None),
            y=alt.Y("PV:Q", title="Energy / power per hour"),
            tooltip=[
                alt.Tooltip("Time:N"),
                alt.Tooltip("PV:Q", title="PV forecast", format=".3f"),
            ],
        )
    )

    energy_chart = (
        alt.layer(
            zero_rule,
            bars,
            pv_area,
            pv_line,
            title="Grid Import / Export with PV Forecast",
        )
        .properties(height=360)
        .interactive()
    )

    price_chart = (
        alt.Chart(tou_df, title="Time-of-Use Tariff")
        .mark_line(point=True, strokeWidth=2.5, color="#7c3aed")
        .encode(
            x=alt.X("Time:N", sort=time_labels, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("TOU Price:Q", title="Tariff ($/kWh)"),
            tooltip=[
                alt.Tooltip("Time:N"),
                alt.Tooltip("TOU Price:Q", format=".3f"),
            ],
        )
        .properties(height=150)
        .interactive()
    )

    combined = alt.vconcat(energy_chart, price_chart, spacing=12)
    return style_chart(combined)


def _summarize_for_assistant(params: dict) -> dict:
    """
    Compact payload for the assistant. Avoid secrets.
    """
    pv = params.get("pv_forecast") or []
    pv_24 = pv[:24] if isinstance(pv, list) else []
    metrics = params.get("forecast_metrics") or {}
    source = params.get("forecast_source")

    opt = params.get("optimization_results") or {}
    opt_summary = {}
    if isinstance(opt, dict) and opt:
        opt_summary = {
            "cost": opt.get("cost"),
            "grid_import": opt.get("grid_import"),
            "grid_export": opt.get("grid_export"),
            "temps": opt.get("temps"),
            "schedule": opt.get("schedule"),
        }

    weather = params.get("weather_hourly") or None
    weather_summary = None
    if isinstance(weather, dict):
        hourly = weather.get("hourly")
        if isinstance(hourly, list):
            weather_summary = {
                "resolved": weather.get("resolved"),
                "lat": weather.get("lat"),
                "lon": weather.get("lon"),
                "hourly_first24": hourly[:24],
            }

    return {
        "setup": {
            "city": params.get("city"),
            "user_type": params.get("user_type"),
            "start_date": params.get("start_date"),
            "end_date": params.get("end_date"),
            "Tmin": params.get("Tmin"),
            "Tmax": params.get("Tmax"),
            "max_power": params.get("max_power"),
            "solar_pv_capacity_kw": params.get("solar_pv_capacity"),
            "appliances": params.get("appliances") or [],
            "do_not_disturb": params.get("do_not_disturb") or [],
        },
        "forecast": {
            "source": source,
            "pv_forecast_24h_kwh": pv_24,
            "metrics": metrics,
            "weather": weather_summary,
        },
        "optimization": opt_summary,
    }


def _is_context_msg(m: dict) -> bool:
    return m.get("role") == "system" and str(m.get("content", "")).startswith("SYSTEM_CONTEXT_JSON:")


def _latest_context_message(params: dict) -> dict:
    context = _summarize_for_assistant(params)
    try:
        if isinstance(context.get("forecast", {}).get("pv_forecast_24h_kwh"), list):
            context["forecast"]["pv_forecast_24h_kwh"] = [
                round(float(x), 3)
                for x in context["forecast"]["pv_forecast_24h_kwh"][:24]
            ]
    except Exception:
        pass

    return {
        "role": "system",
        "content": "SYSTEM_CONTEXT_JSON:\n" + json.dumps(context, indent=2),
    }


def _assistant_system_prompt() -> str:
    return (
        "You are a Home Energy Management System assistant.\n"
        "\n"
        "You will receive system context in a message that starts with 'SYSTEM_CONTEXT_JSON:'.\n"
        "\n"
        "Rules:\n"
        "- NEVER repeat raw JSON or large arrays.\n"
        "- Summarize insights in plain English.\n"
        "- Use short sections + bullet points.\n"
        "- Explain WHY the forecast/optimization looks that way.\n"
        "- Provide actionable suggestions to reduce cost and keep comfort.\n"
        "- If data is missing, say so briefly.\n"
        "- Never ask for or reveal API keys.\n"
    )


# ---------- Init DB ----------
init_db()

# ---------- Sidebar ----------
image_path = Path(__file__).parent / "assets" / "smart-home-icon.png"
if image_path.exists():
    st.sidebar.image(str(image_path), width=80)
else:
    st.sidebar.write("⚡ HEMS Dashboard")

st.sidebar.markdown("## ⚡ HEMS Dashboard")

pages = {
    "🧩 Setup (Form)": "setup",
    "🔮 Forecast PV": "forecast",
    "⚙️ Optimize Schedule": "optimize",
    "📊 View Results": "results",
    "🤖 Assistant": "assistant",
}

if "active_page" not in st.session_state:
    st.session_state.active_page = "setup"

latest = load_latest_parameters()
has_params = bool(latest)

for name, key in pages.items():
    disabled = (key != "setup" and not has_params)
    if st.sidebar.button(name, width="stretch", disabled=disabled):
        st.session_state.active_page = key

if not has_params:
    st.sidebar.warning("⚠️ Please complete Setup first.")
else:
    st.sidebar.success("✅ Setup found in SQLite.")

st.sidebar.markdown("---")
st.sidebar.caption("Two PV modes: PVGIS CSV (historical) + OpenWeather API (real-time).")

page = st.session_state.active_page

# ============================================================
# 🧩 SETUP PAGE (FORM → SQLITE)
# ============================================================
if page == "setup":
    st.header("🧩 Setup your Home Energy System (Form → SQLite)")

    params = load_latest_parameters() or {}
    default_city = params.get("city", "")
    default_user_type = params.get("user_type", "residential")

    default_start = params.get("start_date")
    default_end = params.get("end_date")

    try:
        default_start = date.fromisoformat(default_start) if default_start else date.today()
    except Exception:
        default_start = date.today()
    try:
        default_end = date.fromisoformat(default_end) if default_end else date.today()
    except Exception:
        default_end = date.today()

    default_Tmin = safe_float(params.get("Tmin"), 18.0)
    default_Tmax = safe_float(params.get("Tmax"), 22.0)
    default_max_power = safe_float(params.get("max_power"), 5.0)
    default_pv_cap = safe_float(params.get("solar_pv_capacity"), 0.0)

    raw_dnd = params.get("do_not_disturb", [])
    if isinstance(raw_dnd, list):
        dnd_text = ", ".join([str(x) for x in raw_dnd])
    else:
        dnd_text = str(raw_dnd or "")

    default_apps_df = _as_appliance_rows(params.get("appliances", []))

    with st.form("setup_form", clear_on_submit=False):
        colA, colB = st.columns(2)

        with colA:
            city = st.text_input("City", value=default_city)
            user_type = st.selectbox(
                "User Type",
                ["residential", "industrial"],
                index=["residential", "industrial"].index(default_user_type)
                if default_user_type in ("residential", "industrial")
                else 0,
            )
            start_date = st.date_input("Start Date", value=default_start)
            end_date = st.date_input("End Date", value=default_end)

        with colB:
            solar_pv_capacity = st.number_input("Solar PV Capacity (kW)", 0.0, 10000.0, default_pv_cap)
            Tmin = st.number_input("Tmin (°C)", 0.0, 60.0, default_Tmin)
            Tmax = st.number_input("Tmax (°C)", 0.0, 60.0, default_Tmax)
            max_power = st.number_input("Max Power (kW)", 0.0, 10000.0, default_max_power)

        st.subheader("⚙️ Appliances")
        st.caption(
            "Add/edit appliances. Use HH:MM 24h format or leave empty. "
            "Midnight start should be 00:00; end time may be 24:00."
        )
        edited_df = st.data_editor(
            default_apps_df,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "name": st.column_config.TextColumn("Name", required=True),
                "start_time": st.column_config.TextColumn("Start (HH:MM)", required=False),
                "end_time": st.column_config.TextColumn("End (HH:MM)", required=False),
                "can_shift": st.column_config.CheckboxColumn("Can Shift?", default=False),
            },
        )

        do_not_disturb = st.text_area(
            "Do Not Disturb Periods (comma-separated, e.g., 22:00-07:00)",
            value=dnd_text,
        )

        submitted = st.form_submit_button("💾 Save to SQLite")

    if submitted:
        dnd_list = [d.strip() for d in (do_not_disturb or "").split(",") if d.strip()]
        appliances = _df_to_appliances(edited_df)

        new_params = {
            "city": city.strip(),
            "user_type": user_type,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "appliances": appliances,
            "Tmin": float(Tmin),
            "Tmax": float(Tmax),
            "max_power": float(max_power),
            "do_not_disturb": dnd_list,
            "solar_pv_capacity": float(solar_pv_capacity),
            "pv_forecast": params.get("pv_forecast", []),
            "forecast_source": params.get("forecast_source"),
            "forecast_metrics": params.get("forecast_metrics", {}),
            "weather_hourly": params.get("weather_hourly", None),
            "optimization_results": params.get("optimization_results", None),
        }

        try:
            model = HEMSParameters.parse_obj(new_params)
            overwrite_latest_parameters(model.dict())
            st.success("✅ Saved & validated in SQLite.")
            st.json(model.dict())
        except ValidationError as e:
            st.error("❌ Validation error (schema):")
            st.code(e.json())

# ============================================================
# 🔮 FORECAST PAGE
# ============================================================
elif page == "forecast":
    params = load_latest_parameters()
    if not params:
        st.warning("⚠️ Please complete Setup first.")
    else:
        st.header("🔮 PV Forecasting (Historical PVGIS + Real-time OpenWeather)")

        mode = st.selectbox(
            "Forecast Mode",
            [
                "PVGIS (historical CSV → multivariate LSTM)",
                "OpenWeather (real-time API → 24h PV forecast)",
            ],
        )

        if mode.startswith("PVGIS"):
            st.subheader("📂 Upload PVGIS Hourly CSV")
            st.caption("Upload PVGIS hourly radiation CSV like `time,P,G(i),H_sun,T2m,WS10m,Int`.")

            uploaded_file = st.file_uploader("PVGIS CSV", type=["csv"], accept_multiple_files=False)

            if uploaded_file is None:
                st.info("⬆️ Upload a PVGIS CSV to generate a historical-model forecast.")
            else:
                data_dir = Path(__file__).parent / "data"
                data_dir.mkdir(exist_ok=True)
                pvgis_csv_path = data_dir / "uploaded_pvgis.csv"

                with open(pvgis_csv_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(f"✅ Uploaded `{uploaded_file.name}`")

                try:
                    df_full, X, y = load_pvgis_for_lstm(pvgis_csv_path)
                    st.subheader("📊 Processed sample")
                    st.dataframe(
                        df_full[["pv_output_kwh", "G_i", "H_sun", "T2m"]].head(),
                        width="stretch",
                    )

                    if st.button("🔮 Generate Forecast (LSTM)"):
                        capacity = safe_float(params.get("solar_pv_capacity"), 1.0) or 1.0
                        pv_forecast, metrics = forecast_pv_multivariate(df_full, TIME_SLOTS, capacity)

                        params["pv_forecast"] = pv_forecast
                        params["forecast_metrics"] = metrics
                        params["forecast_source"] = "pvgis_lstm"
                        overwrite_latest_parameters(params)

                        st.altair_chart(build_pv_forecast_chart(pv_forecast), use_container_width=True)
                        st.subheader("📏 Validation Metrics")
                        if metrics.get("error"):
                            st.error(metrics["error"])
                        else:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("MAE (kWh)", f"{metrics['mae_kwh']:.4f}")
                            c2.metric("RMSE (kWh)", f"{metrics['rmse_kwh']:.4f}")
                            c3.metric("R²", f"{metrics['r2']:.3f}")

                        st.success("✅ PVGIS-based forecast saved to SQLite.")

                except Exception as e:
                    st.error(f"❌ Error while loading/forecasting PVGIS: {e}")

        else:
            st.subheader("🌤️ Real-time PV Forecast via OpenWeather")

            env_ow_key = get_env("OPENWEATHER_API_KEY", "") or ""
            api_key = st.text_input("OpenWeather API Key", type="password", value=env_ow_key)

            city = st.text_input("City (for geocoding)", value=params.get("city", ""))
            capacity_kw = st.number_input(
                "PV Capacity used for forecast (kW)",
                0.0,
                10000.0,
                float(safe_float(params.get("solar_pv_capacity"), 0.0)),
            )
            performance_ratio = st.slider("Performance Ratio (PR)", 0.5, 1.0, 0.85, 0.01)

            if st.button("🔮 Fetch Weather + Forecast PV (24h)"):
                if not api_key.strip():
                    st.error("Please enter an OpenWeather API key (or set OPENWEATHER_API_KEY in .env).")
                else:
                    try:
                        lat, lon, resolved = geocode_city(city=city, api_key=api_key)

                        try:
                            hourly = fetch_forecast25_hourly(lat=lat, lon=lon, api_key=api_key)
                            source_note = "onecall_3.0"
                        except OpenWeatherError as e:
                            if "401" in str(e) or "subscription" in str(e).lower():
                                hourly = fetch_forecast25_hourly(lat=lat, lon=lon, api_key=api_key)
                                source_note = "forecast_2.5_free"
                            else:
                                raise

                        weather_hourly = [
                            {"dt": h.dt, "clouds": h.clouds, "temp_c": h.temp_c}
                            for h in hourly[:48]
                        ]

                        pv_24h = estimate_pv_kwh_24h_from_weather(
                            hourly=hourly,
                            lat=lat,
                            capacity_kw=capacity_kw,
                            tz_offset_seconds=0,
                            performance_ratio=performance_ratio,
                        )

                        params["pv_forecast"] = pv_24h
                        params["forecast_source"] = f"openweather_{source_note}"
                        params["forecast_metrics"] = {
                            "note": "clear-sky shape + clouds + temp derate + PR"
                        }
                        params["weather_hourly"] = {
                            "resolved": resolved,
                            "lat": lat,
                            "lon": lon,
                            "hourly": weather_hourly,
                        }
                        overwrite_latest_parameters(params)

                        st.success(f"✅ Forecast saved (city resolved: {resolved})")
                        st.altair_chart(build_pv_forecast_chart(pv_24h), use_container_width=True)

                    except OpenWeatherError as e:
                        st.error(f"OpenWeather error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

# ============================================================
# ⚙️ OPTIMIZE PAGE
# ============================================================
elif page == "optimize":
    params = load_latest_parameters()
    if not params:
        st.warning("⚠️ Please complete Setup first.")
    elif not params.get("pv_forecast"):
        st.warning("⚠️ Forecast PV first.")
    else:
        st.header("⚙️ Optimize Schedule")

        try:
            tou = load_tou_prices()
        except Exception:
            tou = TOU_PRICES

        feed_in_tariff = st.number_input(
            "Feed-in Tariff ($/kWh, export credit)",
            0.0,
            2.0,
            0.0,
            0.01,
        )

        T_ext = None
        wh = (params.get("weather_hourly") or {}).get("hourly")
        if isinstance(wh, list) and len(wh) >= 24:
            T_ext = [safe_float(x.get("temp_c"), 15.0) for x in wh[:24]]

        if st.button("▶ Run Optimization"):
            results = optimize_schedule(
                params=params,
                pv_forecast=params["pv_forecast"],
                tou_prices=tou,
                T_ext=T_ext,
                feed_in_tariff=feed_in_tariff,
            )
            results["T_ext"] = T_ext
            params["optimization_results"] = results
            overwrite_latest_parameters(params)
            st.success("✅ Optimization complete! Go to Results.")

# ============================================================
# 📊 RESULTS PAGE
# ============================================================
elif page == "results":
    params = load_latest_parameters()
    results = (params or {}).get("optimization_results")

    if not results:
        st.warning("⚠️ Run optimization first.")
    else:
        st.header("📊 Optimization Results")

        try:
            tou_prices = load_tou_prices()
        except Exception:
            tou_prices = TOU_PRICES

        grid_import = results.get("grid_import") or []
        grid_export = results.get("grid_export") or []
        pv_forecast = params.get("pv_forecast") or []

        total_cost = safe_float(results.get("cost"), 0.0)
        peak_import = max(grid_import) if grid_import else 0.0
        total_import = sum(grid_import) if grid_import else 0.0
        total_export = sum(grid_export) if grid_export else 0.0
        total_pv = sum(pv_forecast[:24]) if pv_forecast else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Cost", f"${total_cost:.2f}")
        c2.metric("Peak Import", f"{peak_import:.2f} kW")
        c3.metric("Import Energy", f"{total_import:.2f} kWh")
        c4.metric("Export Energy", f"{total_export:.2f} kWh")
        c5.metric("PV Forecast", f"{total_pv:.2f} kWh")

        st.markdown("### Appliance Schedule")
        st.altair_chart(
            build_schedule_heatmap(results.get("schedule", {})),
            use_container_width=True,
        )

        st.markdown("### Thermal Comfort")
        st.altair_chart(
            build_temperature_chart(results, params),
            use_container_width=True,
        )

        st.markdown("### Energy Flow and Tariff")
        st.altair_chart(
            build_energy_panel(results, params, tou_prices),
            use_container_width=True,
        )

# ============================================================
# 🤖 ASSISTANT PAGE (VectorEngine chat)
# ============================================================
elif page == "assistant":
    params = load_latest_parameters()
    st.header("🤖 Assistant")
    st.caption("Ask questions about your PV forecast and optimization results. (Uses VectorEngine GPT)")

    if not params:
        st.warning("⚠️ Please complete Setup first.")
    else:
        key_present = bool(get_env("VECTORENGINE_API_KEY") or get_env("OPENAI_API_KEY"))
        if not key_present:
            st.error("Missing VECTORENGINE_API_KEY (or OPENAI_API_KEY). Add it to .env.")
        else:
            if "assistant_messages" not in st.session_state:
                st.session_state.assistant_messages = [
                    {"role": "system", "content": _assistant_system_prompt()},
                    _latest_context_message(params),
                ]

            colA, colB, _ = st.columns([1, 1, 2])

            with colA:
                if st.button("🔄 Refresh Context", width="stretch"):
                    fresh = load_latest_parameters() or {}
                    new_context = _latest_context_message(fresh)

                    st.session_state.assistant_messages = [
                        m for m in st.session_state.assistant_messages if not _is_context_msg(m)
                    ]
                    st.session_state.assistant_messages.append(new_context)
                    st.success("Context refreshed.")

            with colB:
                if st.button("🧹 Clear Chat", width="stretch"):
                    st.session_state.assistant_messages = [
                        {"role": "system", "content": _assistant_system_prompt()},
                        _latest_context_message(load_latest_parameters() or {}),
                    ]
                    st.success("Chat cleared.")

            for m in st.session_state.assistant_messages:
                if m["role"] == "system":
                    continue
                with st.chat_message("user" if m["role"] == "user" else "assistant"):
                    st.markdown(m["content"])

            user_text = st.chat_input(
                "Ask: Why is PV low at 8am? How can I reduce cost? Should I widen Tmax?"
            )
            if user_text:
                st.session_state.assistant_messages.append(
                    {"role": "user", "content": user_text}
                )

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            answer = chat_with_vectorengine(
                                st.session_state.assistant_messages,
                                model="gpt-5-mini-2025-08-07",
                                max_output_tokens=1000,
                            )

                            if answer.strip().startswith("{") and len(answer) > 2000:
                                answer = (
                                    "I received a large structured response. Here’s the concise summary:\n\n"
                                    "- The forecast/optimization looks consistent, but the output was too verbose.\n"
                                    "- Ask me a specific question (e.g., 'why is PV low at 8am?') and I’ll answer directly."
                                )

                            st.markdown(answer)
                            st.session_state.assistant_messages.append(
                                {"role": "assistant", "content": answer}
                            )

                            max_msgs = 20
                            keep = [
                                m for m in st.session_state.assistant_messages
                                if m["role"] == "system"
                            ]
                            rest = [
                                m for m in st.session_state.assistant_messages
                                if m["role"] != "system"
                            ][-max_msgs:]
                            st.session_state.assistant_messages = keep + rest

                        except Exception as e:
                            st.error("Assistant call failed.")
                            st.exception(e)