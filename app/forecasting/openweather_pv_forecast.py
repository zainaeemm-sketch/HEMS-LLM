# app/forecasting/openweather_pv_forecast.py
from __future__ import annotations

import math
import requests
from dataclasses import dataclass
from typing import List, Tuple, Optional


class OpenWeatherError(RuntimeError):
    pass

def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def geocode_city(city: str, api_key: str, limit: int = 1) -> Tuple[float, float, str]:
    """
    OpenWeather Geocoding API.
    https://openweathermap.org/api/geocoding-api
    """
    if not city or not city.strip():
        raise OpenWeatherError("City is empty.")

    url = "https://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city.strip(), "limit": limit, "appid": api_key}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise OpenWeatherError(f"Geocoding failed ({r.status_code}): {r.text}")

    data = r.json()
    if not data:
        raise OpenWeatherError(f"No geocoding results for city='{city}'.")

    best = data[0]
    lat = _safe_float(best.get("lat"))
    lon = _safe_float(best.get("lon"))
    name = best.get("name") or city.strip()
    country = best.get("country")
    state = best.get("state")
    resolved = ", ".join([p for p in [name, state, country] if p])
    return lat, lon, resolved

@dataclass
class HourlyWx:
    dt: int
    clouds: float  # %
    temp_c: float  # °C


# keep OpenWeatherError, _safe_float, HourlyWx as-is


def fetch_forecast25_hourly(lat: float, lon: float, api_key: str) -> List[HourlyWx]:
    """
    FREE OpenWeather API (Forecast 2.5)
    Endpoint: https://api.openweathermap.org/data/2.5/forecast
    - 3-hour resolution
    - We expand it into hourly by repetition
    """
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
    }

    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise OpenWeatherError(
            f"Forecast 2.5 failed ({r.status_code}): {r.text}"
        )

    data = r.json()
    items = data.get("list", [])
    if not items:
        raise OpenWeatherError("Forecast 2.5 returned empty data.")

    hourly: List[HourlyWx] = []

    for item in items:
        dt = int(item.get("dt", 0))
        clouds = _safe_float(item.get("clouds", {}).get("all"), 0.0)
        temp = _safe_float(item.get("main", {}).get("temp"), 15.0)

        # Each entry represents 3 hours → repeat 3 times
        for i in range(3):
            hourly.append(
                HourlyWx(
                    dt=dt + i * 3600,
                    clouds=clouds,
                    temp_c=temp,
                )
            )

        if len(hourly) >= 48:
            break

    return hourly[:48]

def _day_of_year_from_unix(dt_utc: int) -> int:
    import datetime as _dt
    d = _dt.datetime.utcfromtimestamp(dt_utc)
    return int(d.timetuple().tm_yday)

def _hour_local_from_unix(dt_utc: int, tz_offset_seconds: int) -> int:
    import datetime as _dt
    d = _dt.datetime.utcfromtimestamp(dt_utc + tz_offset_seconds)
    return int(d.hour)

def _clear_sky_shape(lat_deg: float, doy: int, hour_local: int) -> float:
    """
    0..1 clear-sky shape factor via lightweight solar geometry.
    """
    lat = math.radians(lat_deg)
    decl = math.radians(23.44) * math.sin(math.radians((360.0 / 365.0) * (doy - 81)))
    h_angle = math.radians(15.0 * (hour_local - 12))
    cosz = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(h_angle)
    return max(0.0, min(1.0, cosz))

def _temp_derate(temp_c: float) -> float:
    """
    Simple PV temperature derate:
    - assume no penalty <= 25C
    - above 25C: -0.4% per C (typical-ish)
    Clamp: [0.80, 1.00]
    """
    if temp_c <= 25.0:
        return 1.0
    loss = 0.004 * (temp_c - 25.0)
    return max(0.80, 1.0 - loss)

def estimate_pv_kwh_24h_from_weather(
    hourly: List[HourlyWx],
    lat: float,
    capacity_kw: float,
    tz_offset_seconds: int = 0,
    performance_ratio: float = 0.85,
) -> List[float]:
    """
    24h PV energy forecast (kWh per hour) using:
      PV ≈ capacity_kw * clear_sky_shape * cloud_factor * temp_factor * PR
    cloud_factor: 0%->1.0, 100%->0.2 (never fully zero)
    """
    cap = max(_safe_float(capacity_kw, 0.0), 0.0)
    pr = min(max(_safe_float(performance_ratio, 0.85), 0.1), 1.0)

    pv: List[float] = []
    for h in hourly[:24]:
        doy = _day_of_year_from_unix(h.dt)
        hour_local = _hour_local_from_unix(h.dt, tz_offset_seconds)

        clear_shape = _clear_sky_shape(lat_deg=lat, doy=doy, hour_local=hour_local)

        clouds = max(0.0, min(100.0, _safe_float(h.clouds, 0.0)))
        cloud_factor = 1.0 - 0.008 * clouds  # 0%->1.0, 100%->0.2
        cloud_factor = max(0.2, min(1.0, cloud_factor))

        temp_factor = _temp_derate(_safe_float(h.temp_c, 15.0))

        kwh = cap * clear_shape * cloud_factor * temp_factor * pr
        pv.append(float(max(0.0, kwh)))

    if len(pv) < 24:
        pv += [0.0] * (24 - len(pv))
    return pv[:24]
