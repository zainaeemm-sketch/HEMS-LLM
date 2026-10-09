# app/optimization/hems_optimizer.py
import math

import pulp
from utils.constants import TIME_SLOTS, TOU_PRICES

# The optimizer works on 15-minute slots so appliances can start, end and run
# in quarter-hour steps (e.g. 11:15-12:00). Inputs (PV, prices, outdoor temp)
# stay hourly and are held constant within each hour. Results are returned
# both per 15-minute slot and aggregated back to 24 hourly values, so the
# hourly charts and the assistant keep working unchanged.
SLOTS_PER_HOUR = 4
SLOT_MINUTES = 60 // SLOTS_PER_HOUR          # 15
DT = 1.0 / SLOTS_PER_HOUR                     # slot length in hours
N_SLOTS = TIME_SLOTS * SLOTS_PER_HOUR         # 96


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (ValueError, TypeError):
        return float(default)


def _time_to_minutes(value, default):
    """'HH:MM' -> minutes after midnight (24:00 -> 1440). Invalid -> default."""
    try:
        hh, mm = str(value).strip().split(":")[:2]
        minutes = int(hh) * 60 + int(mm)
    except Exception:
        return default
    return max(0, min(minutes, 24 * 60))


def slot_to_hhmm(slot: int) -> str:
    """15-minute slot index -> 'HH:MM' (slot 96 -> '24:00')."""
    minutes = int(slot) * SLOT_MINUTES
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def duration_to_slots(duration_hours) -> int:
    """Cycle length in hours (e.g. 1.25) -> number of 15-minute slots (>= 1)."""
    hours = safe_float(duration_hours, 1.0)
    slots = int(round(hours * SLOTS_PER_HOUR))
    return max(1, min(slots, N_SLOTS))


def _on_windows(values, threshold):
    """Contiguous ON runs in a 15-minute series as [('HH:MM', 'HH:MM'), ...]."""
    windows, start = [], None
    for s, v in enumerate(values):
        on = (v or 0.0) > threshold
        if on and start is None:
            start = s
        elif not on and start is not None:
            windows.append((slot_to_hhmm(start), slot_to_hhmm(s)))
            start = None
    if start is not None:
        windows.append((slot_to_hhmm(start), slot_to_hhmm(len(values))))
    return windows


def _hourly_mean(values):
    """96 quarter-hour values -> 24 hourly means."""
    return [
        sum(values[h * SLOTS_PER_HOUR:(h + 1) * SLOTS_PER_HOUR]) / SLOTS_PER_HOUR
        for h in range(TIME_SLOTS)
    ]


def _expand_hourly(values):
    """24 hourly values -> 96 quarter-hour values (held constant within the hour)."""
    return [values[s // SLOTS_PER_HOUR] for s in range(N_SLOTS)]


def optimize_schedule(
    params: dict,
    pv_forecast: list[float],
    tou_prices: list[float] = TOU_PRICES,
    T_ext: list[float] | None = None,
    feed_in_tariff: float = 0.0,
):
    """
    MILP day-ahead schedule on 15-minute slots.

    - Grid exchange split into import/export (kW):
        net = total_load - pv
        grid_import >= net, grid_import >= 0
        grid_export >= -net, 0 <= grid_export <= pv
      Cost = sum((import*price - export*feed_in_tariff) * slot_hours)
    - Appliance windows and cycle lengths are honoured to the quarter hour.
    - Optional outdoor temperature profile (T_ext, hourly) for the heating model.
    """

    Tmin = safe_float(params.get("Tmin"), 18.0)
    Tmax = safe_float(params.get("Tmax"), 25.0)
    max_power = safe_float(params.get("max_power"), 5.0)

    if not pv_forecast or len(pv_forecast) < TIME_SLOTS:
        pv_forecast = [0.0] * TIME_SLOTS
    else:
        pv_forecast = [safe_float(x, 0.0) for x in pv_forecast[:TIME_SLOTS]]

    if not tou_prices or len(tou_prices) < TIME_SLOTS:
        tou_prices = TOU_PRICES
    else:
        tou_prices = tou_prices[:TIME_SLOTS]

    if T_ext is None or len(T_ext) < TIME_SLOTS:
        T_ext = [15.0] * TIME_SLOTS
    else:
        T_ext = [safe_float(x, 15.0) for x in T_ext[:TIME_SLOTS]]

    # Hourly inputs held constant over each hour's four quarter-hour slots.
    # PV (kWh per hour) equals its average power in kW, so it carries over as kW.
    pv = _expand_hourly(pv_forecast)
    price = _expand_hourly([safe_float(p, 0.0) for p in tou_prices])
    t_out = _expand_hourly(T_ext)
    slots = range(N_SLOTS)

    raw_appliances = params.get("appliances", [])

    appliances = []
    app_settings = {}
    for a in raw_appliances:
        if isinstance(a, dict) and a.get("name"):
            nm = a["name"]
            appliances.append(nm)
            app_settings[nm] = a

    appliances = list(dict.fromkeys(appliances))

    prob = pulp.LpProblem("HEMS_Optimizer", pulp.LpMinimize)

    # Heating is a SPECIAL appliance: when the user adds "Heating" to their
    # appliance list, we add a continuous heating-power variable + thermal
    # model. When the user doesn't add Heating, none of that exists — and
    # "Heating" is NOT shown in the schedule output.
    heating_in_use = any(
        str(name).strip().lower() == "heating" for name in appliances
    )

    if heating_in_use:
        heating_power = pulp.LpVariable.dicts("heating", slots, lowBound=0, upBound=2.0)
    else:
        # Dummy: zero contribution to load, no decision variable.
        heating_power = {s: 0.0 for s in slots}

    app_on = {
        app: pulp.LpVariable.dicts(f"{app}_on", slots, cat="Binary")
        for app in appliances
        if str(app).strip().lower() != "heating"
    }

    app_power = {
        "Air Conditioner": 1.5,
        "Heating": 2.0,
        "Electric Heater": 1.5,
        "Water Heater": 2.0,
        "Dishwasher": 1.2,
        "Washing Machine": 0.8,
        "Dryer": 3.0,
        "EV Charger": 7.0,
    }
    # Case-insensitive lookup so "washing machine" / "WASHING MACHINE"
    # all resolve to the correct rated power instead of falling back to 1.0.
    _app_power_ci = {k.lower(): v for k, v in app_power.items()}

    def _power_for(name: str) -> float:
        return float(_app_power_ci.get(str(name).strip().lower(), 1.0))

    # Power balance per slot (kW).
    total_load = []
    for s in slots:
        fixed = pulp.lpSum(app_on[app][s] * _power_for(app) for app in app_on)
        total_load.append(fixed + heating_power[s])

    net = [total_load[s] - pv[s] for s in slots]

    grid_import = pulp.LpVariable.dicts("grid_import", slots, lowBound=0)
    grid_export = pulp.LpVariable.dicts("grid_export", slots, lowBound=0)

    for s in slots:
        prob += grid_import[s] >= net[s]
        prob += grid_export[s] >= -net[s]
        # Physical cap: you can only export what your PV actually produced.
        # Without this, an inflated feed-in tariff (fit > tou_price) makes
        # grid_export unbounded above and the LP returns "Unbounded".
        prob += grid_export[s] <= pv[s]

    # Energy cost: power (kW) x price (per kWh) x slot length (h).
    fit = max(0.0, safe_float(feed_in_tariff, 0.0))
    prob += pulp.lpSum((grid_import[s] * price[s] - grid_export[s] * fit) * DT for s in slots)

    for s in slots:
        prob += total_load[s] <= max_power

    for app in app_on:
        setting = app_settings.get(app, {})

        # Window in minutes; start rounds UP and end rounds DOWN to the quarter
        # hour, so the appliance never runs outside what the user allowed.
        start_min = _time_to_minutes(setting.get("start_time"), 0) if setting.get("start_time") else 0
        end_min = _time_to_minutes(setting.get("end_time"), 24 * 60) if setting.get("end_time") else 24 * 60
        start_slot = math.ceil(start_min / SLOT_MINUTES)
        end_slot = end_min // SLOT_MINUTES

        # Clamp + sensible fallback (an empty/inverted window means "any time").
        start_slot = max(0, min(start_slot, N_SLOTS))
        end_slot = max(0, min(end_slot, N_SLOTS))
        if end_slot <= start_slot:
            end_slot = N_SLOTS

        # Per-appliance cycle length in 15-minute slots (default 1 h).
        duration = duration_to_slots(setting.get("duration_hours") or 1)

        can_shift = bool(setting.get("can_shift"))

        # Allowed window for the cycle to live in.
        # Shiftable -> anywhere in the day; otherwise within [start, end).
        if can_shift:
            win_start, win_end = 0, N_SLOTS
        else:
            win_start, win_end = start_slot, end_slot

        # Valid cycle-start slots (so the whole cycle fits inside the window).
        if (win_end - win_start) >= duration:
            valid_starts = list(range(win_start, win_end - duration + 1))
        else:
            valid_starts = []

        if not valid_starts:
            # Window too short for even one cycle -> appliance must stay off.
            for s in slots:
                prob += app_on[app][s] == 0
            continue

        # Binary var: did the cycle start at slot k?
        app_start = pulp.LpVariable.dicts(f"{app}_start", valid_starts, cat="Binary")

        # Exactly one cycle per day (must run for `duration` consecutive slots).
        prob += pulp.lpSum(app_start[k] for k in valid_starts) == 1

        # Link on/off to the chosen start: app_on[s] = 1 iff some k with k<=s<k+duration is chosen.
        for s in slots:
            prob += app_on[app][s] == pulp.lpSum(
                app_start[k] for k in valid_starts if k <= s < k + duration
            )

    # Thermal model with SOFT comfort bounds — only when Heating is in use.
    # Hard min/max on T make the LP infeasible whenever heating capacity
    # can't beat outdoor heat loss. Soft bounds use slack variables so
    # violations are allowed but penalized in the objective.
    if heating_in_use:
        T = pulp.LpVariable.dicts("T", slots)  # unbounded
        T_under = pulp.LpVariable.dicts("T_under", slots, lowBound=0)
        T_over = pulp.LpVariable.dicts("T_over", slots, lowBound=0)

        # Hourly rate coefficients, scaled to the 15-minute step.
        alpha, beta = 0.10, 0.05
        prob += T[0] == 20.0
        for s in range(1, N_SLOTS):
            prob += T[s] == T[s - 1] + DT * (
                alpha * heating_power[s - 1] - beta * (T[s - 1] - t_out[s - 1])
            )

        # Soft comfort bounds: T can drift outside [Tmin, Tmax] only by paying a penalty.
        for s in slots:
            prob += T[s] >= Tmin - T_under[s]
            prob += T[s] <= Tmax + T_over[s]

        # Comfort-violation penalty, in $/°C-hour of violation.
        comfort_penalty = 10.0
        prob.objective += comfort_penalty * DT * pulp.lpSum(
            T_under[s] + T_over[s] for s in slots
        )
    else:
        T = None  # no thermal model

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status_code = int(prob.status)
    status_text = pulp.LpStatus.get(status_code, "Unknown")

    def _val(v, default=0.0):
        x = pulp.value(v)
        return float(x) if x is not None else float(default)

    schedule_15 = {app: [_val(app_on[app][s]) for s in slots] for app in app_on}
    windows = {app: _on_windows(vals, 0.5) for app, vals in schedule_15.items()}
    if heating_in_use:
        schedule_15["Heating"] = [_val(heating_power[s]) for s in slots]
        windows["Heating"] = _on_windows(schedule_15["Heating"], 0.01)
        temps_15 = [_val(T[s], default=20.0) for s in slots]
    else:
        # No heating model — indoor temp stays at the initial 20°C placeholder.
        temps_15 = [20.0] * N_SLOTS
    cost = _val(prob.objective)

    gi_15 = [_val(grid_import[s]) for s in slots]
    ge_15 = [_val(grid_export[s]) for s in slots]

    return {
        # Hourly views (24 values) for the charts and the assistant.
        # Appliance values are the fraction of the hour the appliance runs
        # (e.g. 0.25 = 15 min); grid flows are hourly energy (kWh = mean kW).
        "schedule": {app: _hourly_mean(vals) for app, vals in schedule_15.items()},
        "temps": temps_15[::SLOTS_PER_HOUR],
        "grid_import": _hourly_mean(gi_15),
        "grid_export": _hourly_mean(ge_15),
        # Exact 15-minute results.
        "slot_minutes": SLOT_MINUTES,
        "schedule_15min": schedule_15,
        "temps_15min": temps_15,
        "grid_import_15min": gi_15,
        "grid_export_15min": ge_15,
        "windows": {app: [list(w) for w in ws] for app, ws in windows.items()},
        "durations_min": {
            app: int(round(sum(1 for v in vals if v > 0.5) * SLOT_MINUTES))
            for app, vals in schedule_15.items() if app != "Heating"
        },
        "T_ext": T_ext,
        "cost": cost,
        "status": status_text,
        "status_code": status_code,
    }
