# app/optimization/hems_optimizer.py
import pulp
from utils.constants import TIME_SLOTS, TOU_PRICES

def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (ValueError, TypeError):
        return float(default)

def optimize_schedule(
    params: dict,
    pv_forecast: list[float],
    tou_prices: list[float] = TOU_PRICES,
    T_ext: list[float] | None = None,
    feed_in_tariff: float = 0.0,
):
    """
    Improvements:
    - Splits net grid into import/export vars:
        net = total_load - pv
        grid_import >= net, grid_import >= 0
        grid_export >= -net, grid_export >= 0
      Cost = sum(import*price - export*feed_in_tariff)
    - Can use external temperature profile (T_ext) from OpenWeather hourly temps.
    """

    Tmin = safe_float(params.get("Tmin"), 18.0)
    Tmax = safe_float(params.get("Tmax"), 25.0)
    max_power = safe_float(params.get("max_power"), 5.0)

    if not pv_forecast or len(pv_forecast) < TIME_SLOTS:
        pv_forecast = [0.0] * TIME_SLOTS
    else:
        pv_forecast = pv_forecast[:TIME_SLOTS]

    if not tou_prices or len(tou_prices) < TIME_SLOTS:
        tou_prices = TOU_PRICES
    else:
        tou_prices = tou_prices[:TIME_SLOTS]

    if T_ext is None or len(T_ext) < TIME_SLOTS:
        T_ext = [15.0] * TIME_SLOTS
    else:
        T_ext = [safe_float(x, 15.0) for x in T_ext[:TIME_SLOTS]]

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

    heating_power = pulp.LpVariable.dicts("heating", range(TIME_SLOTS), lowBound=0, upBound=2.0)

    app_on = {
        app: pulp.LpVariable.dicts(f"{app}_on", range(TIME_SLOTS), cat="Binary")
        for app in appliances
        if app != "Heating"
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

    total_load = []
    for t in range(TIME_SLOTS):
        fixed = pulp.lpSum(
            app_on.get(app, {}).get(t, 0) * _power_for(app) for app in app_on
        )
        total_load.append(fixed + heating_power[t])

    net = [total_load[t] - pv_forecast[t] for t in range(TIME_SLOTS)]

    grid_import = pulp.LpVariable.dicts("grid_import", range(TIME_SLOTS), lowBound=0)
    grid_export = pulp.LpVariable.dicts("grid_export", range(TIME_SLOTS), lowBound=0)

    for t in range(TIME_SLOTS):
        prob += grid_import[t] >= net[t]
        prob += grid_export[t] >= -net[t]

    fit = max(0.0, safe_float(feed_in_tariff, 0.0))
    prob += pulp.lpSum([grid_import[t] * tou_prices[t] - grid_export[t] * fit for t in range(TIME_SLOTS)])

    for t in range(TIME_SLOTS):
        prob += total_load[t] <= max_power

    for app in app_on:
        setting = app_settings.get(app, {})
        start_h = 0
        end_h = TIME_SLOTS

        if setting.get("start_time"):
            try:
                start_h = int(str(setting["start_time"]).split(":")[0])
            except Exception:
                start_h = 0
        if setting.get("end_time"):
            try:
                end_h = int(str(setting["end_time"]).split(":")[0])
            except Exception:
                end_h = TIME_SLOTS

        # Clamp + sensible fallback
        start_h = max(0, min(start_h, TIME_SLOTS))
        end_h = max(start_h, min(end_h, TIME_SLOTS))
        if end_h <= start_h:
            end_h = TIME_SLOTS

        # Per-appliance cycle length (hours). Defaults to 1h.
        duration = int(setting.get("duration_hours") or 1)
        duration = max(1, min(duration, TIME_SLOTS))

        can_shift = bool(setting.get("can_shift"))

        # Allowed window for the cycle to live in.
        # Shiftable -> anywhere in the day; otherwise within [start_h, end_h).
        if can_shift:
            win_start, win_end = 0, TIME_SLOTS
        else:
            win_start, win_end = start_h, end_h

        # Valid cycle-start indices (so the whole cycle fits inside the window).
        if (win_end - win_start) >= duration:
            valid_starts = list(range(win_start, win_end - duration + 1))
        else:
            valid_starts = []

        if not valid_starts:
            # Window too short for even one cycle -> appliance must stay off.
            for t in range(TIME_SLOTS):
                prob += app_on[app][t] == 0
            continue

        # Binary var: did the cycle start at hour s?
        app_start = pulp.LpVariable.dicts(
            f"{app}_start", valid_starts, cat="Binary"
        )

        # Exactly one cycle per day (must run for `duration` consecutive hours).
        prob += pulp.lpSum(app_start[s] for s in valid_starts) == 1

        # Link on/off to the chosen start: app_on[t] = 1 iff some s with s<=t<s+duration is chosen.
        for t in range(TIME_SLOTS):
            prob += app_on[app][t] == pulp.lpSum(
                app_start[s] for s in valid_starts if s <= t < s + duration
            )

    # Thermal model with SOFT comfort bounds.
    # Hard min/max on T make the LP infeasible whenever heating capacity
    # can't beat outdoor heat loss. Soft bounds use slack variables so
    # violations are allowed but penalized in the objective.
    T = pulp.LpVariable.dicts("T", range(TIME_SLOTS))  # unbounded
    T_under = pulp.LpVariable.dicts("T_under", range(TIME_SLOTS), lowBound=0)
    T_over = pulp.LpVariable.dicts("T_over", range(TIME_SLOTS), lowBound=0)

    alpha, beta = 0.10, 0.05
    prob += T[0] == 20.0
    for t in range(1, TIME_SLOTS):
        prob += T[t] == T[t - 1] + alpha * heating_power[t - 1] - beta * (T[t - 1] - T_ext[t - 1])

    # Soft comfort bounds: T can drift outside [Tmin, Tmax] only by paying a penalty.
    for t in range(TIME_SLOTS):
        prob += T[t] >= Tmin - T_under[t]
        prob += T[t] <= Tmax + T_over[t]

    # Add the comfort-violation penalty to the existing cost objective.
    comfort_penalty = 10.0  # $/°C-hour of violation — high enough to dominate when feasible
    prob.objective += comfort_penalty * pulp.lpSum(
        T_under[t] + T_over[t] for t in range(TIME_SLOTS)
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    def _val(v, default=0.0):
        x = pulp.value(v)
        return float(x) if x is not None else float(default)

    schedule = {app: [_val(app_on[app][t]) for t in range(TIME_SLOTS)] for app in app_on}
    schedule["Heating"] = [_val(heating_power[t]) for t in range(TIME_SLOTS)]

    temps = [_val(T[t], default=20.0) for t in range(TIME_SLOTS)]
    cost = _val(prob.objective)

    gi = [_val(grid_import[t]) for t in range(TIME_SLOTS)]
    ge = [_val(grid_export[t]) for t in range(TIME_SLOTS)]

    return {
        "schedule": schedule,
        "temps": temps,
        "cost": cost,
        "grid_import": gi,
        "grid_export": ge,
        "T_ext": T_ext,
    }