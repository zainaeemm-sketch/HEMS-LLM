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
        "Water Heater": 2.0,
        "Dishwasher": 1.2,
        "Washing Machine": 0.8,
        "Dryer": 3.0,
    }

    total_load = []
    for t in range(TIME_SLOTS):
        fixed = pulp.lpSum(app_on.get(app, {}).get(t, 0) * app_power.get(app, 1.0) for app in app_on)
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
        end_h = 24

        if setting.get("start_time"):
            try:
                start_h = int(str(setting["start_time"]).split(":")[0])
            except Exception:
                start_h = 0
        if setting.get("end_time"):
            try:
                end_h = int(str(setting["end_time"]).split(":")[0])
            except Exception:
                end_h = 24

        can_shift = bool(setting.get("can_shift"))

        if can_shift:
            prob += pulp.lpSum(app_on[app][t] for t in range(TIME_SLOTS)) == 2
        else:
            for t in range(TIME_SLOTS):
                if not (start_h <= t < end_h):
                    prob += app_on[app][t] == 0

    # Thermal model (linear)
    T = pulp.LpVariable.dicts("T", range(TIME_SLOTS), lowBound=Tmin, upBound=Tmax)
    alpha, beta = 0.10, 0.05
    prob += T[0] == 20.0
    for t in range(1, TIME_SLOTS):
        prob += T[t] == T[t - 1] + alpha * heating_power[t - 1] - beta * (T[t - 1] - T_ext[t - 1])

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    schedule = {app: [pulp.value(app_on[app][t]) for t in range(TIME_SLOTS)] for app in app_on}
    schedule["Heating"] = [pulp.value(heating_power[t]) for t in range(TIME_SLOTS)]

    temps = [pulp.value(T[t]) for t in range(TIME_SLOTS)]
    cost = float(pulp.value(prob.objective) or 0.0)

    gi = [float(pulp.value(grid_import[t]) or 0.0) for t in range(TIME_SLOTS)]
    ge = [float(pulp.value(grid_export[t]) or 0.0) for t in range(TIME_SLOTS)]

    return {
        "schedule": schedule,
        "temps": temps,
        "cost": cost,
        "grid_import": gi,
        "grid_export": ge,
        "T_ext": T_ext,
    }
