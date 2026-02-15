# app/optimization/thermal_model.py
def simulate_thermal(heating_powers: list[float], T_ext: list[float], T_init: float = 20.0, alpha=0.1, beta=0.05):
    T = [T_init]
    for t in range(len(heating_powers)):
        next_T = T[-1] + alpha * heating_powers[t] - beta * (T[-1] - T_ext[t])
        T.append(next_T)
    return T[1:]
