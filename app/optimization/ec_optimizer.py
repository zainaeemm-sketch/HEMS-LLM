# app/optimization/ec_optimizer.py
"""
Energy Community (EC) optimizer — Python port of the MATLAB code:
  general.m    -> run_ec_game()          (the round-based game loop)
  ECcost.m     -> _ec_cost_user()        (per-user objective)
  constraint.m -> spread constraint      (adjustable-load spread, via penalty)
  ECcostESS.m  -> _ec_cost_ess()         (community battery objective)

Decision variables per user (MATLAB's 96-vector x):
  x[ 0:24]  binary ON/OFF of shiftable load 1  (exactly n1 ON hours)
  x[24:48]  binary ON/OFF of shiftable load 2  (exactly n2 ON hours)
  x[48:72]  binary ON/OFF of shiftable load 3  (exactly n3 ON hours)
  x[72:96]  continuous adjustable load, 0..max(LA1), sum == sum(LA1)

The MATLAB linear equalities (Aeq/Beq: fixed number of ON slots, fixed
adjustable energy) are enforced *by construction* in the chromosome
encoding, so the GA never produces infeasible individuals for those.
The nonlinear constraint from constraint.m
    count(adjustable > 0) >= count(LA1 > 0) - 2
is enforced with a penalty, exactly as MATLAB's ga() treats @constraint.

No new dependencies: numpy only (works on Streamlit Cloud as-is).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

H = 24  # hourly planning horizon


# ----------------------------------------------------------------------
# Parameters (defaults mirror the "predefine" section of general.m)
# ----------------------------------------------------------------------
@dataclass
class ECParams:
    n_users: int = 10          # Nuser
    rounds: int = 3            # rounds of the game (MATLAB used 10)
    cbuy_grid: float = 0.14    # €/kWh buy price
    csell_grid: float = 0.09   # €/kWh sell price
    gse_inc: float = 0.12      # €/kWh shared-energy incentive (GSE)
    c_comf: float = 0.06       # €/kWh discomfort valuation
    eff_solar: float = 0.9     # PV scaling coefficient

    # --- ESS ---
    ess_enabled: bool = True
    ess_price: float = 4000.0      # €
    ess_size: float = 5.0          # kWh
    ess_lifecycle: float = 7000.0  # deep cycles
    ess_init_cap: float = 2.5      # kWh
    ess_maxrate_frac: float = 0.6  # ESS_maxrate = frac * size (kW)
    ess_min_soc_frac: float = 0.2  # lower SOC bound

    # --- GA (users) ---
    ga_pop: int = 60
    ga_gen: int = 40
    ga_elite_frac: float = 0.02    # k in general.m: carried to next round
    seed: int | None = 0

    @property
    def c_ess(self) -> float:
        # C_ESS = (ESS_price / (2*lifecycle)) / size   (€/kWh throughput)
        return (self.ess_price / (2.0 * self.ess_lifecycle)) / self.ess_size

    @property
    def ess_maxrate(self) -> float:
        return self.ess_maxrate_frac * self.ess_size


# ----------------------------------------------------------------------
# User profile container (rows 1..6 of profh in general.m)
# ----------------------------------------------------------------------
@dataclass
class UserProfile:
    total: np.ndarray      # row 1: total consumption          (24,)
    critical: np.ndarray   # row 2: critical (non-shiftable)   (24,)
    ls1: np.ndarray        # row 3: shiftable load 1           (24,)
    ls2: np.ndarray        # row 4: shiftable load 2           (24,)
    ls3: np.ndarray        # row 5: shiftable load 3           (24,)
    la1: np.ndarray        # row 6: adjustable load            (24,)

    def __post_init__(self):
        for name in ("total", "critical", "ls1", "ls2", "ls3", "la1"):
            v = np.asarray(getattr(self, name), dtype=float).ravel()[:H]
            if v.size < H:
                v = np.pad(v, (0, H - v.size))
            setattr(self, name, v)


# ----------------------------------------------------------------------
# Synthetic default community (stands in for prof30user_load_identify.mat)
# ----------------------------------------------------------------------
def make_synthetic_community(n_users: int = 10, seed: int = 42) -> list[UserProfile]:
    """Generate plausible residential profiles: a critical base curve,
    3 shiftable appliances (washing machine / dishwasher / water heater
    style blocks) and one adjustable load (EV-charging style energy)."""
    rng = np.random.default_rng(seed)
    hours = np.arange(H)
    users: list[UserProfile] = []
    for _ in range(n_users):
        # critical base load: morning + evening humps
        base = (0.25
                + 0.35 * np.exp(-0.5 * ((hours - 8) / 2.2) ** 2)
                + 0.55 * np.exp(-0.5 * ((hours - 20) / 2.5) ** 2))
        crit = base * rng.uniform(0.8, 1.3) + rng.uniform(0.0, 0.05, H)

        def _block(power_rng, dur_rng, start_rng):
            p = rng.uniform(*power_rng)
            d = int(rng.integers(*dur_rng))
            s = int(rng.integers(*start_rng))
            v = np.zeros(H)
            v[s:s + d] = p
            return v

        ls1 = _block((0.7, 1.1), (2, 4), (17, 21))    # washing-machine-ish
        ls2 = _block((0.9, 1.4), (1, 3), (18, 22))    # dishwasher-ish
        ls3 = _block((1.3, 2.1), (2, 4), (6, 10))     # water-heater-ish
        la1 = np.zeros(H)
        n_adj = int(rng.integers(3, 6))
        slots = rng.choice(H, size=n_adj, replace=False)
        la1[slots] = rng.uniform(0.4, 1.2, n_adj)     # adjustable (EV-ish)

        total = crit + ls1 + ls2 + ls3 + la1
        users.append(UserProfile(total, crit, ls1, ls2, ls3, la1))
    return users


def default_solar_profile(peak_kw: float = 12.0) -> np.ndarray:
    """Bell-shaped community PV curve (kW per hour) if no forecast given."""
    hours = np.arange(H)
    g = np.exp(-0.5 * ((hours - 13) / 3.0) ** 2)
    g[(hours < 6) | (hours > 20)] = 0.0
    return peak_kw * g / g.max()


# ----------------------------------------------------------------------
# Chromosome encoding / decoding for one user
# ----------------------------------------------------------------------
@dataclass
class _UserEncoding:
    n1: int; v1: np.ndarray   # ON count + power values, shiftable 1
    n2: int; v2: np.ndarray
    n3: int; v3: np.ndarray
    n4: int                   # count(LA1 > 0)
    la_total: float           # sum(LA1)  (Beq for adjustable)
    la_max: float             # ub for adjustable slots


def _encode_user(u: UserProfile) -> _UserEncoding:
    def nz(v):
        idx = np.flatnonzero(v > 1e-9)
        return len(idx), v[idx].copy()
    n1, v1 = nz(u.ls1)
    n2, v2 = nz(u.ls2)
    n3, v3 = nz(u.ls3)
    n4 = int(np.count_nonzero(u.la1 > 1e-9))
    return _UserEncoding(n1, v1, n2, v2, n3, v3, n4,
                         float(u.la1.sum()), float(u.la1.max() if u.la1.max() > 0 else 0.0))


def _shift_profile(hours_sel: np.ndarray, values: np.ndarray) -> np.ndarray:
    """MATLAB: x(find(x)) = v  — place appliance power values on chosen hours."""
    out = np.zeros(H)
    if len(hours_sel):
        out[np.sort(hours_sel)] = values[: len(hours_sel)]
    return out


def _decode(ind: dict, enc: _UserEncoding) -> np.ndarray:
    """Individual -> Pshift (24,) = LS1+LS2+LS3+LA1 optimized (ECcost's Pshift)."""
    p = (_shift_profile(ind["s1"], enc.v1)
         + _shift_profile(ind["s2"], enc.v2)
         + _shift_profile(ind["s3"], enc.v3)
         + ind["a"])
    return p


# ----------------------------------------------------------------------
# ECcost.m — per-user objective
# ----------------------------------------------------------------------
def _ec_cost_user(pshift: np.ndarray, adj: np.ndarray, *,
                  gensolar, pess, prof_ec_others, crit, prof_total_user,
                  p: ECParams, n4: int) -> float:
    cons_all = prof_ec_others + crit + pshift

    # incentive: reward matching generation & community consumption
    gse = -p.gse_inc * np.abs(gensolar + pess - cons_all)

    # grid purchase (users pay for all consumption first)
    pgbuy = crit + pshift

    # discomfort: only demand reduction is penalised
    pcomf = prof_total_user - (crit + pshift)
    pcomf = pcomf * (pcomf > 1e-4)

    # export (only when community production > consumption), user's share
    pgsell = (gensolar + pess - cons_all) / p.n_users
    pgsell = pgsell * (pgsell > 0)

    z = float(np.sum(-gse + p.cbuy_grid * pgbuy
                     + p.c_comf * pcomf - p.csell_grid * pgsell))

    # constraint.m: count(adjustable>0) >= n4 - 2   (penalty formulation)
    c = -np.count_nonzero(adj > 1e-9) + n4 - 2
    if c > 0:
        z += 1e3 * c
    return z


# ----------------------------------------------------------------------
# GA for one user (mirrors ga(@ECcost, 96, ..., @constraint, integcol))
# ----------------------------------------------------------------------
def _repair_adjustable(a: np.ndarray, enc: _UserEncoding, rng) -> np.ndarray:
    a = np.clip(a, 0.0, enc.la_max if enc.la_max > 0 else 0.0)
    s = a.sum()
    if enc.la_total <= 1e-12:
        return np.zeros(H)
    if s <= 1e-12:  # dead individual: respawn on random slots
        k = max(enc.n4, 1)
        a = np.zeros(H)
        a[rng.choice(H, size=min(k, H), replace=False)] = enc.la_total / k
        return a
    a = a * (enc.la_total / s)          # enforce sum == sum(LA1)  (Beq)
    if enc.la_max > 0:                  # rescale can break ub — fix & re-spread
        over = a > enc.la_max
        if over.any():
            excess = float(np.sum(a[over] - enc.la_max))
            a[over] = enc.la_max
            room = enc.la_max - a
            room[room < 0] = 0
            tr = room.sum()
            if tr > 1e-12:
                a += room * (excess / tr)
    return a


def _random_individual(enc: _UserEncoding, rng) -> dict:
    ind = {
        "s1": rng.choice(H, size=enc.n1, replace=False) if enc.n1 else np.array([], int),
        "s2": rng.choice(H, size=enc.n2, replace=False) if enc.n2 else np.array([], int),
        "s3": rng.choice(H, size=enc.n3, replace=False) if enc.n3 else np.array([], int),
        "a": rng.uniform(0, enc.la_max if enc.la_max > 0 else 0.0, H),
    }
    ind["a"] = _repair_adjustable(ind["a"], enc, rng)
    return ind


def _individual_from_profile(u: UserProfile, enc: _UserEncoding) -> dict:
    return {
        "s1": np.flatnonzero(u.ls1 > 1e-9),
        "s2": np.flatnonzero(u.ls2 > 1e-9),
        "s3": np.flatnonzero(u.ls3 > 1e-9),
        "a": u.la1.copy(),
    }


def _crossover_hours(h1: np.ndarray, h2: np.ndarray, n: int, rng) -> np.ndarray:
    if n == 0:
        return np.array([], int)
    pool = np.union1d(h1, h2)
    if len(pool) < n:
        extra = np.setdiff1d(np.arange(H), pool)
        pool = np.concatenate([pool, rng.choice(extra, size=n - len(pool), replace=False)])
    return rng.choice(pool, size=n, replace=False)


def _mutate_hours(h: np.ndarray, rng, rate=0.15) -> np.ndarray:
    h = h.copy()
    for i in range(len(h)):
        if rng.random() < rate:
            free = np.setdiff1d(np.arange(H), h)
            if len(free):
                h[i] = rng.choice(free)
    return h


def _ga_user(enc: _UserEncoding, cost_fn, seeds: list[dict],
             pop_size: int, gens: int, rng) -> tuple[dict, float, list[dict]]:
    # initial population (seeded like MATLAB's InitialPopulationMatrix)
    pop = [dict(s) for s in seeds][:pop_size]
    while len(pop) < pop_size:
        pop.append(_random_individual(enc, rng))
    fit = np.array([cost_fn(ind) for ind in pop])

    for _ in range(gens):
        order = np.argsort(fit)
        pop = [pop[i] for i in order]
        fit = fit[order]
        elite = max(2, pop_size // 10)
        newpop = [dict(pop[i]) for i in range(elite)]
        while len(newpop) < pop_size:
            # tournament selection
            i, j = rng.integers(0, pop_size, 2)
            p1 = pop[min(i, j)]
            i, j = rng.integers(0, pop_size, 2)
            p2 = pop[min(i, j)]
            if rng.random() < 0.8:  # CrossoverFraction 0.8
                child = {
                    "s1": _crossover_hours(p1["s1"], p2["s1"], enc.n1, rng),
                    "s2": _crossover_hours(p1["s2"], p2["s2"], enc.n2, rng),
                    "s3": _crossover_hours(p1["s3"], p2["s3"], enc.n3, rng),
                    "a": _repair_adjustable(
                        0.5 * (p1["a"] + p2["a"])
                        + rng.normal(0, 0.05 * max(enc.la_max, 1e-6), H),
                        enc, rng),
                }
            else:
                child = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in p1.items()}
            # mutation
            child["s1"] = _mutate_hours(child["s1"], rng)
            child["s2"] = _mutate_hours(child["s2"], rng)
            child["s3"] = _mutate_hours(child["s3"], rng)
            if rng.random() < 0.3:
                a = child["a"] + rng.normal(0, 0.1 * max(enc.la_max, 1e-6), H)
                child["a"] = _repair_adjustable(a, enc, rng)
            newpop.append(child)
        pop = newpop
        fit = np.array([cost_fn(ind) for ind in pop])

    order = np.argsort(fit)
    pop = [pop[i] for i in order]
    fit = fit[order]
    return pop[0], float(fit[0]), pop  # best, best fval, sorted final pop


# ----------------------------------------------------------------------
# ECcostESS.m — community battery objective  (+ linear constraint penalty)
# ----------------------------------------------------------------------
def _ec_cost_ess(ess: np.ndarray, *, gensolar, prof_ec, prof_init_total,
                 p: ECParams) -> float:
    net = gensolar + ess - prof_ec
    gse = -p.gse_inc * np.abs(net)

    pgbuy = prof_ec
    pcomf = prof_init_total - prof_ec
    pcomf = pcomf * (pcomf > 1e-4)
    pgsell = net * (net > 0)
    gess = p.c_ess * np.abs(ess)

    return float(np.sum(-gse + p.cbuy_grid * pgbuy + p.c_comf * pcomf
                        - p.csell_grid * pgsell + gess))


def _ess_penalty(ess: np.ndarray, *, gensolar, prof_ec, p: ECParams) -> float:
    """Linear constraints of general.m (AlinESS/BlinESS), as penalties.
       Convention: ess>0 discharge, ess<0 charge; SOC = init + cumsum(-ess)."""
    pen = 0.0
    # 1) don't charge more than local surplus (no charging from grid)
    surplus = np.maximum(gensolar - prof_ec, 0.0)
    viol = (-ess) - surplus                      # charging power - surplus
    pen += np.sum(np.maximum(viol, 0.0))
    # 2) SOC within [min_frac*size, size]
    soc = p.ess_init_cap + np.cumsum(-ess)
    pen += np.sum(np.maximum(soc - p.ess_size, 0.0))
    pen += np.sum(np.maximum(p.ess_min_soc_frac * p.ess_size - soc[:-1], 0.0))
    # 3) end-of-day SOC >= initial SOC
    pen += max(p.ess_init_cap - soc[-1], 0.0)
    return 1e3 * pen


def _optimize_ess(gensolar, prof_ec, prof_init_total, p: ECParams, rng
                  ) -> tuple[np.ndarray, float]:
    """GA over the 24 ESS power variables (port of the ESS ga() call)."""
    lo, hi = -p.ess_maxrate, p.ess_maxrate

    def cost(e):
        return (_ec_cost_ess(e, gensolar=gensolar, prof_ec=prof_ec,
                             prof_init_total=prof_init_total, p=p)
                + _ess_penalty(e, gensolar=gensolar, prof_ec=prof_ec, p=p))

    pop_size, gens = 80, 80
    pop = rng.uniform(lo, hi, (pop_size, H)) * 0.3
    pop[0] = 0.0  # include the "do nothing" individual
    fit = np.array([cost(e) for e in pop])
    for _ in range(gens):
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        elite = pop_size // 10
        newpop = [pop[i].copy() for i in range(elite)]
        while len(newpop) < pop_size:
            i, j = rng.integers(0, pop_size, 2); p1 = pop[min(i, j)]
            i, j = rng.integers(0, pop_size, 2); p2 = pop[min(i, j)]
            alpha = rng.random()
            child = alpha * p1 + (1 - alpha) * p2
            child += rng.normal(0, 0.08 * p.ess_maxrate, H) * (rng.random(H) < 0.3)
            newpop.append(np.clip(child, lo, hi))
        pop = np.array(newpop)
        fit = np.array([cost(e) for e in pop])
    b = int(np.argmin(fit))
    return pop[b].copy(), float(fit[b])


# ----------------------------------------------------------------------
# general.m — the round-based game
# ----------------------------------------------------------------------
def run_ec_game(users: list[UserProfile] | None = None,
                gensolar: np.ndarray | list | None = None,
                params: ECParams | None = None,
                progress_cb=None) -> dict:
    """
    Run the community optimization game.

    users     : list of UserProfile (defaults to a synthetic 10-user community)
    gensolar  : 24-hour community PV generation in kWh/h (e.g. your app's
                pv_forecast). Scaled by params.eff_solar, like general.m.
    params    : ECParams
    progress_cb(frac, msg) : optional callback for a Streamlit progress bar.

    Returns a results dict with per-round community profiles, per-user
    consumption, ESS schedule + SOC, costs and shared-energy metrics.
    """
    p = params or ECParams()
    rng = np.random.default_rng(p.seed)

    if users is None:
        users = make_synthetic_community(p.n_users, seed=p.seed or 42)
    p.n_users = len(users)

    if gensolar is None:
        gensolar = default_solar_profile(peak_kw=1.2 * p.n_users)
    gensolar = np.asarray(gensolar, dtype=float).ravel()[:H]
    if gensolar.size < H:
        gensolar = np.pad(gensolar, (0, H - gensolar.size))
    gensolar = gensolar * p.eff_solar

    encs = [_encode_user(u) for u in users]
    crit = np.array([u.critical for u in users])          # (N,24)
    prof0 = np.array([u.total for u in users])            # (N,24) initial
    prof_ec_ttl = [prof0.sum(axis=0)]                     # PROFECTTL rows
    prof_ec_user = [prof0.copy()]                         # PROFECUSER cells
    pess = np.zeros(H)                                    # PESS row for round 1
    elite_k = max(1, round(p.ga_elite_frac * p.ga_pop))
    carry: list[list[dict]] = [[] for _ in users]         # final_pop2

    fval = np.zeros((p.rounds, p.n_users))
    ess_hist, soc_hist, fval_ess = [], [], []
    total_steps = p.rounds * (p.n_users + (1 if p.ess_enabled else 0))
    step = 0

    for r in range(p.rounds):
        usercons = np.zeros((p.n_users, H))
        cur_ttl = prof_ec_ttl[r].copy()
        for uix, (u, enc) in enumerate(zip(users, encs)):
            prof_others = cur_ttl - prof_ec_user[r][uix]

            def cost_fn(ind, _enc=enc, _prof_others=prof_others, _uix=uix):
                psh = _decode(ind, _enc)
                return _ec_cost_user(
                    psh, ind["a"],
                    gensolar=gensolar, pess=pess,
                    prof_ec_others=_prof_others,
                    crit=crit[_uix], prof_total_user=users[_uix].total,
                    p=p, n4=_enc.n4)

            seeds = [_individual_from_profile(u, enc)] + carry[uix]
            best, fv, finalpop = _ga_user(enc, cost_fn, seeds,
                                          p.ga_pop, p.ga_gen, rng)
            carry[uix] = [dict(finalpop[i]) for i in range(min(elite_k, len(finalpop)))]
            fval[r, uix] = fv
            usercons[uix] = _decode(best, enc) + crit[uix]

            step += 1
            if progress_cb:
                progress_cb(step / total_steps,
                            f"Round {r+1}/{p.rounds} — user {uix+1}/{p.n_users}")

        prof_ec_ttl.append(usercons.sum(axis=0))
        prof_ec_user.append(usercons)

        if p.ess_enabled:
            ess, fv_ess = _optimize_ess(
                gensolar, prof_ec_ttl[r], prof0.sum(axis=0), p, rng)
            soc = (p.ess_init_cap + np.cumsum(-ess)) / p.ess_size * 100.0
            ess_hist.append(ess); soc_hist.append(soc); fval_ess.append(fv_ess)
            pess = ess.copy()   # users see this ESS profile next round
            step += 1
            if progress_cb:
                progress_cb(step / total_steps, f"Round {r+1}/{p.rounds} — battery")

    # -------- shared-energy KPI:  sum(min(consumption, gen+ESS)) --------
    shared = []
    for r in range(1, p.rounds + 1):
        e = ess_hist[r - 1] if (p.ess_enabled and ess_hist) else np.zeros(H)
        shared.append(float(np.sum(np.minimum(prof_ec_ttl[r], gensolar + e))))
    shared0 = float(np.sum(np.minimum(prof_ec_ttl[0], gensolar)))

    return {
        "gensolar": gensolar.tolist(),
        "prof_ec_ttl": [v.tolist() for v in prof_ec_ttl],
        "prof_ec_user": [m.tolist() for m in prof_ec_user],
        "crit_total": crit.sum(axis=0).tolist(),
        "pess": (ess_hist[-1].tolist() if ess_hist else [0.0] * H),
        "ess_history": [e.tolist() for e in ess_hist],
        "soc": (soc_hist[-1].tolist() if soc_hist else []),
        "fval": fval.tolist(),
        "fval_ess": fval_ess,
        "shared_energy_initial": shared0,
        "shared_energy_by_round": shared,
        "n_users": p.n_users,
        "rounds": p.rounds,
        "params": {
            "cbuy_grid": p.cbuy_grid, "csell_grid": p.csell_grid,
            "gse_inc": p.gse_inc, "c_comf": p.c_comf,
            "ess_size": p.ess_size, "ess_maxrate": p.ess_maxrate,
            "c_ess": p.c_ess, "ess_enabled": p.ess_enabled,
        },
    }


if __name__ == "__main__":
    res = run_ec_game(params=ECParams(rounds=2, ga_pop=40, ga_gen=25),
                      progress_cb=lambda f, m: print(f"{f*100:5.1f}%  {m}"))
    print("\nShared energy (initial):", round(res["shared_energy_initial"], 2), "kWh")
    for i, s in enumerate(res["shared_energy_by_round"], 1):
        print(f"Shared energy round {i}:", round(s, 2), "kWh")
    print("Final community fval per user:", np.round(res["fval"][-1], 3))
