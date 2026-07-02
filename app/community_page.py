# app/community_page.py
"""
Streamlit page for the Energy Community (EC) optimizer — the MATLAB
general.m / ECcost.m / ECcostESS.m / constraint.m model ported to Python
(see optimization/ec_optimizer.py).

Usage in main.py:

    from community_page import render_community_page
    ...
    elif page == "community":
        render_community_page(params)   # params = load_latest_parameters()
"""

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from optimization.ec_optimizer import (
    ECParams,
    run_ec_game,
    make_synthetic_community,
    UserProfile,
)


def _hours_df(series: dict) -> pd.DataFrame:
    rows = []
    for name, vals in series.items():
        for h, v in enumerate(vals):
            rows.append({"hour": h, "kWh": float(v), "series": name})
    return pd.DataFrame(rows)


def render_community_page(params: dict | None = None):
    st.header("🏘️ Energy Community Optimization")
    st.caption(
        "Round-based community game: every household re-schedules its "
        "3 shiftable + 1 adjustable loads against the rest of the community "
        "(GSE shared-energy incentive), then a shared battery is optimized. "
        "Port of the MATLAB `general.m` / `ECcost.m` / `ECcostESS.m` model."
    )

    params = params or {}
    pv_forecast = params.get("pv_forecast")

    # ---------------- configuration ----------------
    with st.expander("⚙️ Community & market parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_users = st.number_input("Number of households", 2, 30, 10)
            rounds = st.number_input("Rounds of the game", 1, 10, 3)
            seed = st.number_input("Random seed", 0, 9999, 0)
        with c2:
            cbuy = st.number_input("Buy price €/kWh (Cbuy_grid)", 0.0, 2.0, 0.14, 0.01)
            csell = st.number_input("Sell price €/kWh (Csell_grid)", 0.0, 2.0, 0.09, 0.01)
            gse = st.number_input("GSE incentive €/kWh (shared energy)", 0.0, 2.0, 0.12, 0.01)
            ccomf = st.number_input("Discomfort cost €/kWh (C_comf)", 0.0, 2.0, 0.06, 0.01)
        with c3:
            use_pv = st.checkbox(
                "Use PV forecast from Forecast page",
                value=bool(pv_forecast),
                disabled=not bool(pv_forecast),
                help="If unchecked (or no forecast saved), a default bell-shaped "
                     "community PV curve is used.",
            )
            pv_scale = st.number_input(
                "PV scale factor (Effsolar × community size)",
                0.1, 100.0, float(n_users) if pv_forecast else 1.0, 0.1,
                help="Your saved forecast is for ONE home — scale it up to "
                     "represent the whole community's generation.",
            )
            eff_solar = st.number_input("Effsolar", 0.1, 3.0, 0.9, 0.05)

    with st.expander("🔋 Shared battery (ESS)"):
        b1, b2, b3 = st.columns(3)
        with b1:
            ess_enabled = st.checkbox("Enable community ESS", value=True)
            ess_size = st.number_input("Capacity (kWh)", 0.5, 100.0, 5.0, 0.5)
        with b2:
            ess_price = st.number_input("Battery price (€)", 100.0, 100000.0, 4000.0, 100.0)
            ess_cycles = st.number_input("Lifecycle (cycles)", 500.0, 20000.0, 7000.0, 100.0)
        with b3:
            ess_init = st.number_input("Initial charge (kWh)", 0.0, 100.0, 2.5, 0.5)
            ess_rate = st.slider("Max charge/discharge rate (× capacity)", 0.1, 1.0, 0.6, 0.05)

    with st.expander("🧬 Genetic algorithm settings"):
        g1, g2 = st.columns(2)
        with g1:
            ga_pop = st.number_input("Population size", 10, 300, 60, 10)
        with g2:
            ga_gen = st.number_input("Generations", 5, 200, 40, 5)
        st.caption(
            "MATLAB defaults were 100 × 50; 60 × 40 keeps each run under "
            "~1 min on Streamlit Cloud while giving very similar schedules."
        )

    # ---------------- run ----------------
    if st.button("▶ Run Community Optimization", type="primary"):
        p = ECParams(
            n_users=int(n_users), rounds=int(rounds), seed=int(seed),
            cbuy_grid=float(cbuy), csell_grid=float(csell),
            gse_inc=float(gse), c_comf=float(ccomf), eff_solar=float(eff_solar),
            ess_enabled=bool(ess_enabled), ess_size=float(ess_size),
            ess_price=float(ess_price), ess_lifecycle=float(ess_cycles),
            ess_init_cap=float(min(ess_init, ess_size)),
            ess_maxrate_frac=float(ess_rate),
            ga_pop=int(ga_pop), ga_gen=int(ga_gen),
        )

        gensolar = None
        if use_pv and pv_forecast:
            gensolar = np.asarray(pv_forecast, dtype=float)[:24] * float(pv_scale)

        users = make_synthetic_community(p.n_users, seed=p.seed or 42)

        bar = st.progress(0.0, text="Starting community game…")

        def _cb(frac, msg):
            bar.progress(min(frac, 1.0), text=msg)

        with st.spinner("Running the community game…"):
            results = run_ec_game(users=users, gensolar=gensolar,
                                  params=p, progress_cb=_cb)
        bar.empty()
        st.session_state["ec_results"] = results
        st.success("✅ Community optimization complete.")

    # ---------------- results ----------------
    results = st.session_state.get("ec_results")
    if not results:
        st.info("Configure the community above and press **Run**.")
        return

    R = results["rounds"]
    gensolar = np.asarray(results["gensolar"])
    prof_ttl = [np.asarray(v) for v in results["prof_ec_ttl"]]
    pess = np.asarray(results["pess"])

    # KPIs
    s0 = results["shared_energy_initial"]
    s_final = results["shared_energy_by_round"][-1]
    best_round = int(np.argmax(results["shared_energy_by_round"]))
    s_best = results["shared_energy_by_round"][best_round]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Shared energy — before", f"{s0:.1f} kWh")
    k2.metric("Shared energy — final round", f"{s_final:.1f} kWh",
              f"{(s_final - s0):+.1f} kWh")
    k3.metric("Best round", f"#{best_round + 1}", f"{s_best:.1f} kWh")
    k4.metric("Community peak (final)",
              f"{prof_ttl[-1].max():.2f} kW",
              f"{(prof_ttl[-1].max() - prof_ttl[0].max()):+.2f} vs initial")

    # ---- community profile chart (the main figure of general.m) ----
    st.subheader("Community load vs generation")
    df = _hours_df({
        "EC total — initial": prof_ttl[0],
        f"EC total — after round {R}": prof_ttl[-1],
        "Generation + ESS": gensolar + pess,
        "Base (critical) load": np.asarray(results["crit_total"]),
    })
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("kWh:Q", title="kWh"),
            color=alt.Color("series:N", title=""),
            tooltip=["series", "hour", alt.Tooltip("kWh", format=".2f")],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)

    # ---- convergence across rounds ----
    st.subheader("Convergence of the game")
    conv = pd.DataFrame({
        "round": list(range(1, R + 1)),
        "shared energy (kWh)": results["shared_energy_by_round"],
        "mean user cost (€)": [float(np.mean(r)) for r in results["fval"]],
    })
    cc1, cc2 = st.columns(2)
    with cc1:
        st.altair_chart(
            alt.Chart(conv).mark_line(point=True).encode(
                x="round:O", y=alt.Y("shared energy (kWh):Q")
            ).properties(height=240),
            use_container_width=True,
        )
    with cc2:
        st.altair_chart(
            alt.Chart(conv).mark_line(point=True, color="#e45756").encode(
                x="round:O", y=alt.Y("mean user cost (€):Q")
            ).properties(height=240),
            use_container_width=True,
        )

    # ---- battery ----
    if results["params"]["ess_enabled"] and results["soc"]:
        st.subheader("Shared battery")
        bdf = pd.DataFrame({
            "hour": np.arange(24),
            "power (kW)": pess,           # >0 discharge, <0 charge
            "SOC (%)": results["soc"],
        })
        bc1, bc2 = st.columns(2)
        with bc1:
            st.altair_chart(
                alt.Chart(bdf).mark_bar().encode(
                    x="hour:O",
                    y=alt.Y("power (kW):Q", title="ESS power (＋discharge / −charge)"),
                    color=alt.condition("datum['power (kW)'] > 0",
                                        alt.value("#54a24b"), alt.value("#4c78a8")),
                ).properties(height=240),
                use_container_width=True,
            )
        with bc2:
            st.altair_chart(
                alt.Chart(bdf).mark_line(point=True).encode(
                    x="hour:O", y=alt.Y("SOC (%):Q", scale=alt.Scale(domain=[0, 100])),
                ).properties(height=240),
                use_container_width=True,
            )

    # ---- per-user stability (the per-user figures at the end of general.m) ----
    st.subheader("Per-household consumption across rounds")
    n_users = results["n_users"]
    uix = st.selectbox("Household", list(range(1, n_users + 1))) - 1
    rows = []
    for r, mat in enumerate(results["prof_ec_user"]):
        label = "initial" if r == 0 else f"round {r}"
        for h, v in enumerate(mat[uix]):
            rows.append({"hour": h, "kWh": float(v), "iteration": label})
    udf = pd.DataFrame(rows)
    st.altair_chart(
        alt.Chart(udf).mark_line().encode(
            x="hour:O", y="kWh:Q", color=alt.Color("iteration:N", title=""),
            tooltip=["iteration", "hour", alt.Tooltip("kWh", format=".2f")],
        ).properties(height=280),
        use_container_width=True,
    )

    with st.expander("Raw results (JSON)"):
        st.json({k: v for k, v in results.items() if k != "prof_ec_user"})
