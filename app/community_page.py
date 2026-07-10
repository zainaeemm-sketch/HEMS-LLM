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

from pathlib import Path

from optimization.ec_optimizer import (
    ECParams,
    run_ec_game,
    make_synthetic_community,
    load_profiles_from_mat,
    load_profiles_from_csv,
    load_generation_from_mat,
    load_any_mat,
    load_bundled_dataset,
    profiles_csv_template,
    UserProfile,
)

_DATA_DIR = Path(__file__).parent / "data"
_BUNDLED = (_DATA_DIR / "prof30user_load_identify.mat").exists()



def _hours_df(series: dict) -> pd.DataFrame:
    rows = []
    for name, vals in series.items():
        for h, v in enumerate(vals):
            rows.append({"hour": h, "kWh": float(v), "series": name})
    return pd.DataFrame(rows)


def render_community_page(params: dict | None = None):
    st.header("🏘️ Energy Community Optimization")
    st.caption(
        "Distributed game: each household's dashboard optimizes ONLY its own "
        "3 shiftable + 1 adjustable loads, seeing just the aggregated community "
        "profile received over the communication link. After optimizing, it "
        "broadcasts the updated aggregate back. Households keep re-optimizing "
        "until no one can improve unilaterally — a Nash equilibrium. "
        "Port of the MATLAB `general.m` / `ECcost.m` / `ECcostESS.m` model."
    )

    params = params or {}
    pv_forecast = params.get("pv_forecast")

    # ---------------- data source (real profiles vs synthetic) ----------------
    if _BUNDLED and "ec_users" not in st.session_state:
        try:
            _u, _gs, _w = load_bundled_dataset(_DATA_DIR)
            st.session_state["ec_users"] = _u
            st.session_state["ec_mat_gensolar"] = (_gs.tolist() if _gs is not None else None)
            st.session_state["ec_mat_wind"] = (_w.tolist() if _w is not None else None)
            st.session_state["ec_data_label"] = "bundled 30-household measured dataset"
        except Exception:
            pass

    with st.expander("📂 Household load profiles (data source)", expanded=False):
        st.markdown(
            "By default the app loads the **bundled real 30-household dataset** "
            "(`prof30user_load_identify.mat` + `gensolar.mat` + `genwind.mat`). "
            "You can also upload a different MATLAB `.mat` file "
            "(`profh1..profhN`, each 6×24: total / critical / shiftable 1–3 / "
            "adjustable, optionally `Gensolar`) or a CSV in the template format. "
            "If neither is available, a synthetic community with the same "
            "6-row structure is generated."
        )
        ups = st.file_uploader(
            "Upload data files (.mat or .csv) — you can drop several at once, "
            "e.g. prof30user_load_identify.mat + gensolar.mat + genwind.mat",
            type=["mat", "csv"], accept_multiple_files=True)
        if ups:
            loaded_bits = []
            for up in ups:
                try:
                    if up.name.lower().endswith(".csv"):
                        st.session_state["ec_users"] = load_profiles_from_csv(up)
                        loaded_bits.append(
                            f"{len(st.session_state['ec_users'])} household "
                            f"profiles ({up.name})")
                    else:
                        found = load_any_mat(up, up.name)
                        if "users" in found:
                            st.session_state["ec_users"] = found["users"]
                            loaded_bits.append(
                                f"{len(found['users'])} household profiles "
                                f"({up.name})")
                        if "gensolar" in found:
                            st.session_state["ec_mat_gensolar"] = found["gensolar"].tolist()
                            loaded_bits.append(f"solar generation ({up.name})")
                        if "wind" in found:
                            st.session_state["ec_mat_wind"] = found["wind"].tolist()
                            loaded_bits.append(f"wind generation ({up.name})")
                except Exception as e:
                    st.error(f"Could not read {up.name}: {e}")
            if loaded_bits:
                st.session_state["ec_data_label"] = "uploaded: " + ", ".join(
                    u.name for u in ups)
                st.success("Loaded " + " + ".join(loaded_bits))
        if st.session_state.get("ec_users"):
            lbl = st.session_state.get("ec_data_label", "uploaded profiles")
            st.info(f"Using **{len(st.session_state['ec_users'])} households** — {lbl}.")
            if st.button("Discard loaded profiles (use synthetic instead)"):
                for k in ("ec_users", "ec_mat_gensolar", "ec_mat_wind", "ec_data_label"):
                    st.session_state.pop(k, None)
                st.rerun()
        st.download_button(
            "⬇️ Download CSV template (pre-filled with the synthetic community)",
            profiles_csv_template(10).to_csv(index=False).encode(),
            file_name="ec_profiles_template.csv", mime="text/csv",
        )

    # ---------------- configuration ----------------
    with st.expander("⚙️ Community & market parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            _loaded = st.session_state.get("ec_users")
            _max_u = len(_loaded) if _loaded else 30
            n_users = st.number_input(
                "Number of households", 2, _max_u, min(10, _max_u),
                help=(f"A loaded dataset provides {_max_u} households; the "
                      f"first N are used (general.m used Nuser=10)." if _loaded
                      else "Size of the synthetic community."))
            rounds = st.number_input("Max rounds (safety cap)", 1, 30, 10)
            tol_eur = st.number_input(
                "Nash tolerance (€/day)", 0.0, 5.0, 0.05, 0.01,
                help="Equilibrium is declared when NO household improves its "
                     "daily cost by more than this by re-optimizing.")
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
            _has_wind = bool(st.session_state.get("ec_mat_wind"))
            eff_wind = st.number_input(
                "Effwind (0 = wind off, as in general.m)", 0.0, 5.0,
                0.0, 0.1, disabled=not _has_wind,
                help="Scaling of the genwind.mat profile added to generation.")
            update_mode = st.radio(
                "Aggregate broadcast", ["sequential", "simultaneous"],
                horizontal=True,
                help="sequential = each household broadcasts the updated "
                     "aggregate immediately (real protocol, converges to Nash). "
                     "simultaneous = exact MATLAB general.m parity (can oscillate).")

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
        uploaded_users = st.session_state.get("ec_users")
        if uploaded_users:
            uploaded_users = uploaded_users[: int(n_users)]
        p = ECParams(
            n_users=int(n_users), rounds=int(rounds), seed=int(seed),
            converge=True, tol_eur=float(tol_eur), update_mode=str(update_mode),
            eff_wind=float(eff_wind),
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
        elif st.session_state.get("ec_mat_gensolar"):
            gensolar = np.asarray(st.session_state["ec_mat_gensolar"], dtype=float)

        users = uploaded_users or make_synthetic_community(p.n_users, seed=p.seed or 42)
        genwind = (np.asarray(st.session_state["ec_mat_wind"], dtype=float)
                   if st.session_state.get("ec_mat_wind") else None)

        bar = st.progress(0.0, text="Starting community game…")

        def _cb(frac, msg):
            bar.progress(min(frac, 1.0), text=msg)

        with st.spinner("Running the community game…"):
            results = run_ec_game(users=users, gensolar=gensolar,
                                  genwind=genwind, params=p, progress_cb=_cb)
        bar.empty()
        st.session_state["ec_results"] = results
        st.success("✅ Community optimization complete.")

    # ---------------- results ----------------
    results = st.session_state.get("ec_results")
    if not results:
        st.info("Configure the community above and press **Run**.")
        return

    R = results["rounds"]
    best_r = results.get("best_round", R)
    if results.get("converged"):
        st.success(
            f"🎯 **Nash equilibrium reached at round {results['converged_round']}** — "
            f"no household can further improve its cost by changing its load "
            f"profile unilaterally (tolerance in effect). "
            f"**Recommended solution: round {best_r}** (highest shared energy — "
            f"the best round is kept even if it is not the last)."
        )
    else:
        st.warning(
            f"Game stopped at the max-round cap ({results.get('max_rounds', R)}) "
            f"without formally reaching equilibrium — increase the cap or the "
            f"tolerance, or use the *sequential* broadcast mode."
        )
    gensolar = np.asarray(results["gensolar"])
    prof_ttl = [np.asarray(v) for v in results["prof_ec_ttl"]]
    pess = np.asarray(results["pess"])

    # KPIs — all "after" figures refer to the RECOMMENDED (best) round
    s0 = results["shared_energy_initial"]
    s_best = results.get("shared_energy_best",
                         results["shared_energy_by_round"][-1])
    s_final = results["shared_energy_by_round"][-1]
    prof_best = np.asarray(results.get("prof_best", results["prof_ec_ttl"][-1]))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Shared energy — before", f"{s0:.1f} kWh")
    k2.metric(f"Shared energy — recommended (round {best_r})",
              f"{s_best:.1f} kWh", f"{(s_best - s0):+.1f} kWh")
    k3.metric("Shared energy — last round", f"{s_final:.1f} kWh",
              f"{(s_final - s0):+.1f} kWh")
    k4.metric("Community peak (recommended)",
              f"{prof_best.max():.2f} kW",
              f"{(prof_best.max() - prof_ttl[0].max()):+.2f} vs initial")

    # ---- community profile chart (the main figure of general.m) ----
    st.subheader("Community load vs generation")
    df = _hours_df({
        "EC total — initial": prof_ttl[0],
        f"EC total — recommended (round {best_r})": prof_best,
        "Generation + ESS (recommended)": gensolar + pess,
        "Generation only": gensolar,
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
        "households that changed schedule":
            results.get("changes_per_round", [None] * R),
    })
    st.caption("Households changing schedule per round: "
               + " → ".join(str(c) for c in results.get("changes_per_round", []))
               + "  (equilibrium = a round with 0 changes)")
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
        st.caption(f"Battery schedule of the recommended solution (round {best_r}); "
                   "＋ = discharge, − = charge.")
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
