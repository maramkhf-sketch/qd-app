# app.py — FINAL VERSION
# Forward ML + Inverse + Hybrid Curve
# Doping inputs are STRUCTURED + optional (reference only)

import os, re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from qd_pipeline import QDSystem

# ===========================
# UI CONFIG
# ===========================
st.set_page_config(page_title="QD Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Quantum Dot Band Gap Predictor")

HERE = Path(__file__).parent
MODEL_PATH = HERE / "qd_system.joblib"
DATA_PATH = HERE / "qd_data.csv"

# ===========================
# LOAD / TRAIN SYSTEM
# ===========================
@st.cache_resource(show_spinner=True)
def load_or_train():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    system = QDSystem().fit(df)
    joblib.dump(system, MODEL_PATH)
    return system

system = load_or_train()

# ===========================
# HELPERS
# ===========================
def normalize_material(x):
    if not x:
        return None
    s = str(x).strip()
    mapping = {
        "cdse": "CdSe",
        "cdte": "CdTe",
        "zno": "ZnO",
        "ws2": "WS2",
        "mos2": "MoS2",
        "wse2": "WSe2",
        "mose2": "MoSe2",
    }
    return mapping.get(s.lower(), s)

def parse_local(t):
    out = {
        "material": None,
        "radius_nm": None,
        "epsilon_r": None,
        "crystal_structure": None,
        "dopant": None,
        "doping_type": None,
        "doping_conc_cm3": None,
    }
    if not t:
        return out

    m = re.search(r"\b([A-Za-z][A-Za-z0-9]{1,8})\b", t)
    if m:
        out["material"] = normalize_material(m.group(1))

    m = re.search(r"(\d+(\.\d+)?)\s*nm", t, re.I)
    if m:
        out["radius_nm"] = float(m.group(1))

    m = re.search(r"(?:er|epsilon)\s*=?\s*(\d+(\.\d+)?)", t, re.I)
    if m:
        out["epsilon_r"] = float(m.group(1))

    if re.search(r"\bwz\b", t, re.I):
        out["crystal_structure"] = "WZ"
    if re.search(r"\bzb\b", t, re.I):
        out["crystal_structure"] = "ZB"

    if re.search(r"n-type", t, re.I):
        out["doping_type"] = "n"
    if re.search(r"p-type", t, re.I):
        out["doping_type"] = "p"

    m = re.search(r"dop(?:ed|ing)?\s*with\s*([A-Za-z]{1,2})", t, re.I)
    if m:
        out["dopant"] = m.group(1).capitalize()

    m = re.search(r"(\d+(\.\d+)?e[+-]?\d+)", t, re.I)
    if m:
        out["doping_conc_cm3"] = float(m.group(1))

    return out

# =====================================================
# 1) CHAT INPUT
# =====================================================
st.subheader("1) Chat-style Input")
chat_text = st.text_input(
    "Example: CdSe 3 nm er=9.5 ZB n-type doped with P"
)

parsed = parse_local(chat_text)
parsed["material"] = normalize_material(parsed.get("material"))

st.markdown("---")

# =====================================================
# 2) STRUCTURED INPUT
# =====================================================
st.subheader("2) Structured Input")

c1, c2 = st.columns(2)

with c1:
    material = st.text_input(
        "Material:",
        value=parsed.get("material") or "CdSe"
    )
    radius = st.number_input(
        "Radius (nm):",
        0.3, 30.0,
        float(parsed.get("radius_nm") or 3.0),
        0.1
    )

with c2:
    epsr = st.number_input(
        "Dielectric Constant (εr):",
        1.0, 100.0,
        float(parsed.get("epsilon_r") or 9.5),
        0.1
    )
    crystal = st.selectbox(
        "Crystal Structure:",
        ["ZB", "WZ"],
        index=["ZB", "WZ"].index(parsed.get("crystal_structure") or "ZB")
    )

# =====================================================
# 3) DOPING INPUTS (STRUCTURED)
# =====================================================
st.subheader("3) Doping Inputs (optional)")

d1, d2, d3 = st.columns(3)

with d1:
    dopant = st.text_input(
        "Dopant (e.g., P, N, B):",
        value=parsed.get("dopant") or ""
    )

with d2:
    doping_type = st.selectbox(
        "Type:",
        ["—", "n", "p"],
        index=["—", "n", "p"].index(
            parsed.get("doping_type")
            if parsed.get("doping_type") in ["n", "p"] else "—"
        )
    )

with d3:
    use_conc = st.checkbox(
        "Add concentration",
        value=parsed.get("doping_conc_cm3") is not None
    )
    doping_conc_cm3 = st.number_input(
        "Concentration (cm⁻³):",
        min_value=0.0,
        value=float(parsed.get("doping_conc_cm3") or 0.0),
        step=1e15,
        format="%.3e",
        disabled=not use_conc
    )

st.caption(
    "ℹ️ Doping inputs are for reference only and are NOT used in the prediction model."
)

st.markdown("---")

# =====================================================
# 4) FORWARD PREDICTION
# =====================================================
st.subheader("4) Forward Prediction (ML)")

if st.button("Predict Band Gap"):
    Eg_ml = system.predict_forward(material, radius, epsr, crystal)
    Eg_brus = system.predict_brus_single(material, radius, epsr, crystal)

    delta = Eg_ml - Eg_brus
    pct = abs(delta) / (abs(Eg_brus) + 1e-12) * 100

    a, b, c = st.columns(3)
    a.metric("ML Predicted Eg", f"{Eg_ml:.3f} eV")
    b.metric("Brus Eg", f"{Eg_brus:.3f} eV")
    c.metric("ΔEg (%)", f"{pct:.2f} %")

st.markdown("---")

# =====================================================
# 5) INVERSE PREDICTION
# =====================================================
st.subheader("5) Inverse Prediction")

targetEg = st.number_input(
    "Target Band Gap (eV):",
    0.0, 10.0, 2.2, 0.05
)

if st.button("Suggest Materials"):
    topk, _ = system.predict_inverse(targetEg, radius, epsr, crystal)
    for i, (name, prob) in enumerate(topk, 1):
        st.write(f"**#{i} — {name}** (confidence {prob*100:.1f}%)")

st.markdown("---")

# =====================================================
# 6) HYBRID CURVE
# =====================================================
st.subheader("6) Band Gap vs Radius (Hybrid Model)")

if st.button("Generate Curve"):
    R, Eg_ml, Eg_phys, Eg_hybrid = system.hybrid_curve(
        material, epsr, crystal
    )

    fig, ax = plt.subplots()
    ax.plot(R, Eg_ml, label="ML Prediction")
    ax.plot(R, Eg_phys, "--", label="Brus Model")
    ax.plot(R, Eg_hybrid, linewidth=3, label="Hybrid Model")

    ax.set_xlabel("Radius (nm)")
    ax.set_ylabel("Band Gap (eV)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
