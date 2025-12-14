# app.py — FULL VERSION
# Chat GPT + Structured Input + Forward ML + Inverse (clean) + Hybrid B2 Curve

import os, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from qd_pipeline import QDSystem

# ===========================
# GPT
# ===========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None


# ===========================
# UI
# ===========================
st.set_page_config(page_title="QD Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Quantum Dot Band Gap Predictor")

HERE = Path(__file__).parent
MODEL_PATH = HERE / "qd_system.joblib"
DATA_PATH = HERE / "qd_data.csv"


# ===========================
# Load / Train
# ===========================
@st.cache_resource(show_spinner=True)
def load_or_train():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    sys = QDSystem().fit(df)
    joblib.dump(sys, MODEL_PATH)
    return sys

system = load_or_train()


# ===========================
# Helpers
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
        "doping_conc_cm3": None
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

left, right = st.columns([0.75, 0.25])
with left:
    chat_text = st.text_input("Example: 'cdse 3 nm er=9.5 zb'")
with right:
    use_gpt = st.toggle("Use GPT", value=bool(client))

if chat_text:
    if use_gpt and client:
        prompt = f"""
Return ONLY valid JSON with keys:
material, radius_nm, epsilon_r, crystal_structure,
dopant, doping_type, doping_conc_cm3.
Use null if missing.

Text: "{chat_text}"
"""
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(r.choices[0].message.content)
        except Exception:
            parsed = parse_local(chat_text)
    else:
        parsed = parse_local(chat_text)

    parsed["material"] = normalize_material(parsed.get("material"))
else:
    parsed = {k: None for k in ["material","radius_nm","epsilon_r","crystal_structure","dopant","doping_type","doping_conc_cm3"]}

st.markdown("---")


# =====================================================
# 2) STRUCTURED INPUT
# =====================================================
st.subheader("2) Structured Input")

c1, c2 = st.columns(2)

with c1:
    material = st.text_input("Material:", value=(parsed.get("material") or "CdSe"))
    radius = st.number_input("Radius (nm):", 0.3, 30.0, float(parsed.get("radius_nm") or 3.0), 0.1)

with c2:
    epsr = st.number_input("Dielectric Constant:", 1.0, 100.0, float(parsed.get("epsilon_r") or 9.5), 0.1)
    crystal = st.selectbox(
        "Crystal:",
        ["ZB","WZ","Rocksalt","Perovskite","Rutile"],
        index=["ZB","WZ","Rocksalt","Perovskite","Rutile"].index(parsed.get("crystal_structure") or "ZB")
    )

st.markdown("---")


# =====================================================
# 3) FORWARD PREDICTION (ML)
# =====================================================
st.subheader("3) Forward Prediction")

if st.button("Predict Band Gap"):
    try:
        Eg = system.predict_forward(material, radius, epsr, crystal)
        a, b, c = st.columns(3)
        a.metric("Predicted Eg", f"{Eg:.3f} eV")
        b.metric("Radius", f"{radius:.2f} nm")
        c.metric("εr / Structure", f"{epsr} / {crystal}")
    except Exception as e:
        st.error(str(e))

st.markdown("---")


# =====================================================
# 4) INVERSE SUGGESTIONS (CLEAN)
# =====================================================
st.subheader("4) Inverse Suggestions")

ic1, ic2 = st.columns([1, 1])
with ic2:
    targetEg = st.number_input("Target Eg (eV):", 0.0, 10.0, 2.2, 0.05)
with ic1:
    run_inv = st.button("Suggest Materials")

if run_inv:
    try:
        topk, _ = system.predict_inverse(targetEg, radius, epsr, crystal)

        for rank, (name, score) in enumerate(topk, start=1):
            st.write(f"**#{rank} — {name}**  (confidence {score*100:.1f}%)")

    except Exception as e:
        st.error(str(e))

st.markdown("---")


# =====================================================
# 5) HYBRID B2 CURVE
# =====================================================
st.subheader("5) Band Gap vs Radius Curve (Hybrid B2)")

if st.button("Generate Curve"):
    try:
        R, Eg_ml, Eg_phys, Eg_hybrid = system.hybrid_curve(material, epsr, crystal)

        fig, ax = plt.subplots()
        ax.plot(R, Eg_ml, label="ML Prediction", linewidth=2)
        ax.plot(R, Eg_phys, label="Brus Physical Model", linestyle="--")
        ax.plot(R, Eg_hybrid, label="Hybrid B2", linewidth=3)

        ax.set_xlabel("Radius (nm)")
        ax.set_ylabel("Band Gap (eV)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(str(e))
