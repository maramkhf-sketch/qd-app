# app.py — Quantum Dot Band Gap Predictor (GPT Chat + Form + Forward + Inverse + Curve + Doping)

import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from qd_pipeline import QDSystem   # ← مهم

# ==================== OPENAI CLIENT ====================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except:
        client = None

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Quantum Dot Band Gap Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Quantum Dot Band Gap Predictor")

HERE = Path(__file__).parent.resolve()
DATA_FILE = HERE / "qd_data.csv"
MODEL_FILE = HERE / "qd_system.joblib"

# ==================== LOAD / TRAIN MODEL ====================
@st.cache_resource(show_spinner=True)
def load_system():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)

    df = pd.read_csv(DATA_FILE)
    sys = QDSystem()
    sys.fit(df)
    joblib.dump(sys, MODEL_FILE)
    return sys

system = load_system()

# ==================== HELPER (LOCAL PARSER) ====================
def parse_local(text):
    out = {
        "material": None,
        "radius_nm": None,
        "epsilon_r": None,
        "crystal_structure": None,
        "dopant": None,
        "doping_type": None,
        "doping_conc": None
    }

    if not text:
        return out

    m = re.search(r"\b([A-Z][a-zA-Z0-9]*)\b", text)
    if m:
        out["material"] = m.group(1)

    m = re.search(r"(\d+(\.\d+)?)\s*nm", text)
    if m:
        out["radius_nm"] = float(m.group(1))

    m = re.search(r"(?:εr|er|epsilon)\s*=?\s*(\d+(\.\d+)?)", text)
    if m:
        out["epsilon_r"] = float(m.group(1))

    for key in ["ZB", "WZ", "Rocksalt", "Perovskite", "Rutile"]:
        if re.search(key, text, re.IGNORECASE):
            out["crystal_structure"] = key
            break

    if "n-type" in text.lower():
        out["doping_type"] = "n"
    if "p-type" in text.lower():
        out["doping_type"] = "p"

    m = re.search(r"doped with ([A-Za-z]+)", text.lower())
    if m:
        out["dopant"] = m.group(1).capitalize()

    m = re.search(r"(\d+(\.\d+)?(e[+-]?\d+)?)", text.lower())
    if m:
        try:
            val = float(m.group(1))
            if 1e14 <= val <= 1e22:
                out["doping_conc"] = val
        except:
            pass

    return out

# ==================== 1) GPT CHAT ====================
st.subheader("1) GPT Chat Input")

chat_text = st.text_input(
    "Describe the material (e.g., 'CdSe 3 nm er=9.5 WZ n-type doped with Si 2e18')"
)

use_gpt = st.toggle("Use GPT to parse input", value=bool(client))

parsed = {
    "material": None,
    "radius_nm": None,
    "epsilon_r": None,
    "crystal_structure": None,
    "dopant": None,
    "doping_type": None,
    "doping_conc": None
}

if chat_text:
    if use_gpt and client:
        prompt = f"""
Extract the following fields from the text. Return ONLY JSON:
material, radius_nm, epsilon_r, crystal_structure,
dopant, doping_type, doping_conc.

Text: "{chat_text}"
"""

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)
        except:
            parsed = parse_local(chat_text)
    else:
        parsed = parse_local(chat_text)

# ==================== 2) FORM INPUT ====================
st.subheader("2) Structured Input")

c1, c2 = st.columns(2)

with c1:
    material = st.text_input("Material", value=parsed["material"] or "CdSe")
    radius_nm = st.number_input("Radius (nm)", 0.3, 30.0,
                                value=float(parsed["radius_nm"] or 3.0))

with c2:
    epsilon_r = st.number_input("Dielectric Constant (εr)", 1.0, 100.0,
                                value=float(parsed["epsilon_r"] or 9.5))
    crystal_structure = st.selectbox(
        "Crystal Structure",
        ["ZB", "WZ", "Rocksalt", "Perovskite", "Rutile"],
        index=["ZB", "WZ", "Rocksalt", "Perovskite", "Rutile"].index(
            parsed["crystal_structure"] or "ZB"
        )
    )

# -------- DOPING --------
st.subheader("Doping (Optional)")

d1, d2, d3 = st.columns(3)

with d1:
    dopant = st.text_input("Dopant (e.g., B, Si, Mn)", value=parsed["dopant"] or "")

with d2:
    doping_type = st.selectbox(
        "Type", ["", "n", "p"],
        index=["", "n", "p"].index(parsed["doping_type"] or "")
    )

with d3:
    doping_conc = st.number_input(
        "Concentration (cm⁻³)",
        min_value=0.0,
        max_value=1e22,
        value=float(parsed["doping_conc"] or 0.0),
        step=1e16,
        format="%.3e"
    )

# ==================== 3) FORWARD PREDICTION ====================
st.subheader("3) Forward Prediction")

if st.button("Predict Band Gap"):
    eg = system.predict_bandgap(
        material, radius_nm, epsilon_r, crystal_structure,
        dopant=dopant or None,
        doping_type=doping_type or None,
        doping_conc=doping_conc or None
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Eg (eV)", f"{eg:.3f}")
    c2.metric("Radius", f"{radius_nm:.2f} nm")
    c3.metric("εr / Structure", f"{epsilon_r} / {crystal_structure}")

# ==================== 4) INVERSE ====================
st.subheader("4) Inverse Band Gap")

target = st.number_input("Target Eg (eV)", 0.0, 10.0, value=2.2)

if st.button("Suggest Materials"):
    top, df_hits = system.suggest_materials_from_bandgap(
        target, radius_nm, epsilon_r, crystal_structure
    )

    st.write("Top Matches:")
    for name, score in top:
        st.write(f"• **{name}** — {score*100:.1f}%")

    st.write("Nearest Dataset Rows:")
    st.dataframe(df_hits.reset_index(drop=True))

# ==================== 5) CURVE ====================
st.subheader("5) Band Gap vs Radius Curve")

if st.button("Generate Curve"):
    Rvals = np.linspace(1, 10, 40)
    Egvals = [
        system.predict_bandgap(
            material, float(r), epsilon_r, crystal_structure,
            dopant=dopant or None,
            doping_type=doping_type or None,
            doping_conc=doping_conc or None
        )
        for r in Rvals
    ]

    fig, ax = plt.subplots()
    ax.plot(Rvals, Egvals, linewidth=2)
    ax.set_xlabel("Radius (nm)")
    ax.set_ylabel("Eg (eV)")
    ax.grid(True)
    st.pyplot(fig)
