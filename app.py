# app.py — FULL VERSION (Chat GPT + Form + Forward + Inverse + Hybrid B2 Curve + Optional Doping)

import os, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from qd_pipeline import QDSystem

# GPT
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except:
        client = None

# UI
st.set_page_config(page_title="QD Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Quantum Dot Band Gap Predictor")

HERE = Path(__file__).parent
MODEL_PATH = HERE/"qd_system.joblib"
DATA_PATH = HERE/"qd_data.csv"

# ---------------- Load / Train ----------------
@st.cache_resource(show_spinner=True)
def load_or_train():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    sys = QDSystem().fit(df)
    joblib.dump(sys, MODEL_PATH)
    return sys

system = load_or_train()

# =====================================================
# 1) CHAT GPT INPUT
# =====================================================
st.subheader("1) Chat-style Input")

left, right = st.columns([0.75, 0.25])

with left:
    chat_text = st.text_input("Example: 'cdse 3 nm er=9.5 wz, n-type, SI 2e18'")

with right:
    use_gpt = st.toggle("Use GPT", value=bool(client))

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
    if not t: return out

    m = re.search(r"\b([A-Z][a-z]?[A-Za-z0-9]{0,6})\b", t)
    if m: out["material"] = m.group(1)

    m = re.search(r"(\d+(\.\d+)?)\s*nm", t)
    if m: out["radius_nm"] = float(m.group(1))

    m = re.search(r"(?:er|epsilon)\s*=?\s*(\d+(\.\d+)?)", t, re.I)
    if m: out["epsilon_r"] = float(m.group(1))

    if re.search(r"\bwz\b", t, re.I): out["crystal_structure"]="WZ"
    if re.search(r"\bzb\b", t, re.I): out["crystal_structure"]="ZB"

    if re.search(r"n-type", t, re.I): out["doping_type"]="n"
    if re.search(r"p-type", t, re.I): out["doping_type"]="p"

    m = re.search(r"dop(?:ed|ing)?\s*with\s*([A-Za-z]{1,2})", t, re.I)
    if m: out["dopant"]=m.group(1).capitalize()

    m = re.search(r"(\d+(\.\d+)?e[+-]?\d+)", t)
    if m: out["doping_conc_cm3"]=float(m.group(1))

    return out

if chat_text:
    if use_gpt and client:
        prompt = f"""
Extract JSON:
material, radius_nm, epsilon_r, crystal_structure, dopant, doping_type, doping_conc_cm3
from: "{chat_text}"
"""
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0
            )
            content = r.choices[0].message.content.strip()
            content = re.sub(r"^```json|```$","",content).strip()
            parsed = json.loads(content)
        except:
            parsed = parse_local(chat_text)
    else:
        parsed = parse_local(chat_text)
else:
    parsed = {k:None for k in ["material","radius_nm","epsilon_r","crystal_structure","dopant","doping_type","doping_conc_cm3"]}

st.markdown("---")

# =====================================================
# 2) STRUCTURED INPUT FORM
# =====================================================
st.subheader("2) Structured Input")

c1, c2 = st.columns(2)

with c1:
    material = st.text_input("Material:", value=(parsed["material"] or "CdSe"))
    radius = st.number_input("Radius (nm):", 0.3, 30.0, float(parsed["radius_nm"] or 3.0), 0.1)

with c2:
    epsr = st.number_input("Dielectric Constant:", 1.0, 100.0, float(parsed["epsilon_r"] or 9.5), 0.1)
    crystal = st.selectbox("Crystal:", ["ZB","WZ","Rocksalt","Perovskite","Rutile"],
                           index=["ZB","WZ","Rocksalt","Perovskite","Rutile"].index(parsed["crystal_structure"] or "ZB"))

# DOPING
st.subheader("Doping (Optional)")

d1,d2,d3 = st.columns([0.4,0.3,0.3])
with d1:
    dopant = st.text_input("Dopant:", value=(parsed["dopant"] or ""))
with d2:
    dtype = st.selectbox("Type:", ["","n","p"], index=["","n","p"].index(parsed["doping_type"] or ""))
with d3:
    dconc = st.number_input("Conc (cm⁻³):", 0.0, 1e22,
                            float(parsed["doping_conc_cm3"] or 0.0),
                            format="%.3e")

st.markdown("---")

# =====================================================
# 3) FORWARD PREDICTION
# =====================================================
st.subheader("3) Forward Prediction")

if st.button("Predict Band Gap"):
    try:
        Eg = system.predict_forward(material, radius, epsr, crystal)
        a,b,c = st.columns(3)
        a.metric("Predicted Eg", f"{Eg:.3f} eV")
        b.metric("Radius", f"{radius:.2f}")
        c.metric("εr / Str", f"{epsr}/{crystal}")

        if dopant or dtype or dconc:
            st.caption(f"Doping: {dopant or '—'} / {dtype or '—'} / {dconc:.2e}")
    except Exception as e:
        st.error(str(e))

st.markdown("---")

# =====================================================
# 4) INVERSE
# =====================================================
st.subheader("4) Inverse Suggestions")

ic1, ic2 = st.columns([1,1])
with ic2:
    targetEg = st.number_input("Target Eg (eV):", 0.0, 10.0, 2.2, 0.05)
with ic1:
    run_inv = st.button("Suggest Materials")

if run_inv:
    try:
        topk, nn = system.predict_inverse(targetEg, radius, epsr, crystal)
        for rank,(name,score) in enumerate(topk,start=1):
            st.write(f"**#{rank} — {name}**  (score {score*100:.1f}%)")

        st.write("Nearest Matches:")
        st.dataframe(nn.reset_index(drop=True))
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
        ax.plot(R, Eg_ml, label="ML", linewidth=2)
        ax.plot(R, Eg_phys, label="Brus Phys", linestyle="--")
        ax.plot(R, Eg_hybrid, label="Hybrid B2", linewidth=3)
        ax.set_xlabel("Radius (nm)")
        ax.set_ylabel("Eg (eV)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(str(e))
