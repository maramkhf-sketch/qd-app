# app.py — Quantum Dot Hybrid ML + Physics (with Doping)

import sys, subprocess, importlib

def ensure(pkg, pip_spec=None):
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_spec or pkg])
        importlib.import_module(pkg)

ensure("joblib", "joblib>=1.3")
ensure("sklearn", "scikit-learn>=1.4")

import os, re, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from qd_pipeline import QDSystem

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Quantum Dot Predictor", page_icon="🧠", layout="centered")

st.markdown("""
<style>
.section-title { font-size:1.35rem; font-weight:700; margin:1rem 0 .5rem; }
.big-title     { font-size:1.55rem; font-weight:800; text-align:center; margin:1.2rem 0 .8rem; }
.inv-card { border:1px solid rgba(255,255,255,0.1); padding:.8rem 1rem; border-radius:10px; margin-bottom:.6rem; }
.inv-row { display:flex; align-items:center; gap:12px; }
.inv-rank { width:32px; height:32px; border-radius:8px; background:rgba(255,255,255,0.1); display:flex; align-items:center; justify-content:center; font-weight:700; }
.inv-name { font-weight:700; font-size:1.05rem; }
.inv-score-wrap { flex:1; }
.inv-bar { height:8px; border-radius:6px; background:rgba(255,255,255,0.1); overflow:hidden; margin-top:6px; }
.inv-bar > div { height:100%; background:linear-gradient(90deg, #3b82f6, #22c55e); }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Quantum Dot Band Gap Predictor")

HERE = Path(__file__).parent.resolve()
MODEL_PATH = HERE / "qd_system.joblib"
DATA_CSV = HERE / "qd_data.csv"

# ---------------- Helpers ----------------
CRYSTAL_ALIASES = {
    "zb":"ZB","zincblende":"ZB","fcc":"ZB",
    "wz":"WZ","wurtzite":"WZ","hex":"WZ",
    "rs":"Rocksalt","rocksalt":"Rocksalt",
    "pvsk":"Perovskite","perovskite":"Perovskite",
    "rutile":"Rutile","tio2":"Rutile"
}

def norm_crystal(s):
    if not s: return "ZB"
    k = re.sub(r"[^A-Za-z]", "", s).lower()
    return CRYSTAL_ALIASES.get(k, s if s in ["ZB","WZ","Rocksalt","Perovskite","Rutile"] else "ZB")

def parse_free(text):
    out = dict(material=None,radius_nm=None,epsilon_r=None,crystal_structure=None,
               dopant=None,doping_type=None,doping_conc_cm3=None)
    if not text: return out

    m = re.search(r"\b([A-Z][a-z]?[A-Za-z0-9]{0,6})\b", text)
    if m: out["material"] = m.group(1)

    m = re.search(r"(\d+(\.\d+)?)\s*nm\b", text)
    if m: out["radius_nm"] = float(m.group(1))

    m = re.search(r"(?:εr|er|epsilon)\s*=?\s*(\d+(\.\d+)?)", text)
    if m: out["epsilon_r"] = float(m.group(1))

    for k in CRYSTAL_ALIASES:
        if re.search(rf"\b{k}\b", text, re.IGNORECASE):
            out["crystal_structure"] = norm_crystal(k)
            break

    if re.search(r"\bn-type\b|\bn\+\b", text, re.IGNORECASE):
        out["doping_type"] = "n"
    if re.search(r"\bp-type\b|\bp\+\b", text, re.IGNORECASE):
        out["doping_type"] = "p"

    m = re.search(r"doped with ([A-Za-z]{1,2})", text, re.IGNORECASE)
    if m: out["dopant"] = m.group(1).capitalize()

    m = re.search(r"(\d+(\.\d+)?e[+-]?\d+)", text, re.IGNORECASE)
    if m:
        conc = float(m.group(1))
        if 1e14 <= conc <= 1e22:
            out["doping_conc_cm3"] = conc

    return out

# ---------------- Load or train ----------------
@st.cache_resource(show_spinner=True)
def load_system():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_CSV)
    sys = QDSystem()
    sys.fit(df)
    joblib.dump(sys, MODEL_PATH)
    return sys

system = load_system()

# ======================= CHAT INPUT =======================
st.markdown('<div class="section-title">1) Chat-style Input</div>', unsafe_allow_html=True)

chat_text = st.text_input("Example: 'CdSe 3 nm er=9.5 wz, n-type, Mn 2e18'")

parsed = dict(material=None,radius_nm=None,epsilon_r=None,
              crystal_structure=None,dopant=None,
              doping_type=None,doping_conc_cm3=None)

if chat_text:
    parsed = parse_free(chat_text)

st.markdown("<hr/>", unsafe_allow_html=True)

# ======================= FORM INPUT =======================
st.markdown('<div class="section-title">2) Structured Input</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    material = st.text_input("Material", value=parsed["material"] or "CdSe")
    radius_nm = st.number_input("Radius (nm)", 0.3, 30.0,
                                value=float(parsed["radius_nm"] or 3.0), step=0.1)
with c2:
    epsilon_r = st.number_input("εr", 1.0, 100.0,
                                value=float(parsed["epsilon_r"] or 9.5), step=0.1)
    crystal_structure = st.selectbox(
        "Structure", ["ZB","WZ","Rocksalt","Perovskite","Rutile"],
        index=["ZB","WZ","Rocksalt","Perovskite","Rutile"].index(
            parsed["crystal_structure"] or "ZB"
        )
    )

# ---- doping UI ----
st.markdown('<div class="section-title">Doping (optional)</div>', unsafe_allow_html=True)
d1, d2, d3 = st.columns([0.4, 0.3, 0.3])

with d1:
    dopant = st.text_input("Dopant", value=parsed["dopant"] or "")
with d2:
    doping_type = st.selectbox("Type", ["","n","p"],
                               index=["","n","p"].index(parsed["doping_type"] or ""))
with d3:
    doping_conc_cm3 = st.number_input(
        "Conc (cm⁻³)", min_value=0.0, max_value=1e22,
        value=float(parsed["doping_conc_cm3"] or 0.0),
        step=1e16, format="%.2e"
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ======================= FORWARD =======================
st.markdown('<div class="section-title">3) Predict Band Gap</div>', unsafe_allow_html=True)

if st.button("Predict Eg"):
    try:
        Eg = system.predict_bandgap(
            material, radius_nm, epsilon_r, crystal_structure,
            dopant=dopant or None,
            doping_type=doping_type or None,
            doping_conc_cm3=(doping_conc_cm3 or None)
        )

        a,b,c = st.columns(3)
        a.metric("Eg", f"{Eg:.3f} eV")
        b.metric("Radius", f"{radius_nm:.2f} nm")
        c.metric("εr / Struct", f"{epsilon_r} / {crystal_structure}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("<hr/>", unsafe_allow_html=True)

# ======================= INVERSE =======================
st.markdown('<div class="big-title">Inverse Suggestions</div>', unsafe_allow_html=True)

target = st.number_input("Target Eg (eV)", 0.0, 10.0, 2.2, 0.05)
if st.button("Suggest Materials"):
    try:
        topk, table = system.suggest_materials_from_bandgap(
            target, radius_nm, epsilon_r, crystal_structure, top_k=3
        )

        for rank,(name,score) in enumerate(topk,1):
            pct = score*100
            st.markdown(
                f"""
                <div class="inv-card">
                  <div class="inv-row">
                    <div class="inv-rank">{rank}</div>
                    <div class="inv-name">{name}</div>
                    <div class="inv-score-wrap">
                      <div style="font-size:0.85rem;opacity:0.8;">
                        Score: {pct:.1f}%
                      </div>
                      <div class="inv-bar"><div style="width:{pct:.1f}%;"></div></div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.dataframe(table[["material","band_gap_eV","radius_nm","epsilon_r","crystal_structure"]],
                     use_container_width=True)

    except Exception as e:
        st.error(f"Inverse failed: {e}")

st.markdown("<hr/>", unsafe_allow_html=True)

# ======================= CURVE =======================
st.markdown('<div class="section-title">5) Brus-like Curve</div>', unsafe_allow_html=True)

if st.button("Generate Curve"):
    R = np.linspace(1, 10, 40)
    EG = [system.predict_bandgap(material, float(r), epsilon_r, crystal_structure,
                                 dopant=dopant or None,
                                 doping_type=doping_type or None,
                                 doping_conc_cm3=doping_conc_cm3 or None)
          for r in R]

    fig,ax = plt.subplots()
    ax.plot(R, EG, linewidth=2)
    ax.set_xlabel("Radius (nm)")
    ax.set_ylabel("Eg (eV)")
    ax.set_title(f"{material} — Eg vs R")
    ax.grid(True)
    st.pyplot(fig)

st.caption("Hybrid ML + physics. Doping optional and applied analytically.")

