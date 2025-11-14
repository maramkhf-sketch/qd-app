# app.py — Quantum Dot Band Gap Predictor (No JSON Viewer + Clean UI)

# --- Bootstrap: ensure light deps only ---
import sys, subprocess, importlib
def ensure(pkg, pip_spec=None):
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_spec or pkg])
        importlib.import_module(pkg)

ensure("joblib", "joblib>=1.3")
ensure("sklearn", "scikit-learn>=1.4")

# --- std imports ---
import os, re, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# your ML system
from qd_pipeline import QDSystem


# ============== OpenAI (optional for Chat parsing) ==============
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None


# ---------------- UI SETUP & STYLES ----------------
st.set_page_config(page_title="Quantum Dot Band Gap Predictor", page_icon="🧠", layout="centered")

st.markdown("""
<style>
.section-title { font-size: 1.35rem; font-weight: 700; margin: 1.0rem 0 0.5rem; }
.big-title     { font-size: 1.55rem; font-weight: 800; text-align:center; margin: 1.2rem 0 0.8rem; }
.inv-card { border: 1px solid rgba(255,255,255,0.1); padding: 0.8rem 1rem; border-radius: 10px; margin-bottom: 0.6rem; }
.inv-row { display:flex; align-items:center; gap: 12px; }
.inv-rank { width:32px; height:32px; border-radius:8px; background:rgba(255,255,255,0.08); display:flex; align-items:center; justify-content:center; font-weight:700; }
.inv-name { font-weight:700; font-size: 1.05rem; }
.inv-score-wrap { flex:1; }
.inv-bar { height:8px; border-radius:6px; background:rgba(255,255,255,0.1); overflow:hidden; margin-top:6px; }
.inv-bar > div { height:100%; background:linear-gradient(90deg, #3b82f6, #22c55e); }
hr { margin: 1.0rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Quantum Dot Band Gap Predictor")

HERE = Path(__file__).parent.resolve()
MODEL_PATH = HERE / "qd_system.joblib"
DATA_CSV = HERE / "qd_data.csv"


# ---------------- Helpers ----------------

CRYSTAL_ALIASES = {
    "zb": "ZB", "zincblende": "ZB", "fcc": "ZB",
    "wz": "WZ", "wurtzite": "WZ", "hex": "WZ",
    "rs": "Rocksalt", "rocksalt": "Rocksalt", "nacl": "Rocksalt",
    "perovskite": "Perovskite", "pvsk": "Perovskite",
    "rutile": "Rutile", "tio2": "Rutile"
}

def normalize_crystal(s: str | None, default="ZB") -> str:
    if not s: 
        return default
    key = re.sub(r"[^A-Za-z]", "", s).lower()
    return CRYSTAL_ALIASES.get(key, s if s in ["ZB","WZ","Rocksalt","Perovskite","Rutile"] else default)


# -------- Regex fallback parser (with doping) --------
def parse_free_text_regex(text: str):
    out = {
        "material": None,
        "radius_nm": None,
        "epsilon_r": None,
        "crystal_structure": None,
        "dopant": None,
        "doping_type": None,
        "doping_conc_cm3": None
    }
    if not text:
        return out

    m = re.search(r"\b([A-Z][a-z]?[A-Za-z0-9]{0,6}(?:[A-Z][a-z]?[A-Za-z0-9]{0,6})*)\b", text)
    if m: out["material"] = m.group(1)

    m = re.search(r"(\d+(?:\.\d+)?)\s*(nm|nanometer|nano)\b", text)
    if m: out["radius_nm"] = float(m.group(1))

    m = re.search(r"(?:εr|epsilon|er)\s*[:=]?\s*(\d+(?:\.\d+)?)", text)
    if m: out["epsilon_r"] = float(m.group(1))

    for key in (set(CRYSTAL_ALIASES.keys()) | {"ZB","WZ","Rocksalt","Perovskite","Rutile"}):
        if re.search(rf"\b{key}\b", text, re.IGNORECASE):
            out["crystal_structure"] = normalize_crystal(key)
            break

    if re.search(r"\bn[\s-]*type\b", text, re.IGNORECASE):
        out["doping_type"] = "n"
    if re.search(r"\bp[\s-]*type\b", text, re.IGNORECASE):
        out["doping_type"] = "p"

    m = re.search(r"dop(?:ed|ing)?\s*(?:with)?\s*([A-Za-z]{1,2}[a-z]?)", text)
    if m: out["dopant"] = m.group(1).capitalize()

    m = re.search(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", text)
    if m:
        val = float(m.group(1))
        if 1e14 <= val <= 1e22:
            out["doping_conc_cm3"] = val

    return out



# ---------------- Load/Train system ----------------
@st.cache_resource(show_spinner=True)
def load_or_train():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_CSV)
    sys = QDSystem()
    sys.fit(df)
    joblib.dump(sys, MODEL_PATH)
    return sys

system = load_or_train()



# ======================= 1) CHAT INPUT =======================
st.markdown('<div class="section-title">1) Chat-style Input</div>', unsafe_allow_html=True)

left, right = st.columns([0.75, 0.25])

with left:
    chat_text = st.text_input("Example: 'cdse 3 nm er=9.5 wz, n-type, Si 2e18'")

with right:
    use_gpt = st.toggle("Use GPT to parse", value=bool(OPENAI_API_KEY))


parsed = {"material": None, "radius_nm": None, "epsilon_r": None,
          "crystal_structure": None, "dopant": None,
          "doping_type": None, "doping_conc_cm3": None}


if chat_text:
    if use_gpt and client:
        prompt = f"""
Extract the quantum-dot parameters from:
"{chat_text}"

Return ONLY JSON with:
material, radius_nm, epsilon_r, crystal_structure,
dopant, doping_type, doping_conc_cm3
"""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content": prompt}],
                temperature=0
            )
            content = resp.choices[0].message.content.strip()
            content = re.sub(r"^```json|```$", "", content).strip()
            parsed = json.loads(content)
        except:
            parsed = parse_free_text_regex(chat_text)
    else:
        parsed = parse_free_text_regex(chat_text)


st.markdown("<hr/>", unsafe_allow_html=True)



# ======================= 2) FORM INPUT =======================
st.markdown('<div class="section-title">2) Structured Input</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    material = st.text_input("Material", value=(parsed["material"] or "CdSe"))
    radius_nm = st.number_input("Radius (nm)", min_value=0.3, max_value=30.0,
                                value=float(parsed["radius_nm"] or 3.0), step=0.1)

with c2:
    epsilon_r = st.number_input("Dielectric Constant (εr)", min_value=1.0, max_value=100.0,
                                value=float(parsed["epsilon_r"] or 9.5), step=0.1)
    crystal_structure = st.selectbox(
        "Crystal Structure",
        ["ZB","WZ","Rocksalt","Perovskite","Rutile"],
        index=["ZB","WZ","Rocksalt","Perovskite","Rutile"].index(parsed["crystal_structure"] or "ZB")
    )

# ===== Doping =====
st.markdown('<div class="section-title">Doping (optional)</div>', unsafe_allow_html=True)

d1, d2, d3 = st.columns([0.5, 0.25, 0.25])

with d1:
    dopant = st.text_input("Dopant", value=(parsed["dopant"] or ""))

with d2:
    doping_type = st.selectbox("Type", ["", "n", "p"],
                               index=["","n","p"].index(parsed["doping_type"] or ""))

with d3:
    doping_conc_cm3 = st.number_input(
        "Conc. (cm⁻³)", min_value=0.0, max_value=1e22,
        value=float(parsed["doping_conc_cm3"] or 0.0),
        step=1e16, format="%.3e"
    )


st.markdown("<hr/>", unsafe_allow_html=True)



# ======================= 3) FORWARD =======================
st.markdown('<div class="section-title">3) Forward Prediction</div>', unsafe_allow_html=True)

if st.button("Predict Eg"):
    try:
        eg = system.predict_bandgap(material, radius_nm, epsilon_r, crystal_structure)

        a,b,c = st.columns(3)
        a.metric("Predicted Eg", f"{eg:.3f} eV")
        b.metric("Radius", f"{radius_nm:.2f} nm")
        c.metric("εr / Structure", f"{epsilon_r} / {crystal_structure}")

        if dopant or doping_type or doping_conc_cm3:
            st.caption(f"Doping: {dopant or '—'} / {doping_type or '—'} / {doping_conc_cm3:.2e} cm⁻³")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.markdown("<hr/>", unsafe_allow_html=True)



# ======================= 4) INVERSE =======================
st.markdown('<div class="big-title">Inverse Suggestions (Top-K)</div>', unsafe_allow_html=True)

ic1, ic2 = st.columns([1,1])

with ic2:
    target = st.number_input("Target Eg (eV)", min_value=0.0, max_value=10.0,
                             value=2.2, step=0.05)

with ic1:
    st.write("")
    run_inverse = st.button("Suggest Materials")

if run_inverse:
    try:
        topk, nn = system.suggest_materials_from_bandgap(
            target, radius_nm, epsilon_r, crystal_structure, top_k=3
        )

        if len(topk)==0:
            st.info("No suggestions.")
        else:
            for rank, (name, score) in enumerate(topk, start=1):
                pct = max(0, min(1, score))*100
                st.markdown(
f"""
<div class="inv-card">
  <div class="inv-row">
    <div class="inv-rank">{rank}</div>
    <div class="inv-name">{name}</div>
    <div class="inv-score-wrap">
      <div style="font-size:0.85rem;opacity:0.8;">Score: {pct:.1f}%</div>
      <div class="inv-bar"><div style="width:{pct:.1f}%;"></div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Nearest Matches</div>', unsafe_allow_html=True)

        cols = ["material","band_gap_eV","radius_nm","epsilon_r","crystal_structure"]
        show_df = nn[cols].copy()
        show_df.columns = ["Material","Eg (eV)","R (nm)","εr","Crystal"]
        show_df["Eg (eV)"] = show_df["Eg (eV)"].round(3)

        st.dataframe(show_df.reset_index(drop=True), use_container_width=True)

    except Exception as e:
        st.error(f"Inverse prediction failed: {e}")



# ======================= 5) CURVE =======================
st.markdown('<div class="section-title">5) Band Gap vs Radius Curve</div>', unsafe_allow_html=True)

if st.button("Generate Curve"):
    try:
        R = np.linspace(1, 10, 40)
        EG = [system.predict_bandgap(material, float(r), epsilon_r, crystal_structure) for r in R]

        fig, ax = plt.subplots()
        ax.plot(R, EG, linewidth=2)
        ax.set_xlabel("Radius (nm)")
        ax.set_ylabel("Band Gap (eV)")
        ax.set_title(f"Eg vs Radius — {material}")
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Plot failed: {e}")



st.caption("GPT parsing optional. Doping stored as metadata only.")
