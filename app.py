# app.py — Quantum Dot Band Gap Predictor (Hybrid + GPT Chat Popup)

# ============================================================
#                           IMPORTS
# ============================================================

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


# ============================================================
#                       GPT Client Setup
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except:
        client = None


# ============================================================
#                       PAGE CONFIG + STYLES
# ============================================================

st.set_page_config(page_title="Quantum Dot Band Gap Predictor", page_icon="🧠", layout="centered")

st.markdown("""
<style>
.section-title { font-size: 1.35rem; font-weight: 700; margin-top: 1rem; }
.big-title     { font-size: 1.55rem; font-weight: 800; text-align:center; margin: 1.2rem 0; }
.inv-card { border: 1px solid rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 10px; margin-bottom: 0.6rem; }
.inv-row { display:flex; align-items:center; gap: 12px; }
.inv-rank { width:32px; height:32px; border-radius:8px; background:rgba(255,255,255,0.08); display:flex; align-items:center; justify-content:center; font-weight:700; }
.inv-name { font-weight:700; font-size: 1.05rem; }
.inv-bar { height:8px; border-radius:6px; background:rgba(255,255,255,0.12); overflow:hidden; margin-top:6px; }
.inv-bar > div { height:100%; background:linear-gradient(90deg,#3b82f6,#22c55e); }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Quantum Dot Band Gap Predictor")


# ============================================================
#                          PATHS
# ============================================================

HERE = Path(__file__).parent.resolve()
MODEL_PATH = HERE / "qd_system.joblib"
DATA_CSV = HERE / "qd_data.csv"


# ============================================================
#                       Helper Functions
# ============================================================

CRYSTAL_ALIASES = {
    "zb": "ZB", "zincblende": "ZB",
    "wz": "WZ", "wurtzite": "WZ",
    "rs": "Rocksalt", "rocksalt": "Rocksalt",
    "perovskite": "Perovskite",
    "rutile": "Rutile"
}

def normalize_crystal(s, default="ZB"):
    if not s: return default
    s2 = re.sub(r"[^A-Za-z]", "", s).lower()
    return CRYSTAL_ALIASES.get(s2, default)


def parse_free_text_regex(text):
    out = {
        "material": None, "radius_nm": None, "epsilon_r": None,
        "crystal_structure": None, "dopant": None,
        "doping_type": None, "doping_conc_cm3": None
    }

    if not text:
        return out

    m = re.search(r"\b([A-Z][a-z]?[A-Za-z0-9]*)\b", text)
    if m: out["material"] = m.group(1)

    m = re.search(r"(\d+(?:\.\d+)?)\s*nm", text)
    if m: out["radius_nm"] = float(m.group(1))

    m = re.search(r"(?:er|epsilon|εr)\s*=?\s*(\d+(?:\.\d+)?)", text)
    if m: out["epsilon_r"] = float(m.group(1))

    for k in CRYSTAL_ALIASES.keys():
        if re.search(k, text, re.IGNORECASE):
            out["crystal_structure"] = normalize_crystal(k)

    if "n-type" in text.lower():
        out["doping_type"] = "n"
    if "p-type" in text.lower():
        out["doping_type"] = "p"

    m = re.search(r"dop(?:ed|ing)?\s*with\s*([A-Za-z]{1,2})", text)
    if m: out["dopant"] = m.group(1)

    m = re.search(r"(\d+(\.\d+)?e[+-]?\d+)", text)
    if m: out["doping_conc_cm3"] = float(m.group(1))

    return out


# ============================================================
#                     Load or Train Model
# ============================================================

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


# ============================================================
#                   1) CHAT INPUT SECTION
# ============================================================

st.markdown('<div class="section-title">1) Chat-style Input</div>', unsafe_allow_html=True)

lcol, rcol = st.columns([0.75, 0.25])
with lcol:
    chat_text = st.text_input("Example: 'cdse 3 nm er=9.5 wz, n-type, Si 2e18'")
with rcol:
    use_gpt = st.toggle("Use GPT", value=bool(OPENAI_API_KEY))

parsed = {"material":None,"radius_nm":None,"epsilon_r":None,
          "crystal_structure":None,"dopant":None,
          "doping_type":None,"doping_conc_cm3":None}

if chat_text:
    if use_gpt and client:
        prompt = f"""
Extract the QD parameters from:
"{chat_text}"
Return JSON only:
material, radius_nm, epsilon_r, crystal_structure, dopant, doping_type, doping_conc_cm3
        """
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0
            )
            js = resp.choices[0].message.content.strip()
            js = re.sub(r"```json|```","",js).strip()
            parsed = json.loads(js)
        except:
            parsed = parse_free_text_regex(chat_text)
    else:
        parsed = parse_free_text_regex(chat_text)

st.markdown("<hr/>", unsafe_allow_html=True)


# ============================================================
#                   2) STRUCTURED INPUT SECTION
# ============================================================

st.markdown('<div class="section-title">2) Structured Input</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    material = st.text_input("Material", parsed["material"] or "CdSe")
    radius_nm = st.number_input("Radius (nm)", 0.3, 30.0, parsed["radius_nm"] or 3.0)

with c2:
    epsilon_r = st.number_input("Dielectric Constant", 1.0, 100.0, parsed["epsilon_r"] or 9.5)
    crystal_structure = st.selectbox(
        "Crystal Structure", ["ZB","WZ","Rocksalt","Perovskite","Rutile"],
        index=["ZB","WZ","Rocksalt","Perovskite","Rutile"].index(parsed["crystal_structure"] or "ZB")
    )

st.markdown('<div class="section-title">Doping (optional)</div>', unsafe_allow_html=True)
d1,d2,d3 = st.columns([0.5,0.25,0.25])
with d1:
    dopant = st.text_input("Dopant", parsed["dopant"] or "")
with d2:
    doping_type = st.selectbox("Type", ["","n","p"], index=["","n","p"].index(parsed["doping_type"] or ""))
with d3:
    doping_conc_cm3 = st.number_input("Conc (cm⁻³)", 0.0, 1e22, parsed["doping_conc_cm3"] or 0.0, format="%.3e")

st.markdown("<hr/>", unsafe_allow_html=True)


# ============================================================
#                3) FORWARD PREDICTION SECTION
# ============================================================

st.markdown('<div class="section-title">3) Forward Prediction</div>', unsafe_allow_html=True)

if st.button("Predict Eg"):
    try:
        eg = system.predict_bandgap(material, radius_nm, epsilon_r, crystal_structure)

        a,b,c = st.columns(3)
        a.metric("Predicted Eg", f"{eg:.3f} eV")
        b.metric("Radius", f"{radius_nm:.2f} nm")
        c.metric("εr / Structure", f"{epsilon_r} / {crystal_structure}")

        if dopant or doping_type or doping_conc_cm3:
            st.caption(f"Doping: {dopant or '—'} / {doping_type or '—'} / {doping_conc_cm3:.2e}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("<hr/>", unsafe_allow_html=True)


# ============================================================
#                4) INVERSE SUGGESTIONS SECTION
# ============================================================

st.markdown('<div class="big-title">Inverse Suggestions (Top-K)</div>', unsafe_allow_html=True)

i1,i2 = st.columns(2)
with i2:
    target = st.number_input("Target Eg (eV)", 0.0, 10.0, 2.2)
with i1:
    st.write("")
    run_inverse = st.button("Suggest Materials")

if run_inverse:
    try:
        topk, nn = system.suggest_materials_from_bandgap(target, radius_nm, epsilon_r, crystal_structure, 3)

        for rank,(name,score) in enumerate(topk, start=1):
            pct = max(0,min(1,score))*100
            st.markdown(f"""
            <div class="inv-card">
              <div class="inv-row">
                <div class="inv-rank">{rank}</div>
                <div>
                  <div class="inv-name">{name}</div>
                  <div class="inv-bar"><div style="width:{pct:.1f}%;"></div></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Nearest Matches</div>', unsafe_allow_html=True)
        df2 = nn[["material","band_gap_eV","radius_nm","epsilon_r","crystal_structure"]].copy()
        df2.columns = ["Material","Eg (eV)","R (nm)","εr","Crystal"]
        df2["Eg (eV)"] = df2["Eg (eV)"].round(3)
        st.dataframe(df2, use_container_width=True)

    except Exception as e:
        st.error(f"Inverse failed: {e}")

st.markdown("<hr/>", unsafe_allow_html=True)


# ============================================================
#                    5) CURVE GENERATION
# ============================================================

st.markdown('<div class="section-title">5) Band Gap vs Radius Curve</div>', unsafe_allow_html=True)

if st.button("Generate Curve"):
    try:
        R = np.linspace(1,10,40)
        Eg_list = [system.predict_bandgap(material,float(r),epsilon_r,crystal_structure) for r in R]

        fig,ax = plt.subplots()
        ax.plot(R,Eg_list,linewidth=2)
        ax.set_xlabel("Radius (nm)")
        ax.set_ylabel("Band Gap (eV)")
        ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Plot failed: {e}")


# ============================================================
#                        GPT CHAT POPUP
# ============================================================

import requests

st.markdown("""
<style>
#chat-btn {
    position: fixed;
    bottom: 30px;
    right: 30px;
    background: #6c63ff;
    color: white;
    width: 65px;
    height: 65px;
    font-size: 32px;
    border-radius: 50%;
    text-align: center;
    line-height: 60px;
    cursor: pointer;
    box-shadow:0px 4px 10px rgba(0,0,0,0.3);
    z-index:999999;
}
</style>

<div id="chat-btn">💬</div>

<script>
document.getElementById("chat-btn").onclick = function(){
    window.parent.postMessage({type:"open-chat"}, "*");
};
</script>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## 💬 GPT Assistant")

    user_key = st.text_input("OpenAI API Key", type="password")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_message = st.text_area("Your message")

    if st.button("Send"):
        if not user_key:
            st.warning("Add your API key first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_message})

            headers = {"Authorization": f"Bearer {user_key}", "Content-Type":"application/json"}
            payload = {
                "model":"gpt-4o-mini",
                "messages": st.session_state.chat_history
            }

            try:
                r = requests.post("https://api.openai.com/v1/chat/completions",
                                  headers=headers, data=json.dumps(payload))
                reply = r.json()["choices"][0]["message"]["content"]
                st.session_state.chat_history.append({"role":"assistant","content":reply})
            except Exception as e:
                st.error(f"GPT Error: {e}")

    for msg in st.session_state.chat_history:
        speaker = "🧑‍💻 You" if msg["role"]=="user" else "🤖 GPT"
        st.markdown(f"**{speaker}:** {msg['content']}")


# END
