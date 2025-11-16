# app.py — Quantum Dot Band Gap Predictor with GPT chat + optional doping

import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests

from qd_pipeline import QDSystem

# ------------------- Config ------------------- #
st.set_page_config(
    page_title="Quantum Dot Band Gap Predictor",
    page_icon="🧠",
    layout="centered",
)

HERE = Path(__file__).parent.resolve()
DATA_CSV = HERE / "qd_data.csv"

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

# ------------------- Style ------------------- #
st.markdown(
    """
<style>
.big-title {
  font-size: 2rem;
  font-weight: 800;
  margin-top: 0.6rem;
  margin-bottom: 0.2rem;
}
.section-title {
  font-size: 1.35rem;
  font-weight: 700;
  margin: 1.0rem 0 0.4rem;
}
hr {
  margin: 0.9rem 0;
}
.inv-card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 10px;
  padding: 0.7rem 0.9rem;
  margin-bottom: 0.5rem;
}
.inv-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.inv-rank {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background: rgba(255,255,255,0.08);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
}
.inv-name {
  font-weight: 700;
}
.inv-score-wrap {
  flex: 1;
}
.inv-bar {
  height: 6px;
  border-radius: 4px;
  background: rgba(255,255,255,0.12);
  overflow: hidden;
  margin-top: 4px;
}
.inv-bar > div {
  height: 100%;
  background: linear-gradient(90deg,#22c55e,#3b82f6);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🧠 Quantum Dot Band Gap Predictor</div>', unsafe_allow_html=True)

# ------------------- GPT Chat (top, always visible) ------------------- #
st.markdown("### 💬 Chat with the Predictor")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi! How can I help you with quantum dots today?"}
    ]

# show history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AI:** {msg['content']}")

user_msg = st.text_input("Type your message here:", key="chat_input")

def call_gpt(messages):
    if not OPENAI_API_KEY:
        raise RuntimeError("No OPENAI_API_KEY in Streamlit secrets.")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.2,
    }
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers=headers, data=json.dumps(payload))
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

col_send, col_hint = st.columns([0.2, 0.8])
with col_hint:
    if not OPENAI_API_KEY:
        st.caption("AI: GPT mode is off because no API key exists in Streamlit secrets.")
    else:
        st.caption("AI: Powered by GPT-4o-mini for explanations and guidance.")

with col_send:
    if st.button("Send", type="primary") and user_msg.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        reply = ""
        try:
            if OPENAI_API_KEY:
                messages = [{"role": "system",
                             "content": "You are a helpful assistant for quantum dots and band-gap physics."}]
                messages += st.session_state.chat_history
                reply = call_gpt(messages)
            else:
                reply = "I don't have API access here, but you can still use the predictor form below."
        except Exception as e:
            reply = f"GPT error: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.experimental_rerun()

st.markdown("<hr/>", unsafe_allow_html=True)

# ------------------- Helper: crystal normalization ------------------- #
CRYSTAL_ALIASES = {
    "zb": "ZB",
    "zincblende": "ZB",
    "fcc": "ZB",
    "wz": "WZ",
    "wurtzite": "WZ",
    "hex": "WZ",
    "rs": "Rocksalt",
    "rocksalt": "Rocksalt",
    "nacl": "Rocksalt",
    "perovskite": "Perovskite",
    "pvsk": "Perovskite",
    "rutile": "Rutile",
    "tio2": "Rutile",
}

def normalize_crystal(s: str | None, default="ZB") -> str:
    if not s:
        return default
    key = re.sub(r"[^A-Za-z]", "", s).lower()
    if key in CRYSTAL_ALIASES:
        return CRYSTAL_ALIASES[key]
    if s in ["ZB", "WZ", "Rocksalt", "Perovskite", "Rutile"]:
        return s
    return default

# ------------------- Free-text parser (regex fallback) ------------------- #
def parse_free_text(text: str):
    out = {
        "material": None,
        "radius_nm": None,
        "epsilon_r": None,
        "crystal_structure": None,
        "dopant": None,
        "doping_type": None,
        "doping_conc_cm3": None,
    }
    if not text:
        return out

    # simple material token
    m = re.search(r"\b([A-Z][a-z]?[A-Za-z0-9]{0,6}(?:[A-Z][a-z]?[A-Za-z0-9]{0,6})*)\b", text)
    if m:
        out["material"] = m.group(1)

    # radius
    m = re.search(r"(\d+(?:\.\d+)?)\s*(nm|nanometer|nano)\b", text, re.IGNORECASE)
    if m:
        out["radius_nm"] = float(m.group(1))

    # epsilon
    m = re.search(r"(?:εr|epsilon|er)\s*[:=]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        out["epsilon_r"] = float(m.group(1))

    # crystal keywords
    for key in (set(CRYSTAL_ALIASES.keys()) | {"ZB", "WZ", "Rocksalt", "Perovskite", "Rutile"}):
        if re.search(rf"\b{key}\b", text, re.IGNORECASE):
            out["crystal_structure"] = normalize_crystal(key)
            break

    # doping type
    if re.search(r"\bn[\s-]*type\b", text, re.IGNORECASE):
        out["doping_type"] = "n"
    if re.search(r"\bp[\s-]*type\b", text, re.IGNORECASE):
        out["doping_type"] = "p"

    # dopant element
    m = re.search(r"dop(?:ed|ing)?\s*(?:with)?\s*([A-Za-z]{1,2}[a-z]?)", text)
    if m:
        out["dopant"] = m.group(1).capitalize()

    # concentration (rough)
    m = re.search(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:cm-3|cm\^-3|cm\⁻³)?", text, re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1))
            if 1e14 <= val <= 1e22:
                out["doping_conc_cm3"] = val
        except Exception:
            pass

    return out

# ------------------- Load & train system ------------------- #
@st.cache_resource(show_spinner=True)
def load_system():
    df = pd.read_csv(DATA_CSV)
    sys = QDSystem()
    sys.fit(df)
    return sys

system = load_system()

# ======================= 1) Chat-style Input (for parsing) ======================= #
st.markdown('<div class="section-title">1) Chat-style Input</div>', unsafe_allow_html=True)
st.caption("Example: `cdse 3 nm er=9.5 wz, n-type, Si 2e18`")

chat_text = st.text_input("Describe your quantum dot in one line (optional):", key="parse_line")

parsed = {
    "material": None,
    "radius_nm": None,
    "epsilon_r": None,
    "crystal_structure": None,
    "dopant": None,
    "doping_type": None,
    "doping_conc_cm3": None,
}

use_gpt_parse = False  # لو حبيتي نرجعه بعدين نضيف خيار، الآن نخليه regex فقط عشان الثبات.

if chat_text:
    parsed = parse_free_text(chat_text)

st.markdown("<hr/>", unsafe_allow_html=True)

# ======================= 2) Structured Input ======================= #
st.markdown('<div class="section-title">2) Structured Input</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    material = st.text_input("Material", value=(parsed["material"] or "CdSe"))
    radius_nm = st.number_input(
        "Radius (nm)",
        min_value=0.3,
        max_value=30.0,
        value=float(parsed["radius_nm"] or 3.0),
        step=0.1,
    )
with c2:
    epsilon_r = st.number_input(
        "Dielectric Constant (εr)",
        min_value=1.0,
        max_value=100.0,
        value=float(parsed["epsilon_r"] or 9.5),
        step=0.1,
    )
    crystal_structure = st.selectbox(
        "Crystal Structure",
        ["ZB", "WZ", "Rocksalt", "Perovskite", "Rutile"],
        index=["ZB", "WZ", "Rocksalt", "Perovskite", "Rutile"].index(
            parsed["crystal_structure"] or "ZB"
        ),
    )

# ===== Doping (optional) =====
st.markdown('<div class="section-title">Doping (optional)</div>', unsafe_allow_html=True)
d1, d2, d3 = st.columns([0.5, 0.25, 0.25])

with d1:
    dopant = st.text_input("Dopant", value=(parsed["dopant"] or ""))

with d2:
    doping_type = st.selectbox(
        "Type",
        ["", "n", "p"],
        index=["", "n", "p"].index(parsed["doping_type"] or ""),
    )

with d3:
    doping_conc_cm3 = st.number_input(
        "Conc. (cm⁻³)",
        min_value=0.0,
        max_value=1e22,
        value=float(parsed["doping_conc_cm3"] or 0.0),
        step=1e16,
        format="%.3e",
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ======================= 3) Forward Prediction ======================= #
st.markdown('<div class="section-title">3) Forward Prediction</div>', unsafe_allow_html=True)

if st.button("Predict Eg", key="forward_btn"):
    try:
        dop_conc = doping_conc_cm3 if doping_conc_cm3 > 0 else None
        eg = system.predict_bandgap(
            material=material,
            radius_nm=radius_nm,
            epsilon_r=epsilon_r,
            crystal_structure=crystal_structure,
            dopant=(dopant or None),
            doping_type=(doping_type or None),
            doping_conc_cm3=dop_conc,
        )
        st.success(f"Predicted Band Gap: {eg:.3f} eV")
        if dop_conc is not None or dopant or doping_type:
            st.caption(
                f"Doping applied → dopant: {dopant or '—'}, "
                f"type: {doping_type or '—'}, "
                f"conc: {dop_conc:.2e} cm⁻³"
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("<hr/>", unsafe_allow_html=True)

# ======================= 4) Inverse Suggestions ======================= #
st.markdown('<div class="section-title">4) Inverse Suggestions (Top-K)</div>', unsafe_allow_html=True)

col_target, col_btn = st.columns([0.6, 0.4])
with col_target:
    target_Eg = st.number_input(
        "Target Eg (eV)",
        min_value=0.0,
        max_value=10.0,
        value=2.2,
        step=0.05,
    )
with col_btn:
    run_inverse = st.button("Suggest Materials", key="inverse_btn")

if run_inverse:
    try:
        topk, nn = system.suggest_materials_from_bandgap(
            target_Eg,
            radius_nm=radius_nm,
            epsilon_r=epsilon_r,
            crystal_structure=crystal_structure,
            top_k=3,
        )

        if not topk:
            st.info("No suggestions found.")
        else:
            for rank, (name, score) in enumerate(topk, start=1):
                pct = float(np.clip(score, 0.0, 1.0) * 100.0)
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
                    """,
                    unsafe_allow_html=True,
                )

        if not nn.empty:
            st.markdown('<div class="section-title">Nearest Matches</div>', unsafe_allow_html=True)
            cols = ["material", "band_gap_eV", "radius_nm", "epsilon_r", "crystal_structure", "Eg_pred"]
            show = nn[cols].copy()
            show.columns = ["Material", "Eg (dataset)", "R (nm)", "εr", "Crystal", "Eg (pred at R)"]
            show["Eg (dataset)"] = show["Eg (dataset)"].round(3)
            show["Eg (pred at R)"] = show["Eg (pred at R)"].round(3)
            st.dataframe(show.reset_index(drop=True), use_container_width=True)

    except Exception as e:
        st.error(f"Inverse prediction failed: {e}")

st.markdown("<hr/>", unsafe_allow_html=True)

# ======================= 5) Band Gap vs Radius Curve ======================= #
st.markdown('<div class="section-title">5) Band Gap vs Radius Curve</div>', unsafe_allow_html=True)

if st.button("Generate Curve", key="curve_btn"):
    try:
        R = np.linspace(1.0, 10.0, 40)
        EG = [
            system.predict_bandgap(
                material=material,
                radius_nm=float(r),
                epsilon_r=epsilon_r,
                crystal_structure=crystal_structure,
                dopant=None,
                doping_type=None,
                doping_conc_cm3=None,
            )
            for r in R
        ]

        fig, ax = plt.subplots()
        ax.plot(R, EG, linewidth=2)
        ax.set_xlabel("Radius (nm)")
        ax.set_ylabel("Band Gap (eV)")
        ax.set_title(f"Eg vs Radius — {material}")
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Plot failed: {e}")
