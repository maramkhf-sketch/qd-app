# ============================================================
#  Quantum Dot Band Gap Predictor — Full Final Version
#  With integrated Chat (GPT from secrets), doping physics,
#  nearest-material fallback, curve plotting, inverse search.
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import requests
import re
from pathlib import Path
import matplotlib.pyplot as plt
from qd_pipeline import QDSystem


# ----------------------------
#   Streamlit Page Settings
# ----------------------------
st.set_page_config(
    page_title="Quantum Dot Band Gap Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Quantum Dot Band Gap Predictor")


# ----------------------------
#   Paths
# ----------------------------
HERE = Path(__file__).parent
DATA_CSV = HERE / "qd_data.csv"
MODEL_PATH = HERE / "qd_system.joblib"


# ============================================================
# 0) GPT Chat — Inside the Page (uses Streamlit secrets)
# ============================================================

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

st.markdown("## 💬 Chat with the Predictor")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! How can I help you with quantum dots today?"}
    ]

# Show chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "assistant":
        st.markdown(f"**AI:** {msg['content']}")
    else:
        st.markdown(f"**You:** {msg['content']}")

# Chat input
user_msg = st.text_input("Type your message here:")

if st.button("Send"):
    if user_msg:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_msg}
        )

        # Use GPT if key exists
        if OPENAI_API_KEY:
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                }

                payload = {
                    "model": "gpt-4o-mini",
                    "messages": st.session_state.chat_history
                }

                r = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )

                reply = r.json()["choices"][0]["message"]["content"]

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": reply}
                )

            except Exception as e:
                st.error(f"GPT Error: {e}")

        else:
            # fallback if no API key
            st.session_state.chat_history.append(
                {"role": "assistant",
                 "content": "GPT mode is off because no API key exists in Streamlit secrets."}
            )

        st.rerun()


st.markdown("---")


# ============================================================
#  Load or Train System (Hybrid)
# ============================================================

@st.cache_resource
def load_system():
    df = pd.read_csv(DATA_CSV)
    sys = QDSystem()
    sys.fit(df)
    return sys


system = load_system()
materials_list = sorted(pd.read_csv(DATA_CSV)["material"].unique())


# ============================================================
# 1) Inputs
# ============================================================

st.header("1) Input Parameters")

c1, c2 = st.columns(2)

with c1:
    material = st.text_input("Material", "CdSe")

with c2:
    crystal_structure = st.selectbox(
        "Crystal Structure",
        ["ZB", "WZ", "Rocksalt", "Perovskite", "Rutile"]
    )

radius_nm = st.number_input("Radius (nm)", min_value=0.3, max_value=30.0,
                            value=3.0, step=0.1)

epsilon_r = st.number_input("Dielectric Constant (εr)",
                            min_value=1.0, max_value=100.0,
                            value=9.5, step=0.1)

# -------------------- optional doping --------------------
st.subheader("Doping (Optional)")

d1, d2, d3 = st.columns(3)

with d1:
    dopant = st.text_input("Dopant", "")

with d2:
    doping_type = st.selectbox("Type", ["", "n", "p"])

with d3:
    doping_conc_cm3 = st.number_input(
        "Conc. (cm⁻³)", min_value=0.0, max_value=1e22,
        value=0.0, step=1e16, format="%.3e"
    )

st.markdown("---")


# ============================================================
# 2) Forward Prediction: Eg
# ============================================================

st.header("2) Forward Prediction")

if st.button("Predict Eg"):
    try:
        Eg = system.predict_bandgap(
            material=material,
            radius_nm=radius_nm,
            epsilon_r=epsilon_r,
            crystal_structure=crystal_structure,
            dopant=dopant if dopant else None,
            doping_type=doping_type if doping_type else None,
            doping_conc=doping_conc_cm3 if doping_conc_cm3 > 0 else None
        )

        st.success(f"Predicted Eg: **{Eg:.3f} eV**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")


# ============================================================
# 3) Inverse Suggestion
# ============================================================

st.header("3) Inverse Suggestion")

target_Eg = st.number_input("Target Eg (eV)", min_value=0.0, max_value=10.0,
                            value=2.0, step=0.05)

if st.button("Suggest Materials"):
    try:
        topk, nn = system.suggest_materials_from_bandgap(
            target_Eg,
            radius_nm,
            epsilon_r,
            crystal_structure,
            top_k=3
        )

        if len(topk) == 0:
            st.info("No matches found.")
        else:
            for rank, (name, score) in enumerate(topk, start=1):
                st.write(f"**#{rank}: {name} — Score {score:.2f}**")

        st.subheader("Nearest Matches")

        cols = ["material", "band_gap_eV", "radius_nm", "epsilon_r", "crystal_structure"]
        show = nn[cols].copy()
        show.columns = ["Material", "Eg", "R", "εr", "Structure"]
        st.dataframe(show)

    except Exception as e:
        st.error(f"Inverse failed: {e}")

st.markdown("---")


# ============================================================
# 4) Curve (Eg vs R)
# ============================================================

st.header("4) Band Gap vs Radius Curve")

if st.button("Generate Curve"):
    try:
        R = np.linspace(1, 10, 40)
        EG = [
            system.predict_bandgap(
                material, float(r), epsilon_r, crystal_structure
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
