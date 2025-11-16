import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from qd_pipeline import QDSystem


# =============================
# 1) PAGE CONFIG
# =============================

st.set_page_config(
    page_title="Quantum Dot Band Gap Predictor",
    layout="centered",
)

st.title("🧠 Quantum Dot Band Gap Predictor")

HERE = Path(__file__).parent.resolve()
DATA_CSV = HERE / "qd_data.csv"


# =============================
# 2) LOAD SYSTEM
# =============================

@st.cache_resource
def load_system():
    df = pd.read_csv(DATA_CSV)
    sys = QDSystem()
    sys.fit(df)
    return sys

system = load_system()


# =============================
# 3) GPT SETUP
# =============================

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

def call_gpt(messages):
    """
    Calls GPT-4o-mini through OpenAI API for chat.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# =============================
# 4) CHAT SECTION (ALWAYS ON TOP)
# =============================

st.subheader("💬 Chat with the Predictor")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi! How can I help you with quantum dots today?"}
    ]

# --- Chat history display ---
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.write(f"**You:** {msg['content']}")
    else:
        st.write(f"**AI:** {msg['content']}")

user_msg = st.text_input("Type your message here:")

col_send = st.columns([1])[0]
if col_send.button("Send", type="primary") and user_msg.strip():

    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

    # Generate GPT reply
    reply = ""
    try:
        if OPENAI_API_KEY:
            msgs = [{"role": "system", "content": "You are a helpful assistant for quantum dots."}]
            msgs += st.session_state.chat_history
            reply = call_gpt(msgs)
        else:
            reply = "GPT mode is disabled — no API key found in Streamlit secrets."
    except Exception as e:
        reply = f"GPT error: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()   # FIXED — replaces deprecated experimental_rerun()


st.markdown("---")


# =============================
# 5) STRUCTURED USER INPUT FORM
# =============================

st.subheader("🧪 Structured Input")

col1, col2 = st.columns(2)

with col1:
    material = st.text_input("Material", "CdSe")
    radius_nm = st.number_input("Radius (nm)", min_value=0.5, max_value=30.0, value=3.0, step=0.1)

with col2:
    epsilon_r = st.number_input("Dielectric Constant (εr)", min_value=1.0, max_value=50.0, value=9.5)
    crystal_structure = st.selectbox("Crystal Structure", ["ZB", "WZ", "Rocksalt"], index=0)

# ----- Doping -----
st.subheader("Doping (optional)")
d1, d2, d3 = st.columns([1,1,1])

with d1:
    dopant = st.text_input("Dopant (e.g., B, P, In, Cu...)")

with d2:
    doping_type = st.selectbox("Type", ["", "n", "p"], index=0)

with d3:
    doping_conc = st.number_input("Conc. (cm⁻³)", min_value=0.0, max_value=1e22,
                                  value=0.0, step=1e16, format="%.2e")


# =============================
# 6) FORWARD PREDICTION
# =============================

st.subheader("🎯 Forward Prediction")

if st.button("Predict Eg"):
    try:
        eg = system.predict_bandgap(
            material,
            radius_nm,
            epsilon_r,
            crystal_structure,
            dopant if dopant.strip() else None,
            doping_type if doping_type.strip() else None,
            doping_conc if doping_conc > 0 else None
        )

        st.success(f"Predicted Band Gap: {eg:.3f} eV")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")


# =============================
# 7) INVERSE SUGGESTIONS
# =============================

st.subheader("🔍 Inverse Suggestions (Top-K)")

target = st.number_input("Target Eg (eV)", min_value=0.1, max_value=10.0, value=2.2, step=0.1)

if st.button("Suggest Materials"):
    try:
        topk, nn_df = system.suggest_materials_from_bandgap(
            target,
            radius_nm,
            epsilon_r,
            crystal_structure,
            top_k=3,
            dopant=dopant if dopant else None,
            doping_type=doping_type if doping_type else None,
            doping_conc=doping_conc if doping_conc > 0 else None
        )

        st.write("### Best Matches")
        for name, score in topk:
            st.write(f"- **{name}** — score: {score*100:.1f}%")

        st.write("### Nearest Neighbors")
        st.dataframe(nn_df)

    except Exception as e:
        st.error(f"Inverse prediction failed: {e}")

st.markdown("---")


# =============================
# 8) BAND GAP vs RADIUS CURVE
# =============================

st.subheader("📈 Band Gap vs Radius Curve")

if st.button("Generate Curve"):
    try:
        Rvals = np.linspace(1, 10, 40)
        EG = [
            system.predict_bandgap(
                material,
                float(r),
                epsilon_r,
                crystal_structure,
                dopant if dopant else None,
                doping_type if doping_type else None,
                doping_conc if doping_conc > 0 else None
            )
            for r in Rvals
        ]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(Rvals, EG, linewidth=2)
        ax.set_xlabel("Radius (nm)")
        ax.set_ylabel("Band Gap (eV)")
        ax.set_title(f"Eg vs Radius — {material}")
        ax.grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Plot failed: {e}")
