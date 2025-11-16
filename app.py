import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from qd_pipeline import QDSystem

st.set_page_config(page_title="Quantum Dot Band Gap Predictor", layout="wide")

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_csv("qd_data.csv")
    return df

data = load_data()

# ========== MODEL SYSTEM ==========
qd = QDSystem(data)

# ========== GPT CHAT (ALWAYS VISIBLE) ==========
st.markdown("<h1 style='text-align:center;'>💡 Quantum Dot Band Gap Predictor</h1>", unsafe_allow_html=True)
st.markdown("## 🧠 Chat with the Predictor")

# حافظين الشات
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "AI: Hello! How can I help you with quantum dots today?"}
    ]

# عرض المحادثة
for msg in st.session_state.chat_history:
    if msg["role"] == "assistant":
        st.markdown(f"**AI:** {msg['content']}")
    else:
        st.markdown(f"**You:** {msg['content']}")

# مدخل الشات
user_text = st.text_input("Type your message here:")

if st.button("Send"):
    if user_text.strip() != "":
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        st.markdown(f"**You:** {user_text}")

        # API KEY من Streamlit Secrets
        api_key = st.secrets.get("OPENAI_KEY", None)

        if api_key is None:
            reply = "GPT mode is OFF because no API key exists in Streamlit secrets."
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": "gpt-4o-mini",
                "messages": st.session_state.chat_history
            }

            try:
                r = requests.post("https://api.openai.com/v1/chat/completions",
                                  headers=headers, data=json.dumps(payload))
                reply = r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                reply = f"GPT Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.experimental_rerun()

st.markdown("---")

# ========== USER INPUT FORM ==========
st.markdown("## 🧪 User Input Form")

col1, col2 = st.columns(2)

with col1:
    material = st.selectbox("Material", sorted(data["material"].unique()))
    radius_nm = st.number_input("Radius (nm)", min_value=0.5, max_value=20.0, value=3.0)

with col2:
    epsilon_r = st.number_input("ε_r (static dielectric constant)", min_value=1.0, max_value=20.0, value=9.5)
    crystal_structure = st.selectbox("Crystal Structure", ["ZB", "WZ"])

# OPTIONAL DOPING
st.markdown("## 🧬 Doping (optional)")

col3, col4, col5 = st.columns(3)

with col3:
    dopant = st.text_input("Dopant (e.g. Cu, Mn)")

with col4:
    dop_type = st.selectbox("Type", ["n-type", "p-type", "None"])

with col5:
    conc = st.number_input("Conc. (cm⁻³)", min_value=0.0, max_value=1e21, value=0.0, step=1e18, format="%.2e")

st.markdown("---")

# ========== FORWARD PREDICTION ==========
st.markdown("## 🎯 Forward Prediction")

if st.button("Predict Eg"):
    try:
        eg = qd.predict_forward(material, radius_nm, epsilon_r, crystal_structure, dopant, dop_type, conc)
        st.success(f"Predicted Band Gap: {eg:.3f} eV")
    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")

# ========== INVERSE PREDICTION ==========
st.markdown("## 🔍 Inverse Suggestions (Top-K)")

target_eg = st.number_input("Target Eg (eV)", min_value=0.1, max_value=8.0, value=2.20)

if st.button("Suggest Materials"):
    try:
        suggestions = qd.inverse_predict(target_eg)
        st.table(suggestions)
    except Exception as e:
        st.error(f"Inverse Error: {e}")

st.markdown("---")

# ========== CURVE PLOT ==========
st.markdown("## 📈 Band Gap vs Radius Curve")

if st.button("Plot Curve"):
    try:
        radii, eg_curve = qd.plot_curve(material, epsilon_r, crystal_structure)
        fig, ax = plt.subplots()
        ax.plot(radii, eg_curve)
        ax.set_xlabel("Radius (nm)")
        ax.set_ylabel("Band Gap (eV)")
        ax.set_title(f"Eg vs Radius — {material}")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Plot Error: {e}")

