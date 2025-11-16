import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
from qd_pipeline import QDSystem

st.set_page_config(page_title="Quantum Dot Band Gap Predictor", layout="wide")

system = QDSystem()

# -----------------------
# Header
# -----------------------
st.markdown("<h1 style='text-align:center;'>💡 Quantum Dot Band Gap Predictor</h1>", unsafe_allow_html=True)

# -----------------------
# ChatGPT Section (fixed)
# -----------------------
st.subheader("🤖 Chat with the Predictor")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_msg = st.text_input("Your question:", "")

if st.button("Send"):
    if user_msg.strip() != "":
        st.session_state.chat.append(("You", user_msg))

        headers = {"Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}"}
        payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": user_msg}]}

        try:
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, data=json.dumps(payload))
            reply = r.json()["choices"][0]["message"]["content"]
        except:
            reply = "Error: GPT unavailable."

        st.session_state.chat.append(("AI", reply))

# Display messages
for sender, msg in st.session_state.chat:
    st.write(f"**{sender}:** {msg}")

st.markdown("---")

# -----------------------
# User Inputs
# -----------------------
st.subheader("🔧 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    material = st.text_input("Material", "CdSe")
    radius_nm = st.number_input("Radius (nm)", 1.0, 10.0, 3.0)
with col2:
    eps_r = st.number_input("Dielectric Constant εr", 1.0, 25.0, 9.5)

st.subheader("Optional Doping")
dopant_conc = st.number_input("Doping concentration (cm⁻³)", 0.0, 1e20, 0.0)

# -----------------------
# Forward Prediction
# -----------------------
st.subheader("🎯 Forward Prediction")

if st.button("Predict Eg"):
    Eg = system.predict(material, radius_nm, eps_r, dopant_conc)
    st.success(f"Predicted Band Gap: **{Eg} eV**")

st.markdown("---")

# -----------------------
# Curve
# -----------------------
st.subheader("📈 Band Gap vs Radius Curve")

r, Eg = system.curve(eps_r)

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(r, Eg)
ax.set_xlabel("Radius (nm)")
ax.set_ylabel("Eg (eV)")
ax.grid(True)

st.pyplot(fig)
