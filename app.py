import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from qd_pipeline import QDSystem

# ---------------------------------------------------
# GPT CHAT — always visible
# ---------------------------------------------------

st.set_page_config(page_title="Quantum Dot Predictor", layout="wide")

st.markdown("<h1 style='text-align:center;'>Quantum Dot Band Gap Predictor</h1>", unsafe_allow_html=True)

st.markdown("## 💬 Chat with the Predictor")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! How can I help you with quantum dots today?"}
    ]

# Show chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div style='color:#00c3ff;'>You: {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:#ff4fd8;'>AI: {msg['content']}</div>", unsafe_allow_html=True)

user_input = st.text_input("Type your message here:")

if st.button("Send"):
    if user_input.strip() != "":
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        try:
            api_key = st.secrets["OPENAI_API_KEY"]
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {"model": "gpt-4o-mini", "messages": st.session_state.chat_history}

            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, data=json.dumps(payload))

            reply = r.json()["choices"][0]["message"]["content"]
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        except:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "GPT mode is off because no API key exists in Streamlit secrets."
            })

        st.experimental_rerun()

st.markdown("---")

# ---------------------------------------------------
# QD FORM (old UI exactly)
# ---------------------------------------------------

st.markdown("## 🔧 Quantum Dot Input Form")

qd = QDSystem("qd_data.csv")

col1, col2 = st.columns(2)

with col1:
    material = st.selectbox("Material", qd.materials)
    radius = st.number_input("Radius (nm)", min_value=1.0, max_value=50.0, value=3.0)
    crystal = st.selectbox("Crystal Structure", ["ZB", "WZ"])

with col2:
    eps_r = st.number_input("Dielectric Constant εᵣ", min_value=1.0, max_value=20.0, value=9.5)

st.markdown("### Doping (optional)")

colA, colB, colC = st.columns(3)

with colA:
    dopant = st.text_input("Dopant")

with colB:
    doping_type = st.selectbox("Type", ["", "n-type", "p-type"])

with colC:
    conc = st.number_input("Conc. (cm⁻³)", min_value=0.0, value=0.0)

st.markdown("---")

# ---------------------------------------------------
# Forward Prediction
# ---------------------------------------------------

st.markdown("## 🎯 Forward Prediction")

if st.button("Predict Eg"):
    Eg = qd.predict_bandgap(material, radius, eps_r, crystal,
                            dopant=dopant, doping_type=doping_type, concentration=conc)
    st.success(f"Predicted Band Gap: **{Eg:.3f} eV**")

st.markdown("---")

# ---------------------------------------------------
# Inverse Suggestion
# ---------------------------------------------------

st.markdown("## 🔍 Inverse Suggestions (Top-K)")

target_Eg = st.number_input("Target Eg (eV)", min_value=0.1, max_value=10.0, value=2.20)

if st.button("Suggest Materials"):
    results = []
    for m in qd.materials:
        Eg = qd.predict_bandgap(m, radius, eps_r, crystal)
        results.append((m, abs(Eg - target_Eg)))

    results = sorted(results, key=lambda x: x[1])[:5]

    st.write("Top suggested materials:")
    for r in results:
        st.write(f"- **{r[0]}** (Δ = {r[1]:.3f})")

st.markdown("---")

# ---------------------------------------------------
# Band Gap vs Radius Curve
# ---------------------------------------------------

st.markdown("## 📈 Band Gap vs Radius Curve")

import matplotlib.pyplot as plt

r_list = np.linspace(1, 20, 40)
Eg_list = [qd.predict_bandgap(material, r, eps_r, crystal) for r in r_list]

fig, ax = plt.subplots()
ax.plot(r_list, Eg_list)
ax.set_xlabel("Radius (nm)")
ax.set_ylabel("Band Gap (eV)")
ax.set_title(f"Eg vs Radius — {material}")

st.pyplot(fig)
