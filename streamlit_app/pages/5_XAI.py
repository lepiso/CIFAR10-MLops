import streamlit as st
import sys
import os
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import tensorflow as tf

st.set_page_config(page_title="XAI - Explicabilité", page_icon="🔍", layout="wide")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_model, preprocess_image
from xai_utils import XAIExplainer
from config import CLASSES

st.title("🔍 Explicabilité du Modèle (XAI)")
st.markdown("""
Cette page utilise **LIME (Local Interpretable Model-agnostic Explanations)** pour vous montrer 
quelles zones de l'image sont les plus importantes pour la prédiction.
""")

# Chargement du modèle
model = load_model()

if model is None:
    st.error("❌ Modèle non disponible")
    st.stop()

explainer = XAIExplainer(model, CLASSES, preprocess_image)

# Upload d'image
uploaded_file = st.file_uploader(
    "Choisissez une image...",
    type=['jpg', 'jpeg', 'png'],
    help="Téléchargez une image pour analyser la décision du modèle"
)

if uploaded_file is None:
    st.info("📤 Téléchargez une image pour commencer")
    st.stop()

# Chargement l'image
image = Image.open(uploaded_file).convert('RGB')

col1, col2 = st.columns([1, 1])

with col1:
    st.image(image, caption="Image chargée", width=300)

# Faire une prédiction
result = explainer.model.predict(preprocess_image(image), verbose=0)
pred_idx = np.argmax(result[0])
pred_class = CLASSES[pred_idx]
probs = tf.nn.softmax(result[0]).numpy()
confidence = float(probs[pred_idx])
confidence = min(max(confidence, 0.0), 1.0)
confidence = result[0][pred_idx]

with col2:
    st.success(f"🎯 Prédiction: **{pred_class.upper()}**")
    st.metric("Confiance", f"{confidence:.2%}")

st.markdown("---")
st.subheader("🌿 Explication LIME")

# Générer l'explication LIME
with st.spinner("Génération de l'explication en cours..."):
    fig = explainer.explain_lime(image)
    
    if fig:
        st.pyplot(fig)
        st.markdown("""
        ### 🧠 Comment interpréter LIME ?
        
        - **Zones rouges** : pixels qui **favorisent** la prédiction
        - **Zones bleues** : pixels qui **s'opposent** à la prédiction
        - L'algorithme analyse localement autour de l'image pour déterminer quels pixels influencent la décision
        
        Plus la zone est intense, plus elle est importante pour la décision du modèle.
        """)
    else:
        st.error("❌ LIME n'a pas pu générer l'explication")

# Top 3 des classes
st.markdown("---")
st.subheader("🏆 Top 3 des classes")
probs = result[0]
top3_idx = np.argsort(probs)[-3:][::-1]
top3_probs = probs[top3_idx]
top3_classes = [CLASSES[i] for i in top3_idx]

fig = go.Figure(go.Bar(
    x=top3_classes,
    y=top3_probs,
    marker_color=['#FF4B4B' if i == 0 else '#1E88E5' for i in range(3)],
    text=[f"{p:.2%}" for p in top3_probs],
    textposition='auto'
))
fig.update_layout(title="Probabilités par classe", height=300)
st.plotly_chart(fig, use_container_width=True)
