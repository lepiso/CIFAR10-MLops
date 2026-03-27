import streamlit as st
import sys
import os

st.set_page_config(page_title="Prédiction", page_icon="🔮", layout="wide")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_model, predict, log_prediction
from config import CLASSES
from PIL import Image
import plotly.graph_objects as go

# Ajouter un bouton retour en haut
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("⬅️ Retour", use_container_width=True):
        st.switch_page("app.py")
with col2:
    st.title("🔮 Classification d'image")

model = load_model()

if model is None:
    st.error("❌ Modèle non disponible")
    st.stop()

uploaded_file = st.file_uploader("Choisissez une image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Image chargée", width=300)
    
    with col2:
        with st.spinner("Analyse..."):
            result = predict(model, image)
            log_prediction(result)
        
        st.success(f"Prédiction: **{result['class'].upper()}**")
        st.metric("Confiance", f"{result['confidence']:.2%}")
        st.progress(result['confidence'])
    
    fig = go.Figure(data=[go.Bar(
        x=list(result['all_probs'].keys()),
        y=list(result['all_probs'].values()),
        marker_color=['red' if c == result['class'] else 'blue' for c in result['all_probs'].keys()]
    )])
    fig.update_layout(title="Probabilités par classe", height=400)
    st.plotly_chart(fig, use_container_width=True)
