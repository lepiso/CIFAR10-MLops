import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        header { display: flex !important; }
        [data-testid="stSidebarNav"] { display: none !important; }
        .main .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_model, predict, log_prediction, load_history_as_dataframe, get_stats, get_class_distribution, clear_history
from config import CLASSES

# Charger le modèle
model = load_model()

# Sidebar avec navigation
with st.sidebar:
    st.title("📊 CIFAR-10 Classifier")
    st.markdown("---")
    
    st.markdown("### 🧭 Navigation")
    
    # Boutons de navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Accueil", use_container_width=True):
            st.switch_page("app.py")
    with col2:
        if st.button("🔮 Prédiction", use_container_width=True):
            st.switch_page("pages/1_predictions.py")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Métriques", use_container_width=True):
            st.switch_page("pages/2_metriques.py")
    with col2:
        if st.button("📈 Drift", use_container_width=True):
            st.switch_page("pages/3_drift.py")
    
    if st.button("📜 Historique", use_container_width=True):
        st.switch_page("pages/4_historiques.py")
    
    st.markdown("---")

    if st.button("🔍 XAI", use_container_width=True):
        st.switch_page("pages/5_XAI.py")
    
    if model:
        st.success("✅ Modèle chargé")
    else:
        st.error("❌ Modèle non chargé")
    
    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.info(
        "Classification d'images en 10 catégories avec un CNN "
        "entraîné sur CIFAR-10."
    )

# ==================== PAGE D'ACCUEIL ====================
st.title("🖼️ Classification d'images CIFAR-10")

st.markdown("""
Bienvenue sur l'application de classification d'images !

Cette application utilise un réseau de neurones convolutif (CNN) entraîné sur 
le dataset CIFAR-10 pour classifier des images en 10 catégories différentes.
""")

st.markdown("### 🏷️ Classes disponibles")
cols = st.columns(5)
for i, col in enumerate(cols):
    with col:
        st.markdown(f"**{i+1}.** {CLASSES[i]}")
        st.markdown(f"**{i+6}.** {CLASSES[i+5]}")

st.markdown("---")
st.markdown("### 🚀 Utilisation")
st.markdown("""
1. Allez dans la page **Prédiction** pour classifier une image
2. Téléchargez une image (JPG, PNG, JPEG)
3. Visualisez la classe prédite et les probabilités associées
4. Consultez les métriques et l'historique dans les autres onglets
""")
