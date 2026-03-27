import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Data Drift", page_icon="📈", layout="wide")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_history_as_dataframe, get_class_distribution
from config import CLASSES

# Ajouter un bouton retour en haut
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("⬅️ Retour", use_container_width=True):
        st.switch_page("app.py")
with col2:
    st.title("📈 Détection de Data Drift")

df = load_history_as_dataframe()

if df.empty:
    st.warning("⚠️ Aucune donnée disponible")
    st.stop()

if 'timestamp' in df.columns and len(df) > 1:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    mid_point = df['timestamp'].min() + (df['timestamp'].max() - df['timestamp'].min()) / 2
    ref_data = df[df['timestamp'] <= mid_point]
    current_data = df[df['timestamp'] > mid_point]
    
    st.info(f"📊 Période de référence: {len(ref_data)} prédictions")
    st.info(f"📊 Période actuelle: {len(current_data)} prédictions")
    
    ref_dist = get_class_distribution(ref_data.to_dict('records'))
    current_dist = get_class_distribution(current_data.to_dict('records'))
    
    ref_total = len(ref_data) if len(ref_data) > 0 else 1
    current_total = len(current_data) if len(current_data) > 0 else 1
    
    ref_pct = [ref_dist.get(c, 0) / ref_total * 100 for c in CLASSES]
    current_pct = [current_dist.get(c, 0) / current_total * 100 for c in CLASSES]
    
    fig = make_subplots(1, 2, subplot_titles=("Période de référence", "Période actuelle"))
    fig.add_trace(go.Bar(x=CLASSES, y=ref_pct, name="Référence", marker_color='blue'), row=1, col=1)
    fig.add_trace(go.Bar(x=CLASSES, y=current_pct, name="Actuelle", marker_color='red'), row=1, col=2)
    fig.update_layout(height=500, title_text="Distribution des classes (%)")
    st.plotly_chart(fig, use_container_width=True)
    
    import numpy as np
    drift_score = np.mean([abs(ref_pct[i] - current_pct[i]) for i in range(len(CLASSES))])
    
    st.markdown("---")
    st.subheader("⚠️ Analyse de drift")
    
    if drift_score < 5:
        st.success(f"✅ Distribution stable (score: {drift_score:.2f}%)")
    elif drift_score < 15:
        st.warning(f"⚠️ Léger drift détecté (score: {drift_score:.2f}%)")
    else:
        st.error(f"❌ Drift significatif détecté (score: {drift_score:.2f}%)")
else:
    st.warning("⚠️ Pas assez de données pour analyser le drift")
