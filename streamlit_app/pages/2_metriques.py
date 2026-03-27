import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Métriques", page_icon="📊", layout="wide")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_history_as_dataframe, get_stats
from config import CLASSES

# Ajouter un bouton retour en haut
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("⬅️ Retour", use_container_width=True):
        st.switch_page("app.py")
with col2:
    st.title("📊 Métriques et Statistiques")

df = load_history_as_dataframe()

if df.empty:
    st.warning("⚠️ Aucune donnée disponible. Effectuez d'abord des prédictions.")
    st.stop()

stats = get_stats(df.to_dict('records'))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total prédictions", stats['total'])
with col2:
    st.metric("Confiance moyenne", f"{stats['avg_confidence']:.2%}")
with col3:
    st.metric("Classe la plus prédite", stats['most_common_class'])
with col4:
    st.metric("Classes distinctes", stats['unique_classes'])

st.markdown("---")

if 'predicted' in df.columns:
    class_counts = df['predicted'].value_counts().reset_index()
    class_counts.columns = ['Classe', 'Nombre']
    
    fig = px.bar(class_counts, x='Classe', y='Nombre', color='Nombre',
                title="Distribution des classes")
    st.plotly_chart(fig, use_container_width=True)
