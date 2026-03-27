import streamlit as st
import sys
import os
import pandas as pd

st.set_page_config(page_title="Historique", page_icon="📜", layout="wide")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_history_as_dataframe, clear_history, get_stats

# Ajouter un bouton retour en haut
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("⬅️ Retour", use_container_width=True):
        st.switch_page("app.py")
with col2:
    st.title("📜 Historique des prédictions")

df = load_history_as_dataframe()

with st.sidebar:
    if st.button("🗑️ Effacer l'historique", type="secondary"):
        if clear_history():
            st.success("✅ Historique effacé")
            st.rerun()

if df.empty:
    st.info("ℹ️ Aucune prédiction dans l'historique")
    st.stop()

stats = get_stats(df.to_dict('records'))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total", stats['total'])
with col2:
    st.metric("Confiance moyenne", f"{stats['avg_confidence']:.2%}")
with col3:
    st.metric("Classe fréquente", stats['most_common_class'])

st.markdown("---")

if 'timestamp' in df.columns:
    display_df = df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df = display_df.rename(columns={
        'timestamp': 'Date',
        'predicted': 'Prédiction',
        'confidence': 'Confiance'
    })
    display_df['Confiance'] = display_df['Confiance'].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(display_df[['Date', 'Prédiction', 'Confiance']], use_container_width=True)
