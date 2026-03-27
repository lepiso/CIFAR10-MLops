import os
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric
from evidently import ColumnMapping

# Configuration
LOGS_PATH = Path('logs/predictions.jsonl')
REPORTS_DIR = Path('reports/monitoring')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_PATH = Path('data/reference_stats.csv')

def load_and_prepare_data():
    """Charge et prépare les données pour Evidently"""
    
    if not LOGS_PATH.exists():
        print('❌ Pas encore de prédictions loggées.')
        return None
    
    try:
        # Chargement des données de production
        prod_df = pd.read_json(LOGS_PATH, lines=True)
        print(f"📊 {len(prod_df)} prédictions chargées")
        
        # Vérification des colonnes nécessaires
        required_cols = ['confidence', 'feat_mean', 'feat_std']
        for col in required_cols:
            if col not in prod_df.columns:
                print(f"⚠️  Colonne manquante: {col}")
                # Création des valeurs par défaut si nécessaire
                if col == 'feat_mean':
                    prod_df['feat_mean'] = 0.0
                elif col == 'feat_std':
                    prod_df['feat_std'] = 0.0
        
        return prod_df
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        return None

def create_reference_data(prod_df):
    """Crée les données de référence"""
    
    # Création du dossier data s'il n'existe pas
    Path('data').mkdir(exist_ok=True)
    
    # Sauvegarde des stats de référence
    prod_df.to_csv(REFERENCE_PATH, index=False)
    print(f"✅ Stats de référence créées: {REFERENCE_PATH}")
    print(f"   {len(prod_df)} échantillons sauvegardés")
    
    # Sauvegarde des métadonnées
    metadata = {
        'created_at': datetime.datetime.now().isoformat(),
        'num_samples': len(prod_df),
        'columns': list(prod_df.columns)
    }
    with open('data/reference_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def run_drift_report():
    """Exécute le rapport de data drift"""
    
    print("\n" + "="*50)
    print("🔍 ANALYSE DE DATA DRIFT")
    print("="*50)
    
    # Chargement les données de production
    prod_df = load_and_prepare_data()
    if prod_df is None or len(prod_df) == 0:
        return
    
    # Vérification s'il y a assez de données
    if len(prod_df) < 10:
        print(f"⚠️  Seulement {len(prod_df)} prédictions — besoin d'au moins 10 échantillons.")
        return
    
    # Gérer les données de référence
    if not REFERENCE_PATH.exists():
        print("📝 Création des données de référence...")
        create_reference_data(prod_df)
        print("ℹ️  Le prochain rapport pourra comparer avec ces données.")
        return
    
    # Chargement des données de référence
    try:
        ref_df = pd.read_csv(REFERENCE_PATH)
        print(f"📚 Données de référence: {len(ref_df)} échantillons")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données de référence: {e}")
        return
    
    # Création du mapping des colonnes
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = ['confidence', 'feat_mean', 'feat_std']
    
    # Création du rapport
    print("\n📊 Génération du rapport de drift...")
    report = Report(metrics=[
        DataDriftPreset(),
        ColumnDriftMetric(column_name='confidence'),
        ColumnDriftMetric(column_name='feat_mean'),
        ColumnDriftMetric(column_name='feat_std'),
    ])
    
    try:
        report.run(
            reference_data=ref_df,
            current_data=prod_df,
            column_mapping=column_mapping
        )
        
        # Sauvegarde du rapport HTML
        today = datetime.date.today().isoformat()
        html_path = REPORTS_DIR / f'drift_report_{today}.html'
        report.save_html(str(html_path))
        print(f"✅ Rapport sauvegardé: {html_path}")
        
        # Sauvegarde en JSON
        json_path = REPORTS_DIR / f'drift_report_{today}.json'
        results = report.as_dict()
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Résultats JSON sauvegardés: {json_path}")
        
        # Analyse des résultats
        analyze_drift_results(results, prod_df)
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération du rapport: {e}")
        import traceback
        traceback.print_exc()

def analyze_drift_results(results, prod_df):
    """Analyse les résultats du drift"""
    
    print("\n" + "="*50)
    print("📈 ANALYSE DES RÉSULTATS")
    print("="*50)
    
    # Détection du drift global
    try:
    
        if 'metrics' in results and len(results['metrics']) > 0:
            first_metric = results['metrics'][0]
            if 'result' in first_metric:
                drift_result = first_metric['result']
                
                # Vérification du drift dataset
                dataset_drift = drift_result.get('dataset_drift', False)
                
                if dataset_drift:
                    print("⚠️  ALERTE: Data drift détecté!")
                    
                    # Affichage des colonnes avec drift
                    if 'drift_by_columns' in drift_result:
                        drifted_cols = []
                        for col, info in drift_result['drift_by_columns'].items():
                            if info.get('drift_detected', False):
                                drifted_cols.append(col)
                        
                        if drifted_cols:
                            print(f"   Colonnes affectées: {', '.join(drifted_cols)}")
                    
                    print("💡 Recommandation: Envisagez de réentraîner le modèle.")
                else:
                    print("✅ Pas de data drift détecté.")
                    print("   La distribution des données reste stable.")
    except Exception as e:
        print(f"⚠️  Impossible d'analyser les résultats: {e}")
    
    # Statistiques sur les prédictions récentes
    print(f"\n📊 Statistiques des {len(prod_df)} dernières prédictions:")
    print(f"   Confiance moyenne: {prod_df['confidence'].mean():.4f}")
    print(f"   Confiance std: {prod_df['confidence'].std():.4f}")
    
    # Distribution des classes prédites
    if 'predicted' in prod_df.columns:
        class_dist = prod_df['predicted'].value_counts()
        print(f"\n   Distribution des classes prédites:")
        for classe, count in class_dist.head(5).items():
            print(f"      {classe}: {count} ({count/len(prod_df)*100:.1f}%)")

def update_reference_data():
    """Met à jour les données de référence avec les prédictions récentes"""
    
    prod_df = load_and_prepare_data()
    if prod_df is None:
        return
    
    print("🔄 Mise à jour des données de référence...")
    
    if REFERENCE_PATH.exists():
        # Fusionner avec les données existantes
        ref_df = pd.read_csv(REFERENCE_PATH)
        updated_df = pd.concat([ref_df, prod_df]).drop_duplicates(subset=['timestamp'])
        
        # Garder seulement les 1000 derniers échantillons
        if len(updated_df) > 1000:
            updated_df = updated_df.tail(1000)
            print(f"   (Limité à 1000 échantillons)")
        
        updated_df.to_csv(REFERENCE_PATH, index=False)
        print(f"✅ Référence mise à jour: {len(updated_df)} échantillons")
    else:
        create_reference_data(prod_df)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyse de data drift')
    parser.add_argument('--update-ref', action='store_true',
                        help='Mettre à jour les données de référence')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Seuil de drift (défaut: 0.8)')
    
    args = parser.parse_args()
    
    if args.update_ref:
        update_reference_data()
    else:
        run_drift_report()