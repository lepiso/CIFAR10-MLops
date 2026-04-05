import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric

class ModelMonitor:
    def __init__(self, logs_path='logs/predictions.jsonl', 
                reference_path='data/reference_stats.csv',
                reports_dir='reports/monitoring'):
        self.logs_path = Path(logs_path)
        self.reference_path = Path(reference_path)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Charger les données de prédictions"""
        if not self.logs_path.exists():
            return None
        
        df = pd.read_json(self.logs_path, lines=True)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def create_reference(self, df):
        """Créer les données de référence"""
        if len(df) < 10:
            print(f"⚠️ Pas assez de données ({len(df)}), besoin de 10")
            return False
        
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.reference_path, index=False)
        
        reference = {
            'created_at': datetime.now().isoformat(),
            'num_samples': len(df),
            'confidence_mean': df['confidence'].mean(),
            'confidence_std': df['confidence'].std(),
            'class_distribution': df['predicted'].value_counts().to_dict()
        }
        
        with open(self.reports_dir / 'reference_metadata.json', 'w') as f:
            json.dump(reference, f, indent=2)
        
        print(f"✅ Référence créée avec {len(df)} échantillons")
        return True
    
    def detect_drift(self, current_df):
        """Détecter le drift"""
        if not self.reference_path.exists():
            print("❌ Pas de référence")
            return None
        
        ref_df = pd.read_csv(self.reference_path)
        
        report = Report(metrics=[
            DataDriftPreset(),
            ColumnDriftMetric(column_name='confidence')
        ])
        
        report.run(reference_data=ref_df, current_data=current_df)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.reports_dir / f'drift_report_{timestamp}.html'
        report.save_html(str(report_path))
        
        results = report.as_dict()
        drift_detected = False
        drift_score = 0
        
        if 'metrics' in results and len(results['metrics']) > 0:
            metric = results['metrics'][0]
            if 'result' in metric:
                drift_detected = metric['result'].get('dataset_drift', False)
                if 'drift_by_columns' in metric['result']:
                    for col, info in metric['result']['drift_by_columns'].items():
                        if info.get('drift_detected', False):
                            drift_score = info.get('drift_score', 0)
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'report_path': str(report_path),
            'timestamp': timestamp
        }
    
    def run(self):
        """Exécuter la surveillance"""
        print(f"🔍 Analyse de drift - {datetime.now()}")
        
        current_df = self.load_data()
        if current_df is None or len(current_df) == 0:
            print("❌ Aucune donnée")
            return None
        
        if not self.reference_path.exists() and len(current_df) >= 10:
            self.create_reference(current_df)
            return None
        
        if self.reference_path.exists():
            result = self.detect_drift(current_df)
            if result and result['drift_detected']:
                print(f"⚠️ DRIFT DÉTECTÉ! Score: {result['drift_score']:.4f}")
                return result
            else:
                print("✅ Pas de drift détecté")
                return result
        
        return None

if __name__ == '__main__':
    monitor = ModelMonitor()
    monitor.run()
