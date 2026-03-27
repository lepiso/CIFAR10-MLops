import os
import sys
import json
import datetime
import pickle
import yaml
from pathlib import Path
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.model import SimpleCNN
    print("✅ SimpleCNN importée avec succès")
except ImportError as e:
    print(f"⚠️ Erreur d'import SimpleCNN: {e}")
    #Classe SimpleCNN
    class SimpleCNN(tf.keras.Model):
        def __init__(self, num_classes=10, dropout=0.3, **kwargs):
            super().__init__(**kwargs)
            self.num_classes = num_classes
            self.dropout = dropout
            self.features = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(2),
                tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(2),
                tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(2),
            ])
            self.classifier = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(num_classes)
            ])
        
        def call(self, x, training=False):
            x = self.features(x, training=training)
            return self.classifier(x, training=training)

# Constantes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck']

# Chemin des logs
LOGS_PATH = Path(os.environ.get('LOGS_PATH', 'logs/predictions.jsonl'))

# Paramètres de normalisation
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def load_params():
    """Charge les paramètres depuis params.yaml"""
    try:
        with open('params.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {
            'model': {'num_classes': 10, 'dropout': 0.3},
            'data': {'normalize_mean': MEAN, 'normalize_std': STD}
        }


@st.cache_resource
def load_model():
    """Charge le modèle en reconstruisant l'architecture"""
    try:
        params = load_params()
        num_classes = params['model']['num_classes']
        dropout = params['model']['dropout']
        
        print(f"🔄 Reconstruction du modèle avec num_classes={num_classes}, dropout={dropout}")
        
        # Recréer le modèle
        model = SimpleCNN(num_classes=num_classes, dropout=dropout)
        
        dummy_input = tf.zeros((1, 32, 32, 3))
        _ = model(dummy_input, training=False)
        
        # Chargement des poids depuis le fichier .h5
        if os.path.exists('model.h5'):
            try:
                #Chargement direct des poids
                model.load_weights('model.h5')
                print("✅ Modèle chargé avec load_weights depuis model.h5")
                return model
            except Exception as e:
                print(f"⚠️ load_weights a échoué: {e}")
                
                #Chargement avec temp_model et extraction du poids
                try:
                    temp_model = tf.keras.models.load_model(
                        'model.h5',
                        custom_objects={'SimpleCNN': SimpleCNN},
                        compile=False
                    )
                    model.set_weights(temp_model.get_weights())
                    print("✅ Modèle chargé via reconstruction des poids")
                    return model
                except Exception as e2:
                    print(f"⚠️ Échec de la méthode alternative: {e2}")
        
        # Essayer avec model.pkl
        if os.path.exists('model.pkl'):
            try:
                with open('model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                if 'weights' in model_data:
                    model.set_weights(model_data['weights'])
                    print("✅ Modèle chargé depuis model.pkl")
                    return model
                elif 'model_weights' in model_data:
                    model.set_weights(model_data['model_weights'])
                    print("✅ Modèle chargé depuis model.pkl")
                    return model
            except Exception as e:
                print(f"⚠️ Erreur chargement model.pkl: {e}")
        
        # Dernier essai: charger en SavedModel
        if os.path.exists('model_savedmodel'):
            try:
                model = tf.keras.models.load_model('model_savedmodel', compile=False)
                print("✅ Modèle chargé depuis SavedModel")
                return model
            except Exception as e:
                print(f"⚠️ Erreur SavedModel: {e}")
        
        st.error("❌ Aucun modèle trouvé ou chargé!")
        return None
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Prétraite l'image pour le modèle"""
    img = img.convert("RGB").resize((32, 32))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.expand_dims(arr, axis=0)


def predict(model, img: Image.Image) -> dict:
    """Fait une prédiction sur une image"""
    if model is None:
        return {
            "class": "Erreur",
            "confidence": 0.0,
            "all_probs": {c: 0.0 for c in CLASSES}
        }
    
    try:
        arr = preprocess_image(img)
        logits = model.predict(arr, verbose=0)
        probs = tf.nn.softmax(logits[0]).numpy()
        idx = int(np.argmax(probs))
        
        return {
            "class": CLASSES[idx],
            "confidence": float(probs[idx]),
            "all_probs": {c: float(p) for c, p in zip(CLASSES, probs)},
        }
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return {
            "class": "Erreur",
            "confidence": 0.0,
            "all_probs": {c: 0.0 for c in CLASSES}
        }


def log_prediction(result: dict):
    """Sauvegarde la prédiction dans les logs"""
    try:
        LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "predicted": result["class"],
            "confidence": round(result["confidence"], 4),
        }
        with open(LOGS_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Erreur lors du logging: {e}")


@st.cache_data(ttl=60)
def load_history_as_dataframe():
    """Charge l'historique comme DataFrame pandas"""
    import pandas as pd
    
    if not LOGS_PATH.exists():
        return pd.DataFrame()
    
    try:
        history = []
        with open(LOGS_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return pd.DataFrame()


def get_stats(history):
    """Calcule les statistiques des prédictions"""
    if not history:
        return {
            'total': 0,
            'avg_confidence': 0.0,
            'most_common_class': 'N/A',
            'most_common_count': 0,
            'unique_classes': 0
        }
    
    confidences = [e.get('confidence', 0) for e in history if 'confidence' in e]
    
    from collections import Counter
    classes = [e.get('predicted') for e in history if 'predicted' in e]
    class_counts = Counter(classes)
    most_common = class_counts.most_common(1)[0] if class_counts else ('N/A', 0)
    
    return {
        'total': len(history),
        'avg_confidence': np.mean(confidences) if confidences else 0.0,
        'most_common_class': most_common[0],
        'most_common_count': most_common[1],
        'unique_classes': len(class_counts)
    }


def get_class_distribution(history):
    """Calcule la distribution des classes"""
    if not history:
        return {}
    
    from collections import Counter
    classes = [e.get('predicted') for e in history if 'predicted' in e]
    return dict(Counter(classes))


def clear_history():
    """Efface l'historique des prédictions"""
    try:
        if LOGS_PATH.exists():
            LOGS_PATH.unlink()
        return True
    except Exception as e:
        print(f"Erreur lors de l'effacement: {e}")
        return False


def load_css():
    """Charge le fichier CSS personnalisé"""
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Prétraite l'image pour le modèle (batch size 1)"""
    img = img.convert("RGB").resize((32, 32))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.expand_dims(arr, axis=0)