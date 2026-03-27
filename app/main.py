import os
import io
import json
import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from app.model import SimpleCNN

# ── Initialisation ────────────────────────────────────────────
app = FastAPI(
    title='CIFAR-10 CNN API',
    description='Classification d images en 10 categories',
    version='1.0.0'
)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck']

MODEL_PATH = os.environ.get('MODEL_PATH', 'model.h5')
LOGS_PATH = Path('logs/predictions.jsonl')
LOGS_PATH.parent.mkdir(exist_ok=True)

# ── Chargement du modele au demarrage ────────────────────────────
def load_model_custom(model_path):
    """Charge le modèle en reconstruisant l'architecture"""
    
    # Chargement des paramètres
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    
    # Recréation du modèle
    print(f"🔄 Reconstruction du modèle avec num_classes={params['model']['num_classes']}, dropout={params['model']['dropout']}")
    model = SimpleCNN(
        num_classes=params['model']['num_classes'],
        dropout=params['model']['dropout']
    )
    
    # Construction du modèle
    dummy_input = tf.zeros((1, 32, 32, 3))
    _ = model(dummy_input, training=False)
    
    # Chargement des poids
    print(f"📂 Chargement des poids depuis {model_path}...")
    model.load_weights(model_path)
    
    # Compilation pour l'inférence
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print("✅ Modèle chargé avec succès!")
    return model, params

# Chargement du modèle
print("🤖 Chargement du modèle...")
model, params = load_model_custom(MODEL_PATH)

# Normalization parameters (pareil que  training)
MEAN = params['data']['normalize_mean']
STD = params['data']['normalize_std']

def preprocess_image(image):
    """Preprocess PIL image for model input"""
    image = image.resize((32, 32))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = (img_array - MEAN) / STD
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_features(model, input_tensor):
    """Extract features from the CNN layers"""
    try:
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer('features').output
        )
        features = feature_extractor(input_tensor, training=False)
        features = tf.reduce_mean(features, axis=[1, 2])
        return features.numpy()[0].tolist()
    except:
        return [0.0] * 128


# ── Endpoints ─────────────────────────────────────────────────
@app.get('/health')
def health():
    return {'status': 'ok', 'model': MODEL_PATH}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, 'Le fichier doit etre une image')

    try:
        img = Image.open(io.BytesIO(await file.read())).convert('RGB')
    except Exception:
        raise HTTPException(400, 'Image invalide ou corrompue')

    tensor = preprocess_image(img)
    logits = model.predict(tensor, verbose=0)
    probs = tf.nn.softmax(logits[0]).numpy()
    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = round(float(probs[pred_idx]), 4)

    features = get_features(model, tensor)
    
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'predicted': pred_class,
        'confidence': confidence,
        'feat_mean': round(sum(features) / len(features), 4),
        'feat_std': round(float(np.std(features)), 4),
    }
    
    with open(LOGS_PATH, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

    return {
        'class': pred_class,
        'confidence': confidence,
        'all_probs': {c: round(float(p), 4)
                    for c, p in zip(CLASSES, probs)},
    }


@app.get('/model_info')
def model_info():
    """Get information about the loaded model"""
    return {
        'model_path': MODEL_PATH,
        'classes': CLASSES,
        'input_shape': (32, 32, 3),
        'normalization_mean': MEAN,
        'normalization_std': STD
    }