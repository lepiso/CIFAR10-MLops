import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from app.model import SimpleCNN
from src.visualize import ModelVisualizer

def evaluate_advanced():
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    
    # Charger les données
    (_, _), (x_test, y_test) = datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    mean = params['data']['normalize_mean']
    std = params['data']['normalize_std']
    x_test = (x_test - mean) / std
    
    # Charger le modèle
    model = tf.keras.models.load_model('model.h5', custom_objects={'SimpleCNN': SimpleCNN})
    
    # Noms des classes
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Créer le visualiseur
    visualizer = ModelVisualizer(model, class_names, save_dir='reports/figures')
    
    # Convertir les labels en one-hot
    y_test_onehot = to_categorical(y_test, num_classes=10)
    
    # Générer toutes les visualisations
    print("Génération de la matrice de confusion...")
    visualizer.plot_confusion_matrix(x_test, y_test_onehot, save=True)
    
    print("Génération de l'accuracy par classe...")
    visualizer.plot_per_class_accuracy(x_test, y_test_onehot, save=True)
    
    print("Génération des prédictions...")
    visualizer.plot_predictions(x_test, y_test_onehot, num_samples=15, save=True)
    
    print("\n✅ Toutes les visualisations ont été sauvegardées dans reports/figures/")
    
    # Évaluation finale
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    evaluate_advanced()