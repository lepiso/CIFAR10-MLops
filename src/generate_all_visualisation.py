import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from app.model import SimpleCNN

def load_model_custom():
    """Charger le modèle correctement"""
    
    # Méthode 1: Charger depuis SavedModel
    if os.path.exists('model_savedmodel'):
        model = tf.keras.models.load_model('model_savedmodel', custom_objects={'SimpleCNN': SimpleCNN})
        print("✅ Modèle chargé depuis SavedModel")
        return model
    
    # Méthode 2: Charger depuis les poids .h5
    elif os.path.exists('model_weights.h5'):
        with open('params.yaml') as f:
            params = yaml.safe_load(f)
        
        # Recréation du modèle
        model = SimpleCNN(
            num_classes=params['model']['num_classes'],
            dropout=params['model']['dropout']
        )
        
        # Build du modèle
        dummy_input = tf.zeros((1, 32, 32, 3))
        _ = model(dummy_input, training=False)
        
        # Chargement des poids
        model.load_weights('model_weights.h5')
        print("✅ Modèle chargé depuis model_weights.h5")
        return model
    
    # Méthode 3: Chargement depuis .pkl
    elif os.path.exists('model.pkl'):
        with open('params.yaml') as f:
            params = yaml.safe_load(f)
        
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Recréation du modèle
        model = SimpleCNN(
            num_classes=model_data['config']['num_classes'],
            dropout=model_data['config']['dropout']
        )
        
        # Build du modèle
        dummy_input = tf.zeros((1, 32, 32, 3))
        _ = model(dummy_input, training=False)
        
        # Chargement des poids
        model.set_weights(model_data['weights'])
        print("✅ Modèle chargé depuis model.pkl")
        return model
    
    else:
        raise FileNotFoundError("Aucun modèle trouvé!")

def main():
    # Création du dossier pour les figures
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('reports/metrics', exist_ok=True)
    
    # 1. Chargement des paramètres
    print("📂 Chargement des paramètres...")
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    
    # 2. Chargement des données
    print("📂 Chargement des données CIFAR-10...")
    (_, _), (x_test, y_test) = datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    mean = params['data']['normalize_mean']
    std = params['data']['normalize_std']
    x_test = (x_test - mean) / std
    
    # 3. Chargement du modèle
    print("🤖 Chargement du modèle...")
    try:
        model = load_model_custom()
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 4. Faire les prédictions
    print("🔮 Génération des prédictions...")
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = y_test.flatten()
    
    # 5. Matrice de confusion
    print("📊 Génération de la matrice de confusion...")
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion', fontsize=16)
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Vérité Terrain', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 6. Accuracy par classe
    print("📊 Génération de l'accuracy par classe...")
    per_class_acc = []
    for i in range(10):
        mask = (y_true == i)
        if mask.sum() > 0:
            acc = (y_pred_classes[mask] == i).sum() / mask.sum()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(10), per_class_acc, color='steelblue')
    plt.xticks(range(10), class_names, rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy par Classe', fontsize=14)
    plt.axhline(y=np.mean(per_class_acc), color='red', linestyle='--',
                label=f'Accuracy Moyenne: {np.mean(per_class_acc):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Ajout des valeurs sur les barres
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('reports/figures/per_class_accuracy.png', dpi=300)
    plt.close()
    
    # 7. Exemples de prédictions
    print("📊 Génération des exemples de prédictions...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    indices = np.random.choice(len(x_test), 10, replace=False)
    
    for idx, ax in enumerate(axes):
        i = indices[idx]
        img = x_test[i]
        # Dé-normaliser pour l'affichage
        img_display = np.clip(img, 0, 1)
        ax.imshow(img_display)
        true_class = class_names[y_true[i]]
        pred_class = class_names[y_pred_classes[i]]
        confidence = np.max(y_pred[i])
        color = 'green' if true_class == pred_class else 'red'
        ax.set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}',
                    color=color, fontsize=9)
        ax.axis('off')
    
    plt.suptitle('Exemples de Prédictions du Modèle', fontsize=16)
    plt.tight_layout()
    plt.savefig('reports/figures/predictions.png', dpi=300)
    plt.close()
    
    # 8. Rapport de classification
    print("\n📈 Rapport de Classification:")
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print(report)
    
    # Sauvegarde des métriques
    with open('reports/metrics/classification_report.txt', 'w') as f:
        f.write("=== RAPPORT D'ÉVALUATION ===\n\n")
        f.write(f"Test Accuracy: {np.mean(y_pred_classes == y_true):.4f}\n\n")
        f.write("=== CLASSIFICATION REPORT ===\n")
        f.write(report)
    
    # 9. Évaluation finale
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ Test Loss: {test_loss:.4f}")
    print("\n✅ Toutes les visualisations ont été sauvegardées dans reports/figures/")

if __name__ == '__main__':
    main()