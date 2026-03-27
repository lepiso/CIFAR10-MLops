import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, optimizers, losses
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from app.model import SimpleCNN

def load_model_with_weights():
    """Charger le modèle en reconstruisant l'architecture"""
    
    # Chargement les paramètres
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    
    # Recréation du modèle avec la même architecture
    model = SimpleCNN(
        num_classes=params['model']['num_classes'],
        dropout=params['model']['dropout']
    )
    
    # Construction du modèle avec une entrée factice
    dummy_input = tf.zeros((1, 32, 32, 3))
    _ = model(dummy_input, training=False)
    
    # Chargement des poids depuis le fichier .h5
    model.load_weights('model.h5')
    
    # Compilation du modèle pour l'évaluation
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print("✅ Modèle chargé et compilé avec succès!")
    return model, params

def main():
    # Création des dossiers
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('reports/metrics', exist_ok=True)
    
    # Chargement du modèle
    print("🤖 Chargement du modèle...")
    model, params = load_model_with_weights()
    
    # Chargement des données
    print("📂 Chargement des données CIFAR-10...")
    (_, _), (x_test, y_test) = datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    mean = params['data']['normalize_mean']
    std = params['data']['normalize_std']
    x_test = (x_test - mean) / std
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Faire les prédictions
    print("🔮 Génération des prédictions...")
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = y_test.flatten()
    
    # 1. Matrice de confusion
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
    print("  ✓ Matrice de confusion sauvegardée")
    
    # 2. Accuracy par classe
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
    
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('reports/figures/per_class_accuracy.png', dpi=300)
    plt.close()
    print("  ✓ Accuracy par classe sauvegardée")
    
    # 3. Exemples de prédictions
    print("📊 Génération des exemples de prédictions...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    indices = np.random.choice(len(x_test), 10, replace=False)
    
    for idx, ax in enumerate(axes):
        i = indices[idx]
        img = x_test[i]
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
    print("  ✓ Exemples de prédictions sauvegardés")
    
    # 4. Rapport de classification
    print("\n📈 Rapport de Classification:")
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print(report)
    
    # Sauvegarde des métriques
    with open('reports/metrics/classification_report.txt', 'w') as f:
        f.write("=== RAPPORT D'ÉVALUATION ===\n\n")
        f.write(f"Test Accuracy: {np.mean(y_pred_classes == y_true):.4f}\n\n")
        f.write("=== CLASSIFICATION REPORT ===\n")
        f.write(report)
    
    # 5. Évaluation finale (maintenant avec modèle compilé)
    print("\n📊 Évaluation finale...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    # 6. Analyse détaillée par classe
    print("\n📊 Analyse détaillée par classe:")
    for i, (class_name, acc) in enumerate(zip(class_names, per_class_acc)):
        print(f"  {class_name:12s}: {acc:.3f}")
    
    best_class = np.argmax(per_class_acc)
    worst_class = np.argmin(per_class_acc)
    print(f"\n🏆 Meilleure classe: {class_names[best_class]} ({per_class_acc[best_class]:.3f})")
    print(f"⚠️  Classe la moins bien classifiée: {class_names[worst_class]} ({per_class_acc[worst_class]:.3f})")
    
    # 7. Sauvegarde des métriques dans un fichier JSON
    import json
    metrics = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'per_class_accuracy': {class_names[i]: float(per_class_acc[i]) for i in range(10)},
        'mean_accuracy': float(np.mean(per_class_acc)),
        'best_class': class_names[best_class],
        'worst_class': class_names[worst_class]
    }
    
    with open('reports/metrics/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\n✅ Métriques sauvegardées dans reports/metrics/metrics.json")
    
    print("\n🎉 Visualisation terminée avec succès!")
    print("📁 Fichiers générés:")
    print("  - reports/figures/confusion_matrix.png")
    print("  - reports/figures/per_class_accuracy.png")
    print("  - reports/figures/predictions.png")
    print("  - reports/metrics/classification_report.txt")
    print("  - reports/metrics/metrics.json")

if __name__ == '__main__':
    main()