---
title: CIFAR-10 Image Classifier
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🖼️ CIFAR-10 Image Classifier

Classification d'images en 10 catégories avec un réseau de neurones convolutif.

## 📊 Classes disponibles

| Classe | Emoji |
|--------|-------|
| Airplane | ✈️ |
| Automobile | 🚗 |
| Bird | 🐦 |
| Cat | 🐱 |
| Deer | 🦌 |
| Dog | 🐶 |
| Frog | 🐸 |
| Horse | 🐴 |
| Ship | 🚢 |
| Truck | 🚛 |

## 🔍 XAI - Explicabilité

L'application intègre des techniques d'explicabilité :
- **LIME** : zones importantes dans l'image
- **Grad-CAM** : heatmap des activations
- **SHAP** : importance des pixels

## 🚀 Utilisation

1. Allez dans la page **Prédiction**
2. Téléchargez une image (JPG, PNG, JPEG)
3. Obtenez la classification instantanée
4. Explorez les explications dans la page **XAI**

## 📈 Performance

- Accuracy: ~79% sur CIFAR-10
- Architecture: CNN avec 3 couches convolutives
