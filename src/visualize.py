# src/visualize.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
from datetime import datetime

class ModelVisualizer:
    """Classe pour générer des visualisations du modèle"""
    
    def __init__(self, model, class_names, save_dir='reports/figures'):
        self.model = model
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_history(self, history, save=True):
        """Visualiser l'historique d'entraînement"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.save_dir}/training_history.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.save_dir}/training_history.pdf', bbox_inches='tight')
        
        plt.show()
        return fig
    
    def plot_confusion_matrix(self, x_test, y_test, save=True):
        """Générer la matrice de confusion"""
        # Prédictions
        y_pred = self.model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        
        # Calculer la matrice
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Visualiser
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    ax=ax)
        ax.set_title('Confusion Matrix', fontsize=16)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.save_dir}/confusion_matrix.pdf', bbox_inches='tight')
        
        plt.show()
        
        # Afficher le rapport de classification
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=self.class_names))
        
        return fig, cm
    
    def plot_per_class_accuracy(self, y_test, save=True):
        """Visualiser l'accuracy par classe"""
        y_pred = self.model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        
        # Calculer l'accuracy par classe
        per_class_acc = []
        for i in range(len(self.class_names)):
            mask = (y_true == i)
            if mask.sum() > 0:
                acc = (y_pred_classes[mask] == i).sum() / mask.sum()
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0)
        
        # Visualiser
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(self.class_names)), per_class_acc, color='steelblue')
        ax.set_xticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Class Accuracy', fontsize=14)
        ax.axhline(y=np.mean(per_class_acc), color='red', linestyle='--', 
                label=f'Mean Accuracy: {np.mean(per_class_acc):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs sur les barres
        for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.save_dir}/per_class_accuracy.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def plot_learning_curves(self, history, save=True):
        """Visualiser les courbes d'apprentissage détaillées"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss Curves', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Accuracy Curves', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (si disponible)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'], color='green', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting detection
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        overfitting = [val - train for val, train in zip(val_loss, train_loss)]
        
        axes[1, 1].plot(overfitting, color='red', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Overfitting Detection (Val Loss - Train Loss)', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Difference')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].fill_between(range(len(overfitting)), 0, overfitting, 
                        where=np.array(overfitting) > 0, 
                        color='red', alpha=0.3, label='Overfitting Zone')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.save_dir}/learning_curves.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def plot_predictions(self, x_test, y_test, num_samples=10, save=True):
        """Visualiser quelques prédictions du modèle"""
        y_pred = self.model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        # Sélectionner des échantillons aléatoires
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        
        for idx, ax in enumerate(axes):
            i = indices[idx]
            img = x_test[i]
            
            # Dé-normaliser pour l'affichage
            img_display = np.clip(img, 0, 1)
            
            ax.imshow(img_display)
            true_class = self.class_names[y_true[i]]
            pred_class = self.class_names[y_pred_classes[i]]
            confidence = np.max(y_pred[i])
            
            color = 'green' if true_class == pred_class else 'red'
            ax.set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}', 
                        color=color, fontsize=10)
            ax.axis('off')
        
        plt.suptitle('Model Predictions', fontsize=16)
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.save_dir}/predictions.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig