import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from app.model import SimpleCNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, optimizers, losses
from src.visualize import ModelVisualizer


def get_data_generators(params):
    """Create data generators with augmentation for training"""
    mean = params['data']['normalize_mean']
    std = params['data']['normalize_std']
    
    # Normalisation
    def normalize(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - mean) / std
        return image, label
    
    # Data augmentation
    train_datagen = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomCrop(32, 32, padding=4),
        layers.Rescaling(1./255),
        layers.Normalization(mean=mean, variance=[s**2 for s in std])
    ])
    
    return train_datagen, normalize


def main():
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    # Normalisation pixels
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    mean = params['data']['normalize_mean']
    std = params['data']['normalize_std']
    
    def normalize(x, y):
        x = (x - mean) / std
        return x, y
    
    val_split = params['train']['val_split']
    val_size = int(len(x_train) * val_split)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train[:-val_size], y_train[:-val_size]))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_train[-val_size:], y_train[-val_size:]))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, 32+8, 32+8)
        image = tf.image.random_crop(image, [32, 32, 3])
        return image, label
    
    train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    
    batch_size = params['train']['batch_size']
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    model = SimpleCNN(
        num_classes=params['model']['num_classes'],
        dropout=params['model']['dropout']
    )
    
    # Compilation
    optimizer = optimizers.Adam(learning_rate=params['train']['learning_rate'])
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # Learning rate 
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.1, patience=7, min_lr=1e-6
    )
    
    class MLflowCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mlflow.log_metric('train_loss', logs['loss'], step=epoch)
            mlflow.log_metric('val_accuracy', logs['val_accuracy'], step=epoch)
            print(f'Epoch {epoch+1:02d} | loss={logs["loss"]:.4f} | val_acc={logs["val_accuracy"]:.4f}')
    
    # Start MLflow 
    with mlflow.start_run():
        mlflow.log_params(params['train'])
        mlflow.log_params(params['model'])
        
        # Entrainement du  model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=params['train']['epochs'],
            callbacks=[reduce_lr, MLflowCallback()],
            verbose=0  # We'll print manually in callback
        )
        
        # Enregistrement du modèle
        os.makedirs('data/processed', exist_ok=True)
        model.save('model.h5')
        
        # Log model with MLflow
        mlflow.tensorflow.log_model(model, 'model')
        mlflow.log_artifact('model.h5')
        
        print('Modèle sauvegardé dans model.h5')

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    visualizer = ModelVisualizer(model, class_names, save_dir='reports/figures')
    visualizer.plot_training_history(history, save=True)
    (_, _), (x_test, y_test) = datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    mean = params['data']['normalize_mean']
    std = params['data']['normalize_std']
    x_test = (x_test - mean) / std
    
    y_test_onehot = to_categorical(y_test, num_classes=params['model']['num_classes'])
    
    # 3. Matrice de confusion
    visualizer.plot_confusion_matrix(x_test, y_test_onehot, save=True)
    
    # 4. Accuracy par classe
    visualizer.plot_per_class_accuracy(x_test, y_test_onehot, save=True)
    
    # 5. Courbes d'apprentissage détaillées
    visualizer.plot_learning_curves(history, save=True)
    
    # 6. Exemples de prédictions
    visualizer.plot_predictions(x_test, y_test_onehot, num_samples=10, save=True)
    
    # Log des figures avec MLflow
    with mlflow.start_run():
        # ... (code MLflow existant) ...
        
        # Log des figures
        for fig_file in os.listdir('reports/figures'):
            if fig_file.endswith(('.png', '.pdf')):
                mlflow.log_artifact(f'reports/figures/{fig_file}')

if __name__ == '__main__':
    main()