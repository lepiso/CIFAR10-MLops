# app/model.py
import tensorflow as tf
from tensorflow.keras import layers, models

class SimpleCNN(tf.keras.Model):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        

        self.features = models.Sequential([
            layers.Conv2D(32, kernel_size=3, padding='same', activation=None),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=2, strides=2),
            
            layers.Conv2D(64, kernel_size=3, padding='same', activation=None),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=2, strides=2),
            
            layers.Conv2D(128, kernel_size=3, padding='same', activation=None),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=2, strides=2),
        ])
        
        
        self.classifier = models.Sequential([
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(num_classes)
        ])
    
    def call(self, x, training=False):
        x = self.features(x, training=training)
        return self.classifier(x, training=training)