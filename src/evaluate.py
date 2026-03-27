import argparse
import sys
import os
import yaml
import tensorflow as tf
from tensorflow.keras import datasets
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import SimpleCNN


def evaluate(threshold=0.80):
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    
    #Chargement du dataset CIFAR10
    (_, _), (x_test, y_test) = datasets.cifar10.load_data()
    
    # Normalisation
    x_test = x_test.astype('float32') / 255.0
    mean = params['data']['normalize_mean']
    std = params['data']['normalize_std']
    x_test = (x_test - mean) / std
    
    #Chargement du model
    model = tf.keras.models.load_model('model.h5', custom_objects={'SimpleCNN': SimpleCNN})
    
    # Evaluation du model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    
    print(f'Test accuracy: {test_accuracy:.4f}')
    
    if test_accuracy < threshold:
        print(f'ECHEC: accuracy {test_accuracy:.4f} < seuil {threshold}')
        sys.exit(1)
    else:
        print(f'OK: accuracy {test_accuracy:.4f} >= seuil {threshold}')
    
    return test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.80)
    args = parser.parse_args()
    evaluate(args.threshold)