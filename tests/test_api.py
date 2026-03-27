import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck']


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    print("✅ Health OK")


def test_model_info():
    response = client.get('/model_info')
    assert response.status_code == 200
    data = response.json()
    assert 'classes' in data
    print(f"✅ Model info: {len(data['classes'])} classes")


def test_predict():
    img = Image.new('RGB', (32, 32), color='blue')
    pixels = np.array(img)
    pixels[10:22, 10:22] = [255, 0, 0]
    img = Image.fromarray(pixels)
    
    os.makedirs('tests', exist_ok=True)
    img.save('tests/test_img.jpg')
    
    with open('tests/test_img.jpg', 'rb') as f:
        response = client.post(
            '/predict',
            files={'file': ('test.jpg', f, 'image/jpeg')}
        )
    
    os.remove('tests/test_img.jpg')
    
    assert response.status_code == 200
    data = response.json()
    assert data['class'] in CLASSES
    print(f"✅ Prédiction: {data['class']} (conf: {data['confidence']:.4f})")


def test_invalid_file():
    response = client.post(
        '/predict',
        files={'file': ('test.txt', b'not an image', 'text/plain')}
    )
    assert response.status_code == 400
    print("✅ Fichier invalide rejeté")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
