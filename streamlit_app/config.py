import os
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = os.environ.get('MODEL_PATH', str(BASE_DIR / 'model.h5'))
LOGS_PATH = BASE_DIR / 'logs' / 'predictions.jsonl'
REPORTS_DIR = BASE_DIR / 'reports' / 'figures'
DATA_DIR = BASE_DIR / 'data'

# Classes CIFAR-10
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck']

# Paramètres de normalisation
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]

# Configuration Streamlit
STREAMLIT_CONFIG = {
    'page_title': 'CIFAR-10 Image Classifier',
    'page_icon': '🖼️',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

XAI_METHODS = ['LIME', 'Grad-CAM', 'SHAP', 'Integrated Gradients']
