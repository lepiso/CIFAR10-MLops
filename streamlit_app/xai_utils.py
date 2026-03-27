import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class XAIExplainer:
    """Version simplifiée avec seulement LIME"""
    
    def __init__(self, model, class_names, preprocess_fn):
        self.model = model
        self.class_names = class_names
        self.preprocess = preprocess_fn
        
    def explain_lime(self, image):
        """Explication LIME"""
        try:
            import lime
            import lime.lime_image
        except ImportError:
            print("LIME non installé")
            return None
            
        explainer = lime.lime_image.LimeImageExplainer()
        
        def predict_fn(images):
            processed = []
            for img in images:
                img_pil = Image.fromarray((img * 255).astype('uint8'))
                processed.append(self.preprocess(img_pil))
            batch = np.concatenate(processed, axis=0)
            return self.model.predict(batch, verbose=0)
        
        img_array = np.array(image) / 255.0
        
        explanation = explainer.explain_instance(
            img_array, predict_fn, top_labels=1, hide_color=0, num_samples=500
        )
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=10, 
            hide_rest=False
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img_array)
        axes[0].set_title('Image originale')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='hot', alpha=0.7)
        axes[1].imshow(img_array, alpha=0.3)
        axes[1].set_title('Zones importantes pour la prédiction')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
