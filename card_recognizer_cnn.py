"""
CNN-based Card Recognizer
Uses trained EfficientNet model for card recognition
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import cv2
import numpy as np

class CNNCardRecognizer:
    """Card recognizer using trained CNN model"""
    
    def __init__(self, model_path='models/card_recognition_best.pth', 
                 data_dir='data/augmented_images'):
        """
        Initialize CNN card recognizer
        
        Args:
            model_path: Path to trained model
            data_dir: Path to dataset directory (for class mapping)
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading CNN Card Recognizer...")
        print(f"   Device: {self.device}")
        
        # Load class mapping
        self.class_to_idx, self.idx_to_class = self._load_class_mapping()
        self.num_classes = len(self.class_to_idx)
        print(f"   ✓ Loaded {self.num_classes} card classes")
        
        # Load model
        self.model = self._load_model()
        print(f"   ✓ Model loaded successfully!")
        
        # Define transforms (same as validation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_class_mapping(self):
        """Load class to index mapping"""
        class_to_idx_path = os.path.join(self.data_dir, 'class_to_idx.json')
        
        if not os.path.exists(class_to_idx_path):
            raise FileNotFoundError(f"Class mapping not found: {class_to_idx_path}")
        
        with open(class_to_idx_path, 'r') as f:
            class_to_idx = json.load(f)
        
        # Create reverse mapping
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        return class_to_idx, idx_to_class
    
    def _load_model(self):
        """Load the trained model"""
        # Create model architecture (MUST match training architecture)
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        
        # This MUST match the architecture in train_model.py
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        # Load weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def recognize_from_cv2(self, cv2_image, top_k=5):
        """
        Recognize card from OpenCV image (BGR format)
        
        Args:
            cv2_image: OpenCV image (numpy array, BGR format)
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples: [(card_name, confidence), ...]
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        return self.recognize(pil_image, top_k)
    
    def recognize(self, pil_image, top_k=5):
        """
        Recognize card from PIL Image
        
        Args:
            pil_image: PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples: [(card_name, confidence), ...]
        """
        # Ensure RGB mode
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
            
            results = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                card_name = self.idx_to_class[idx.item()]
                confidence = prob.item()
                results.append((card_name, confidence))
            
            return results
    
    def get_best_match(self, cv2_image):
        """
        Get the best matching card
        
        Args:
            cv2_image: OpenCV image (numpy array, BGR format)
            
        Returns:
            Tuple: (card_name, confidence) or (None, 0.0) if no good match
        """
        results = self.recognize_from_cv2(cv2_image, top_k=1)
        
        if results and len(results) > 0:
            return results[0]
        
        return None, 0.0
    
    def cleanup(self):
        """Clean up model resources"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Move model to CPU and delete
                self.model.cpu()
                del self.model
                self.model = None
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print("✓ CNN model cleaned up")
        except Exception as e:
            print(f"✗ Error cleaning up CNN model: {e}")


# Test function
def test_recognizer():
    """Test the recognizer with a sample image"""
    recognizer = CNNCardRecognizer()
    
    # Test with a sample image from augmented dataset
    test_image_dir = 'data/augmented_images'
    
    if not os.path.exists(test_image_dir):
        print("Test image directory not found")
        return
    
    # Get first few images
    images = [f for f in os.listdir(test_image_dir) if f.endswith('.jpg')][:5]
    
    if not images:
        print("No images found")
        return
    
    print(f"\n Testing with {len(images)} sample images\n")
    
    for img_file in images:
        test_image_path = os.path.join(test_image_dir, img_file)
        
        # Extract expected card ID from filename (format: cardid_augX.jpg)
        card_id = int(img_file.split('_')[0])
        
        print(f"Testing: {img_file}")
        print(f"   Card ID: {card_id}")
        
        # Load image
        cv2_image = cv2.imread(test_image_path)
        
        # Recognize
        results = recognizer.recognize_from_cv2(cv2_image, top_k=3)
        
        print("   Top 3 Predictions:")
        for i, (card_name, confidence) in enumerate(results, 1):
            print(f"      {i}. {card_name} ({confidence*100:.2f}%)")
        print()

if __name__ == "__main__":
    test_recognizer()

