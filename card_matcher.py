"""
Card Recognition using Image Similarity Matching
Uses perceptual hashing and feature extraction to match cards
"""
import cv2
import numpy as np
import pandas as pd
import os
from typing import Tuple, Optional, Dict
import pickle
from pathlib import Path
import imagehash
from PIL import Image

class CardMatcher:
    """Match cards using image similarity"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'card_images')
        self.cards_df = None
        self.image_hashes = {}
        self.image_features = {}
        self.cache_file = os.path.join(data_dir, 'card_features_cache.pkl')
        
        # Load database
        self.load_database()
        
        # Load or build feature cache
        self.load_or_build_cache()
    
    def load_database(self):
        """Load card database"""
        csv_path = os.path.join(self.data_dir, 'cards.csv')
        if os.path.exists(csv_path):
            self.cards_df = pd.read_csv(csv_path)
            print(f"✓ Loaded {len(self.cards_df)} cards from database")
        else:
            print("✗ Card database not found!")
    
    def compute_phash(self, image_path: str) -> str:
        """Compute perceptual hash of image"""
        try:
            img = Image.open(image_path)
            return str(imagehash.phash(img, hash_size=16))
        except:
            return None
    
    def compute_features(self, image_path: str) -> Optional[np.ndarray]:
        """Compute feature vector from image using ORB"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Resize to standard size
            img = cv2.resize(img, (200, 291))  # Yu-Gi-Oh card aspect ratio
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Compute histogram as feature
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Compute color histogram
            color_hist = []
            for i in range(3):
                h = cv2.calcHist([img], [i], None, [32], [0, 256])
                h = cv2.normalize(h, h).flatten()
                color_hist.extend(h)
            
            # Combine features
            features = np.concatenate([hist, color_hist])
            
            return features
        except Exception as e:
            print(f"Error computing features for {image_path}: {e}")
            return None
    
    def build_feature_cache(self):
        """Build feature cache for all card images"""
        print("Building feature cache for card images...")
        
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        total = len(image_files)
        
        print(f"Processing {total} images...")
        
        for idx, filename in enumerate(image_files):
            if idx % 500 == 0:
                print(f"  Progress: {idx}/{total} ({idx*100//total}%)")
            
            card_id = int(filename.replace('.jpg', ''))
            image_path = os.path.join(self.images_dir, filename)
            
            # Compute hash
            phash = self.compute_phash(image_path)
            if phash:
                self.image_hashes[card_id] = phash
            
            # Compute features
            features = self.compute_features(image_path)
            if features is not None:
                self.image_features[card_id] = features
        
        print(f"✓ Built cache for {len(self.image_features)} cards")
        
        # Save cache
        self.save_cache()
    
    def save_cache(self):
        """Save feature cache to disk"""
        cache_data = {
            'hashes': self.image_hashes,
            'features': self.image_features
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"✓ Saved cache to {self.cache_file}")
    
    def load_cache(self):
        """Load feature cache from disk"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            self.image_hashes = cache_data['hashes']
            self.image_features = cache_data['features']
            print(f"✓ Loaded cache with {len(self.image_features)} cards")
            return True
        return False
    
    def load_or_build_cache(self):
        """Load cache if exists, otherwise build it"""
        if not self.load_cache():
            self.build_feature_cache()
    
    def match_card(self, card_image: np.ndarray, top_k=5) -> list:
        """
        Match a card image against the database
        
        Args:
            card_image: Extracted card image from camera
            top_k: Number of top matches to return
            
        Returns:
            List of (card_id, similarity_score, card_info) tuples
        """
        # Compute features for input image
        # Save temp image
        temp_path = '/tmp/temp_card.jpg'
        cv2.imwrite(temp_path, card_image)
        
        input_features = self.compute_features(temp_path)
        if input_features is None:
            return []
        
        # Compare with all cards
        similarities = []
        for card_id, features in self.image_features.items():
            # Compute cosine similarity
            similarity = np.dot(input_features, features) / (
                np.linalg.norm(input_features) * np.linalg.norm(features)
            )
            similarities.append((card_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top K matches
        top_matches = []
        for card_id, score in similarities[:top_k]:
            # Get card info from database
            card_info = self.get_card_info(card_id)
            if card_info:
                top_matches.append((card_id, score, card_info))
        
        return top_matches
    
    def get_card_info(self, card_id: int) -> Optional[Dict]:
        """Get card information from database"""
        if self.cards_df is None:
            return None
        
        result = self.cards_df[self.cards_df['id'] == card_id]
        if not result.empty:
            return result.iloc[0].to_dict()
        
        return None


def test_matcher():
    """Test the card matcher"""
    print("Testing Card Matcher...")
    
    matcher = CardMatcher()
    
    # Test with a random card image
    test_images = [f for f in os.listdir('data/card_images') if f.endswith('.jpg')][:5]
    
    for test_img in test_images:
        print(f"\n Testing with: {test_img}")
        img_path = os.path.join('data/card_images', test_img)
        img = cv2.imread(img_path)
        
        matches = matcher.match_card(img, top_k=3)
        
        print(f"  Top 3 matches:")
        for idx, (card_id, score, info) in enumerate(matches, 1):
            print(f"    {idx}. {info['name']} (ID: {card_id}, Score: {score:.3f})")


if __name__ == "__main__":
    test_matcher()
