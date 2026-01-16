"""
Crop Artwork from Yu-Gi-Oh! Cards
Extract only the artwork region from full card images for better recognition
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


class ArtworkCropper:
    """Crop artwork region from Yu-Gi-Oh! cards"""
    
    def __init__(self, input_dir='data/card_images', output_dir='data/card_artworks'):
        """
        Initialize artwork cropper
        
        Args:
            input_dir: Directory containing full card images
            output_dir: Directory to save cropped artworks
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Artwork Cropper Initialized")
        print(f"   Input: {self.input_dir}")
        print(f"   Output: {self.output_dir}")
    
    def crop_artwork(self, image_path, card_id):
        """
        Crop artwork region from a Yu-Gi-Oh! card
        
        Yu-Gi-Oh! card artwork is typically in the upper portion:
        - Starts around 10% from top
        - Ends around 55% from top
        - Horizontally centered with some margin
        
        Args:
            image_path: Path to full card image
            card_id: Card ID for naming
            
        Returns:
            bool: Success status
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read: {image_path}")
                return False
            
            h, w = img.shape[:2]
            
            # Define artwork region (approximate percentages for Yu-Gi-Oh! cards)
            # These values work for standard Yu-Gi-Oh! card layout
            top_margin = 0.18      # 18% from top (fine-tuned)
            bottom_margin = 0.70   # 70% from top (capture full bottom artwork)
            left_margin = 0.10     # 10% from left
            right_margin = 0.90    # 90% from left
            
            # Calculate crop coordinates
            y1 = int(h * top_margin)
            y2 = int(h * bottom_margin)
            x1 = int(w * left_margin)
            x2 = int(w * right_margin)
            
            # Crop artwork
            artwork = img[y1:y2, x1:x2]
            
            # Resize to standard size for consistency
            artwork_resized = cv2.resize(artwork, (224, 224))
            
            # Save cropped artwork
            output_path = os.path.join(self.output_dir, f"{card_id}.jpg")
            cv2.imwrite(output_path, artwork_resized)
            
            return True
            
        except Exception as e:
            print(f"Error cropping {image_path}: {e}")
            return False
    
    def crop_dataset(self, max_cards=None, num_threads=4):
        """
        Crop artworks from entire dataset
        
        Args:
            max_cards: Maximum number of cards to process (None = all)
            num_threads: Number of threads for parallel processing
        """
        # Get all card images
        image_files = [f for f in os.listdir(self.input_dir) if f.endswith('.jpg')]
        
        if max_cards:
            image_files = image_files[:max_cards]
        
        print(f"\nProcessing {len(image_files)} cards...")
        print(f"   Using {num_threads} threads")
        
        # Process with progress bar
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for img_file in image_files:
                image_path = os.path.join(self.input_dir, img_file)
                card_id = img_file.replace('.jpg', '')
                
                future = executor.submit(self.crop_artwork, image_path, card_id)
                futures.append(future)
            
            # Wait for completion with progress bar
            for future in tqdm(futures, desc="Cropping artworks"):
                if future.result():
                    success_count += 1
        
        print(f"\nCropping Complete!")
        print(f"   Success: {success_count}/{len(image_files)}")
        print(f"   Output: {self.output_dir}")
        
        return success_count


def main():
    """Main function to crop artworks"""
    
    print("=" * 60)
    print("Yu-Gi-Oh! Artwork Cropper")
    print("=" * 60)
    
    # Initialize cropper
    cropper = ArtworkCropper(
        input_dir='data/card_images',
        output_dir='data/card_artworks'
    )
    
    # Crop all cards (or specify max_cards for testing)
    # For testing: max_cards=100
    # For full dataset: max_cards=None
    cropper.crop_dataset(max_cards=None, num_threads=8)
    
    print("\nArtwork cropping complete!")
    print("   Next step: Run augment_data_realistic.py")


if __name__ == "__main__":
    main()
