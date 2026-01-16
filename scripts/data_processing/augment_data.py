"""
Data Augmentation for Yu-Gi-Oh! Card Dataset
Generate augmented images for training: rotation, flip, brightness, etc.
"""
import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import random

class CardAugmentor:
    """Augment card images for training"""
    
    def __init__(self, images_dir='data/card_images', output_dir='data/augmented_images'):
        self.images_dir = images_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def augment_single_image(self, image_path, card_id, num_augmentations=10):
        """Generate augmented versions of a single card image"""
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        augmented_paths = []
        
        # Save original
        original_path = os.path.join(self.output_dir, f"{card_id}_original.jpg")
        cv2.imwrite(original_path, img)
        augmented_paths.append(original_path)
        
        for i in range(num_augmentations):
            aug_img = img.copy()
            
            # Random rotation (-20 to +20 degrees)
            if random.random() > 0.3:
                angle = random.uniform(-20, 20)
                aug_img = self.rotate_image(aug_img, angle)
            
            # Random perspective transform (simulate viewing angle)
            if random.random() > 0.5:
                aug_img = self.perspective_transform(aug_img)
            
            # Random brightness adjustment
            if random.random() > 0.4:
                factor = random.uniform(0.7, 1.3)
                aug_img = self.adjust_brightness(aug_img, factor)
            
            # Random contrast
            if random.random() > 0.4:
                factor = random.uniform(0.8, 1.2)
                aug_img = self.adjust_contrast(aug_img, factor)
            
            # Random blur (simulate camera focus)
            if random.random() > 0.6:
                kernel_size = random.choice([3, 5])
                aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)
            
            # Random noise
            if random.random() > 0.7:
                aug_img = self.add_noise(aug_img)
            
            # Horizontal flip (lật ngang)
            if random.random() > 0.7:
                aug_img = cv2.flip(aug_img, 1)
            
            # Vertical flip (lật dọc)
            if random.random() > 0.8:
                aug_img = cv2.flip(aug_img, 0)
            
            # Rotate 180 degrees (đảo ngược trên dưới)
            if random.random() > 0.85:
                aug_img = cv2.flip(aug_img, -1)  # Flip both horizontal and vertical
            
            # Rotate 90 degrees (xoay ngang)
            if random.random() > 0.9:
                aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
            
            # Save augmented image
            aug_path = os.path.join(self.output_dir, f"{card_id}_aug_{i}.jpg")
            cv2.imwrite(aug_path, aug_img)
            augmented_paths.append(aug_path)
        
        return augmented_paths
    
    def rotate_image(self, image, angle):
        """Rotate image by angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        return rotated
    
    def perspective_transform(self, image):
        """Apply random perspective transform"""
        h, w = image.shape[:2]
        
        # Random perspective points
        offset = int(min(h, w) * 0.1)
        
        src_points = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        
        dst_points = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), h - random.randint(0, offset)],
            [random.randint(0, offset), h - random.randint(0, offset)]
        ])
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = cv2.warpPerspective(image, M, (w, h),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
        return transformed
    
    def adjust_brightness(self, image, factor):
        """Adjust image brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, image, factor):
        """Adjust image contrast"""
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def add_noise(self, image):
        """Add random noise to image"""
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        return noisy
    
    def augment_dataset(self, num_augmentations=10, max_cards=None, num_threads=4):
        """Augment entire dataset with multi-threading"""
        print(f"Augmenting card dataset...")
        print(f"   Augmentations per card: {num_augmentations}")
        print(f"   Threads: {num_threads}")
        
        # Get all card images
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        
        if max_cards:
            image_files = image_files[:max_cards]
        
        print(f"   Total cards to process: {len(image_files)}")
        print(f"   Expected output: {len(image_files) * (num_augmentations + 1)} images")
        
        # Process images with multi-threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        total_augmented = 0
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            futures = {}
            for filename in image_files:
                card_id = filename.replace('.jpg', '')
                image_path = os.path.join(self.images_dir, filename)
                future = executor.submit(
                    self.augment_single_image, 
                    image_path, 
                    card_id, 
                    num_augmentations
                )
                futures[future] = card_id
            
            # Process completed tasks with progress bar
            with tqdm(total=len(futures), desc="Augmenting") as pbar:
                for future in as_completed(futures):
                    try:
                        augmented = future.result()
                        total_augmented += len(augmented)
                    except Exception as e:
                        card_id = futures[future]
                        print(f"\n✗ Error processing {card_id}: {e}")
                    pbar.update(1)
        
        print(f"\nAugmentation complete!")
        print(f"   Total images generated: {total_augmented}")
        print(f"   Output directory: {self.output_dir}")
        
        return total_augmented


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment Yu-Gi-Oh! card images for training')
    parser.add_argument('--num-aug', type=int, default=10,
                       help='Number of augmentations per card (default: 10)')
    parser.add_argument('--max-cards', type=int, default=None,
                       help='Maximum number of cards to process (default: all)')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of parallel threads (default: 4)')
    
    args = parser.parse_args()
    
    augmentor = CardAugmentor()
    augmentor.augment_dataset(
        num_augmentations=args.num_aug,
        max_cards=args.max_cards,
        num_threads=args.threads
    )


if __name__ == "__main__":
    main()
