"""
Realistic Data Augmentation for Yu-Gi-Oh! Card Recognition
Simulates real-world camera conditions: low light, motion blur, angles, etc.
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import random


class RealisticCardAugmentor:
    """Augment card images with realistic camera conditions"""
    
    def __init__(self, images_dir='data/card_artworks', output_dir='data/augmented_realistic'):
        """
        Initialize realistic augmentor
        
        Args:
            images_dir: Directory containing cropped artwork images
            output_dir: Directory to save augmented images
        """
        self.images_dir = images_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Realistic Augmentor Initialized")
        print(f"   Input: {self.images_dir}")
        print(f"   Output: {self.output_dir}")
    
    def augment_single_image(self, image_path, card_id, num_augmentations=15):
        """
        Generate realistic augmented versions of a single card artwork
        
        Args:
            image_path: Path to artwork image
            card_id: Card ID for naming
            num_augmentations: Number of augmented versions to create
            
        Returns:
            int: Number of augmentations created
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return 0
            
            count = 0
            
            # Save original
            output_path = os.path.join(self.output_dir, f"{card_id}_orig.jpg")
            cv2.imwrite(output_path, img)
            count += 1
            
            # Generate augmentations
            for i in range(num_augmentations):
                augmented = img.copy()
                
                # Apply random combination of augmentations
                # Simulate real-world conditions
                
                # 1. Lighting variations (very important!)
                if random.random() < 0.7:  # 70% chance
                    augmented = self.adjust_lighting(augmented)
                
                # 2. Rotation (slight angles)
                if random.random() < 0.5:  # 50% chance
                    augmented = self.rotate_image(augmented, angle=random.uniform(-15, 15))
                
                # 3. Perspective transform (viewing angle)
                if random.random() < 0.4:  # 40% chance
                    augmented = self.perspective_transform(augmented)
                
                # 4. Motion blur (camera shake)
                if random.random() < 0.3:  # 30% chance
                    augmented = self.motion_blur(augmented)
                
                # 5. Gaussian blur (out of focus)
                if random.random() < 0.2:  # 20% chance
                    augmented = self.gaussian_blur(augmented)
                
                # 6. Noise (low quality camera)
                if random.random() < 0.3:  # 30% chance
                    augmented = self.add_noise(augmented)
                
                # 7. Contrast adjustment
                if random.random() < 0.4:  # 40% chance
                    augmented = self.adjust_contrast(augmented, factor=random.uniform(0.7, 1.3))
                
                # 8. Shadow/reflection simulation
                if random.random() < 0.2:  # 20% chance
                    augmented = self.add_shadow(augmented)
                
                # 9. Color temperature shift
                if random.random() < 0.3:  # 30% chance
                    augmented = self.color_temperature_shift(augmented)
                
                # Save augmented image
                output_path = os.path.join(self.output_dir, f"{card_id}_aug{i+1}.jpg")
                cv2.imwrite(output_path, augmented)
                count += 1
            
            return count
            
        except Exception as e:
            print(f"Error augmenting {image_path}: {e}")
            return 0
    
    def adjust_lighting(self, image):
        """
        Adjust brightness to simulate different lighting conditions
        Critical for real-world camera scenarios
        """
        # Random brightness factor (0.4 = very dark, 1.6 = very bright)
        factor = random.uniform(0.4, 1.6)
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust V channel (brightness)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result
    
    def rotate_image(self, image, angle):
        """Rotate image by angle (simulates tilted camera)"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def perspective_transform(self, image):
        """Apply random perspective transform (viewing angle)"""
        h, w = image.shape[:2]
        
        # Define source points (original corners)
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Define destination points (with random distortion)
        offset = int(w * 0.1)  # 10% offset
        dst_pts = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), h - random.randint(0, offset)],
            [random.randint(0, offset), h - random.randint(0, offset)]
        ])
        
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply transform
        warped = cv2.warpPerspective(image, M, (w, h), 
                                     borderMode=cv2.BORDER_REPLICATE)
        return warped
    
    def motion_blur(self, image):
        """Add motion blur (camera shake)"""
        # Random kernel size
        size = random.choice([5, 7, 9])
        
        # Create motion blur kernel
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        
        # Apply blur
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred
    
    def gaussian_blur(self, image):
        """Add Gaussian blur (out of focus)"""
        kernel_size = random.choice([3, 5])
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred
    
    def add_noise(self, image):
        """Add random noise (low quality camera)"""
        # Gaussian noise
        noise = np.random.normal(0, random.uniform(5, 15), image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    def adjust_contrast(self, image, factor):
        """Adjust image contrast"""
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Adjust contrast
        mean = np.mean(img_float)
        contrasted = (img_float - mean) * factor + mean
        contrasted = np.clip(contrasted, 0, 255).astype(np.uint8)
        
        return contrasted
    
    def add_shadow(self, image):
        """Add shadow effect (partial lighting)"""
        h, w = image.shape[:2]
        
        # Create gradient mask
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Random shadow direction
        if random.random() < 0.5:
            # Vertical gradient
            for i in range(h):
                mask[i, :] = i / h
        else:
            # Horizontal gradient
            for j in range(w):
                mask[:, j] = j / w
        
        # Random shadow intensity
        intensity = random.uniform(0.3, 0.7)
        mask = mask * intensity + (1 - intensity)
        
        # Apply shadow
        shadowed = image.astype(np.float32)
        for c in range(3):
            shadowed[:, :, c] = shadowed[:, :, c] * mask
        
        shadowed = np.clip(shadowed, 0, 255).astype(np.uint8)
        return shadowed
    
    def color_temperature_shift(self, image):
        """Shift color temperature (warm/cool lighting)"""
        # Random shift
        shift = random.uniform(-30, 30)
        
        shifted = image.astype(np.float32)
        
        if shift > 0:
            # Warm (increase red, decrease blue)
            shifted[:, :, 2] += shift  # Red
            shifted[:, :, 0] -= shift  # Blue
        else:
            # Cool (decrease red, increase blue)
            shifted[:, :, 2] += shift  # Red
            shifted[:, :, 0] -= shift  # Blue
        
        shifted = np.clip(shifted, 0, 255).astype(np.uint8)
        return shifted
    
    def augment_dataset(self, num_augmentations=15, max_cards=None, num_threads=4):
        """
        Augment entire dataset with realistic conditions
        
        Args:
            num_augmentations: Number of augmentations per card
            max_cards: Maximum number of cards to process (None = all)
            num_threads: Number of threads for parallel processing
        """
        # Get all artwork images
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        
        if max_cards:
            image_files = image_files[:max_cards]
        
        print(f"\nAugmenting {len(image_files)} cards...")
        print(f"   {num_augmentations} augmentations per card")
        print(f"   Using {num_threads} threads")
        print(f"   Total images: {len(image_files) * (num_augmentations + 1)}")
        
        # Process with progress bar
        total_count = 0
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for img_file in image_files:
                image_path = os.path.join(self.images_dir, img_file)
                card_id = img_file.replace('.jpg', '')
                
                future = executor.submit(
                    self.augment_single_image, 
                    image_path, 
                    card_id, 
                    num_augmentations
                )
                futures.append(future)
            
            # Wait for completion with progress bar
            for future in tqdm(futures, desc="Augmenting cards"):
                total_count += future.result()
        
        print(f"\nAugmentation Complete!")
        print(f"   Total images created: {total_count}")
        print(f"   Output: {self.output_dir}")
        
        return total_count


def main():
    """Main function to augment dataset"""
    
    print("=" * 60)
    print("Realistic Card Augmentation")
    print("=" * 60)
    
    # Initialize augmentor
    augmentor = RealisticCardAugmentor(
        images_dir='data/card_artworks',
        output_dir='data/augmented_realistic'
    )
    
    # Augment dataset
    # For testing: max_cards=50, num_augmentations=10
    # For full training: max_cards=None, num_augmentations=15
    augmentor.augment_dataset(
        num_augmentations=15,
        max_cards=None,
        num_threads=8
    )
    
    print("\nRealistic augmentation complete!")
    print("   Next step: Run train_model_improved.py")


if __name__ == "__main__":
    main()
