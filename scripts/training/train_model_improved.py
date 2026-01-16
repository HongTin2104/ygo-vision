"""
Improved CNN Training for Yu-Gi-Oh! Card Recognition
Train on realistic augmented artwork images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class CardDataset(Dataset):
    """Dataset for card artwork images"""
    
    def __init__(self, data_dir, transform=None):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing augmented images
            transform: Image transformations
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all images and create class mapping
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Scan directory
        image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        
        # Extract card IDs (format: cardid_augX.jpg or cardid_orig.jpg)
        card_ids = set()
        for img_file in image_files:
            # Extract card ID (everything before first underscore)
            card_id = img_file.split('_')[0]
            card_ids.add(card_id)
        
        # Create class mapping
        card_ids_sorted = sorted(card_ids)
        for idx, card_id in enumerate(card_ids_sorted):
            self.class_to_idx[card_id] = idx
            self.idx_to_class[idx] = card_id
        
        # Create image-label pairs
        for img_file in image_files:
            card_id = img_file.split('_')[0]
            self.images.append(img_file)
            self.labels.append(self.class_to_idx[card_id])
        
        print(f"Dataset loaded:")
        print(f"   Total images: {len(self.images)}")
        print(f"   Total classes: {len(self.class_to_idx)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


class ImprovedCardRecognizer:
    """Improved card recognizer with better training"""
    
    def __init__(self, data_dir='data/augmented_realistic', 
                 model_save_path='models/card_recognition_improved.pth'):
        """
        Initialize improved recognizer
        
        Args:
            data_dir: Directory containing augmented images
            model_save_path: Path to save trained model
        """
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Improved Card Recognizer")
        print(f"   Device: {self.device}")
        print(f"   Data: {data_dir}")
        
        # Create models directory
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    def prepare_data(self, train_split=0.8, batch_size=32):
        """
        Prepare training and validation data
        
        Args:
            train_split: Fraction of data for training
            batch_size: Batch size for training
            
        Returns:
            train_loader, val_loader, num_classes
        """
        # Define transforms
        # Training: with additional augmentation
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),  # Slight chance
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Validation: no augmentation
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load full dataset
        full_dataset = CardDataset(self.data_dir, transform=None)
        num_classes = len(full_dataset.class_to_idx)
        
        # Save class mapping
        class_mapping_path = os.path.join(self.data_dir, 'class_to_idx.json')
        with open(class_mapping_path, 'w') as f:
            json.dump(full_dataset.class_to_idx, f, indent=2)
        print(f"   ✓ Saved class mapping to {class_mapping_path}")
        
        # Split dataset
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Apply transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        print(f"\nData Split:")
        print(f"   Training: {train_size} images")
        print(f"   Validation: {val_size} images")
        print(f"   Classes: {num_classes}")
        print(f"   Batch size: {batch_size}")
        
        return train_loader, val_loader, num_classes
    
    def create_model(self, num_classes):
        """
        Create EfficientNet model
        
        Args:
            num_classes: Number of card classes
            
        Returns:
            model
        """
        # Load pre-trained EfficientNet-B0
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Freeze early layers (transfer learning)
        for param in model.features[:5].parameters():
            param.requires_grad = False
        
        # Replace classifier
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        model = model.to(self.device)
        
        print(f"\nModel Architecture:")
        print(f"   Base: EfficientNet-B0 (pre-trained)")
        print(f"   Frozen layers: features[:5]")
        print(f"   Classifier: {num_features} → 512 → {num_classes}")
        
        return model
    
    def train(self, epochs=30, learning_rate=0.001, batch_size=32):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
        """
        # Prepare data
        train_loader, val_loader, num_classes = self.prepare_data(
            batch_size=batch_size
        )
        
        # Create model
        model = self.create_model(num_classes)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        print(f"\nTraining Started!")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Optimizer: Adam")
        print("=" * 60)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * train_correct / train_total:.2f}%'
                })
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for images, labels in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * val_correct / val_total:.2f}%'
                    })
            
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'num_classes': num_classes
                }, self.model_save_path)
                print(f"   Best model saved! (Val Acc: {val_acc:.2f}%)")
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            print("=" * 60)
        
        print(f"\nTraining Complete!")
        print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"   Model saved to: {self.model_save_path}")
        
        # Plot training history
        self.plot_history(history)
        
        return history
    
    def plot_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=150)
        print(f"   Training history saved to models/training_history.png")


def main():
    """Main training function"""
    
    print("=" * 60)
    print("Improved Card Recognition Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ImprovedCardRecognizer(
        data_dir='data/augmented_realistic',
        model_save_path='models/card_recognition_improved.pth'
    )
    
    # Train model
    # Adjust epochs and batch_size based on your GPU memory
    # For testing: epochs=5, batch_size=16
    # For production: epochs=30, batch_size=32
    trainer.train(
        epochs=30,
        learning_rate=0.001,
        batch_size=32
    )
    
    print("\nTraining complete!")
    print("   Next step: Update app.py to use the improved model")


if __name__ == "__main__":
    main()
