# Yu-Gi-Oh! Card Recognition System

[![GitHub](https://img.shields.io/badge/GitHub-ygo--vision-blue?logo=github)](https://github.com/HongTin2104/ygo-vision)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.97%25-brightgreen)](README.md)

Real-time Yu-Gi-Oh! card recognition using deep learning and computer vision.

**Repository**: [github.com/HongTin2104/ygo-vision](https://github.com/HongTin2104/ygo-vision)

## Demo

![Yu-Gi-Oh! Vision Demo](demo/demo.gif)


## Features

- **Real-time Detection**: Detect cards from webcam feed
- **CNN Recognition**: 99.97% accuracy with EfficientNet-B0
- **Artwork Focus**: Trained on card artworks (not full cards)
- **Realistic Augmentation**: Handles low-light, angles, blur, etc.
- **1,006 Cards**: Recognizes top 1,000 popular cards + custom additions
- **GPU Accelerated**: Fast inference with CUDA support

## Current Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 100% |
| **Training Accuracy** | 99.97% |
| **Cards Recognized** | 1,006 |
| **Training Images** | 16,096 |
| **Inference Speed** | ~50ms (GPU) |
| **Model Size** | 17MB |

---

# Quick Start

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Webcam
- 20GB disk space

## Installation

```bash
# Clone repository
git clone git@github.com:HongTin2104/ygo-vision.git
cd ygo-vision

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Run Application

```bash
python app.py
```

Open browser: `http://localhost:5000`

---

# Training Pipeline - From Scratch to 99.97% Accuracy

This is the complete process to create a high-accuracy card recognition model.

## Overview

```
Raw Card Images (13K)
    ↓ Step 1: Download & Prepare
Card Database + Images
    ↓ Step 2: Crop Artworks
Cropped Artworks (13K)
    ↓ Step 3: Select Subset
Top 1,006 Cards
    ↓ Step 4: Realistic Augmentation
16,096 Training Images
    ↓ Step 5: Train CNN Model
Model (99.97% accuracy)
    ↓ Step 6: Deploy
Production App
```

---

## Step 1: Download Card Database & Images

### 1.1 Get Card Database

Download `cards.csv` containing 13,281 Yu-Gi-Oh! cards:
- Card ID, name, type, description
- ATK, DEF, level, race, attribute
- Image URLs

Place in `data/cards.csv`

### 1.2 Download Card Images

```bash
python scripts/utils/download_dataset.py
```

**Output**: `data/card_images/` (~13,000 full card images)

**Time**: ~2-3 hours (depends on network)

---

## Step 2: Crop Card Artworks

**Problem**: Full card images contain borders, text, stats → noise for model

**Solution**: Crop only the artwork region

### 2.1 Configure Crop Region

Edit `scripts/data_processing/crop_artwork.py`:

```python
# Yu-Gi-Oh! card artwork region
top_margin = 0.18      # 18% from top
bottom_margin = 0.70   # 70% from top
left_margin = 0.10     # 10% from left
right_margin = 0.90    # 90% from left
```

### 2.2 Run Cropping

```bash
python scripts/data_processing/crop_artwork.py
```

**Output**: `data/card_artworks/` (~13,000 cropped artworks, 224x224)

**Time**: ~5-10 minutes

**Result**: Clean artwork images without borders/text

---

## Step 3: Select Training Subset

**Problem**: 13,000 classes is too many (only ~1 image per class after augmentation)

**Solution**: Train with top 1,000 most popular cards

### 3.1 Why Subset?

| Approach | Classes | Images/Class | Accuracy |
|----------|---------|--------------|----------|
| Full (13K) | 13,269 | ~1.2 | **0%** |
| Subset (1K) | 1,000 | ~16 | **99.97%** |

### 3.2 Selection Criteria

The subset includes:
- Top 1,000 most popular/viewed cards
- Can be customized (see Step 7)

---

## Step 4: Realistic Data Augmentation

**Problem**: Clean dataset images ≠ real-world camera conditions

**Solution**: Simulate realistic conditions

### 4.1 Augmentation Types

| Augmentation | Probability | Purpose |
|--------------|-------------|---------|
| **Lighting** | 70% | Low-light, bright light |
| **Rotation** | 50% | Tilted camera (±15°) |
| **Perspective** | 40% | Viewing angle |
| **Motion Blur** | 30% | Camera shake |
| **Gaussian Blur** | 20% | Out of focus |
| **Noise** | 30% | Low quality camera |
| **Contrast** | 40% | Different displays |
| **Shadow** | 20% | Partial lighting |
| **Color Temp** | 30% | Warm/cool lighting |

### 4.2 Run Augmentation

```bash
python scripts/data_processing/augment_data_realistic.py
```

**Configuration**:
- Input: `data/card_artworks/` (subset of 1,000)
- Output: `data/augmented_realistic/`
- Augmentations per card: 15
- Total images: 1,000 × 16 = **16,000**

**Time**: ~20-30 minutes

**Result**: Realistic training data that handles real-world conditions

---

## Step 5: Train CNN Model

### 5.1 Model Architecture

**Base**: EfficientNet-B0 (pre-trained on ImageNet)

**Modifications**:
- Freeze early layers (transfer learning)
- Custom classifier: 1280 → 512 → 1,006 classes
- Dropout: 0.3

### 5.2 Training Configuration

```python
# Hyperparameters
epochs = 30
batch_size = 32
learning_rate = 0.001
optimizer = Adam
scheduler = ReduceLROnPlateau(patience=3)
```

### 5.3 Run Training

```bash
python scripts/training/train_model_improved.py
```

**Time**: ~2-3 hours (GPU), ~10-15 hours (CPU)

**Output**:
- `models/card_recognition_subset.pth` - Trained model
- `models/training_history.png` - Training curves

### 5.4 Expected Results

```
Epoch 1:  Train Acc: 40%  | Val Acc: 85%
Epoch 5:  Train Acc: 95%  | Val Acc: 98%
Epoch 10: Train Acc: 98%  | Val Acc: 99%
Epoch 30: Train Acc: 99.97% | Val Acc: 100%
```

**Key Metrics**:
- Validation accuracy should reach 95%+ by epoch 10
- Final accuracy: 99.97% - 100%
- No significant overfitting (train ≈ val)

---

## Step 6: Deploy Model

### 6.1 Update Application

Edit `app.py`:

```python
cnn_recognizer = CNNCardRecognizer(
    model_path='models/card_recognition_subset.pth',
    data_dir='data/augmented_realistic'
)
```

### 6.2 Test Application

```bash
python app.py
```

Open `http://localhost:5000` and test with real cards!

---

## Step 7: Add Custom Cards (Optional)

Want to add specific cards to the model?

### 7.1 Edit Card List

Edit `scripts/training/add_cards_and_retrain.py`:

```python
cards_to_add = [
    "Cyber Angel Benten",
    "Traptrix Sera",
    "Egyptian God Slime",
    # Add your cards here...
]
```

### 7.2 Run Retrain

```bash
python scripts/training/add_cards_and_retrain.py
```

**Process**:
1. Find card IDs in database
2. Copy artworks to subset
3. Augment new cards
4. Retrain model
5. Save as `card_recognition_subset_v2.pth`

**Time**: ~2-3 hours

**Result**: New model with additional cards

---

# Understanding the Pipeline

## Why This Approach Works

### 1. Artwork Cropping (Critical!)
- Full card: Model learns borders, text → confusion
- Artwork only: Model learns actual card art → accuracy

### 2. Realistic Augmentation (Critical!)
- Clean images: Model fails in real conditions
- Realistic augmentation: Model handles low-light, blur, angles

### 3. Subset Training (Critical!)
- 13K classes, 1 image/class: Model can't learn
- 1K classes, 16 images/class: Model learns well

### 4. Transfer Learning
- Pre-trained EfficientNet knows general features
- Fine-tune on card artworks
- Faster training, better accuracy

## Common Issues & Solutions

### Issue 1: 0% Accuracy During Training

**Cause**: Too many classes, not enough data per class

**Solution**: Use subset (1,000 cards max)

### Issue 2: Good Training Acc, Bad Real-World Performance

**Cause**: Dataset too clean, doesn't match real conditions

**Solution**: Use realistic augmentation (lighting, blur, angles)

### Issue 3: Model Detects Full Card Instead of Artwork

**Cause**: Trained on full card images

**Solution**: Crop artworks before training

### Issue 4: CUDA Out of Memory

**Solution**: Reduce batch_size
```python
trainer.train(batch_size=16)  # or 8
```

---

# Project Structure

```
ygo_vision/
├── app.py                          # Flask web server
├── card_detector.py                # Card detection (CV)
├── card_recognizer_cnn.py          # CNN recognizer
│
├── scripts/
│   ├── training/
│   │   ├── train_model_improved.py      # Main training
│   │   ├── train_subset.py              # Train subset
│   │   └── add_cards_and_retrain.py     # Add cards
│   │
│   ├── data_processing/
│   │   ├── crop_artwork.py              # Crop artworks
│   │   ├── augment_data_realistic.py    # Augmentation
│   │   └── create_class_mapping.py      # Class mapping
│   │
│   └── utils/
│       └── download_dataset.py          # Download images
│
├── models/
│   ├── card_recognition_subset_v2.pth   # Current model
│   └── training_history.png             # Training curves
│
├── data/
│   ├── cards.csv                        # Card database
│   ├── card_images/                     # Full cards (13K)
│   ├── card_artworks/                   # Cropped (13K)
│   └── augmented_subset_new/            # Training data (16K)
│
├── templates/
│   └── index.html                       # Web UI
│
└── static/
    ├── css/
    ├── js/
    └── images/
```

---

# Advanced Usage

## Adjust Crop Region

If artworks are cut off:

```python
# In scripts/data_processing/crop_artwork.py
top_margin = 0.15      # Increase to crop more from top
bottom_margin = 0.75   # Increase to include more bottom
```

## Increase Training Data

```python
# In augment_data_realistic.py
num_augmentations = 20  # More variations per card
```

## Train Longer

```python
# In train_model_improved.py
trainer.train(epochs=50)  # More epochs
```

## Use Different Model

```python
# In train_model_improved.py
model = models.efficientnet_b1(weights='IMAGENET1K_V1')  # Larger model
```

---

# Performance Benchmarks

## Training Time (GPU: RTX 3060)

| Step | Time |
|------|------|
| Download images | 2-3 hours |
| Crop artworks | 5-10 min |
| Augmentation | 20-30 min |
| Training (30 epochs) | 2-3 hours |
| **Total** | **~5-7 hours** |

## Inference Performance

| Device | Speed | Batch |
|--------|-------|-------|
| RTX 3060 | 50ms | 1 |
| RTX 3090 | 30ms | 1 |
| CPU (i7) | 200ms | 1 |

---

# Troubleshooting

## Model Not Loading

```python
# Check paths
print(os.path.exists('models/card_recognition_subset_v2.pth'))
print(os.path.exists('data/augmented_subset_new/class_to_idx.json'))
```

## Low Accuracy

1. Check training curves (`models/training_history.png`)
2. Ensure validation accuracy > 95%
3. Try more augmentation
4. Check crop region (artworks not cut off)

## Camera Not Working

```bash
# Test camera
ls /dev/video*

# Try different camera index
# In app.py: cv2.VideoCapture(1)  # Try 0, 1, 2...
```

---

# Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
flask>=2.3.0
flask-cors>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
pillow>=10.0.0
tqdm>=4.65.0
```

---

# Next Steps

1. Follow training pipeline to create model
2. Test with real cards
3. Add custom cards if needed
4. Deploy to production

---

# License

MIT License

# Credits

- YGOProDeck API for card database
- PyTorch & EfficientNet
- OpenCV for computer vision

---

**Built with care for Yu-Gi-Oh! players**

**Model Version**: v2  
**Last Updated**: 2026-01-16  
**Accuracy**: 99.97%
