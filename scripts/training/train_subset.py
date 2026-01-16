"""
Train with subset of most common cards
Giải quyết vấn đề quá nhiều classes
"""

import os
import shutil
from collections import Counter
from tqdm import tqdm

# Count images per card
print("Counting images per card...")
data_dir = 'data/augmented_realistic'
image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

card_counts = Counter()
for img_file in image_files:
    card_id = img_file.split('_')[0]
    card_counts[card_id] += 1

print(f"   Total unique cards: {len(card_counts)}")
print(f"   Total images: {len(image_files)}")

# Get top N cards
TOP_N = 1000  # Train với 1000 thẻ phổ biến nhất
top_cards = [card_id for card_id, count in card_counts.most_common(TOP_N)]

print(f"\nTraining with top {TOP_N} cards")
print(f"   Images per card: ~{card_counts.most_common(1)[0][1]}")

# Create subset directory
subset_dir = 'data/augmented_subset'
os.makedirs(subset_dir, exist_ok=True)

# Copy images
print(f"\nCreating subset dataset...")
copied = 0
for img_file in tqdm(image_files, desc="Copying images"):
    card_id = img_file.split('_')[0]
    if card_id in top_cards:
        src = os.path.join(data_dir, img_file)
        dst = os.path.join(subset_dir, img_file)
        shutil.copy2(src, dst)
        copied += 1

print(f"\nSubset created!")
print(f"   Cards: {TOP_N}")
print(f"   Images: {copied}")
print(f"   Output: {subset_dir}")

# Now train with subset
print(f"\nStarting training with subset...")

from train_model_improved import ImprovedCardRecognizer

trainer = ImprovedCardRecognizer(
    data_dir='data/augmented_subset',
    model_save_path='models/card_recognition_subset.pth'
)

trainer.train(
    epochs=30,
    learning_rate=0.001,
    batch_size=32
)

print("\nTraining complete!")
