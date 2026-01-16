"""
Add specific cards to training subset and retrain
"""

import pandas as pd
import shutil
import os
from tqdm import tqdm

# Load database
print("Loading card database...")
df = pd.read_csv('data/cards.csv')

# Cards to add
cards_to_add = [
    "Cyber Angel Benten",
    "Traptrix Sera",
    "Egyptian God Slime",
    "Skyscraper",
    "Magician's Rod",
    "Elemental HERO Honest Neos"
]

print(f"\nFinding {len(cards_to_add)} cards...")

# Find card IDs
card_ids = []
for card_name in cards_to_add:
    # Case-insensitive search
    result = df[df['name'].str.lower() == card_name.lower()]
    
    if not result.empty:
        card_id = str(result.iloc[0]['id'])
        card_ids.append(card_id)
        print(f"   ✓ Found: {card_name} (ID: {card_id})")
    else:
        print(f"   ✗ Not found: {card_name}")

print(f"\nFound {len(card_ids)} cards")

# Check if artworks exist
print(f"\nChecking artworks...")
existing_artworks = []
for card_id in card_ids:
    artwork_path = f'data/card_artworks/{card_id}.jpg'
    if os.path.exists(artwork_path):
        existing_artworks.append(card_id)
        print(f"   ✓ Artwork exists: {card_id}")
    else:
        print(f"   ✗ Artwork missing: {card_id}")

print(f"\n{len(existing_artworks)}/{len(card_ids)} artworks available")

if len(existing_artworks) == 0:
    print("\nNo artworks found! Please check card_artworks directory.")
    exit(1)

# Copy artworks to subset
print(f"\nCopying artworks to subset...")
subset_dir = 'data/card_artworks_subset'
os.makedirs(subset_dir, exist_ok=True)

# First, copy existing subset
print("   Copying existing subset...")
existing_subset = os.listdir('data/augmented_subset')
existing_card_ids = set()
for img_file in existing_subset:
    if img_file.endswith('.jpg'):
        card_id = img_file.split('_')[0]
        existing_card_ids.add(card_id)

print(f"   Existing cards in subset: {len(existing_card_ids)}")

# Copy existing artworks
for card_id in tqdm(existing_card_ids, desc="Copying existing"):
    src = f'data/card_artworks/{card_id}.jpg'
    dst = f'{subset_dir}/{card_id}.jpg'
    if os.path.exists(src):
        shutil.copy2(src, dst)

# Add new cards
print(f"\nAdding {len(existing_artworks)} new cards...")
for card_id in existing_artworks:
    src = f'data/card_artworks/{card_id}.jpg'
    dst = f'{subset_dir}/{card_id}.jpg'
    shutil.copy2(src, dst)
    print(f"   ✓ Added: {card_id}")

total_cards = len(os.listdir(subset_dir))
print(f"\nTotal cards in new subset: {total_cards}")
print(f"   Previous: {len(existing_card_ids)}")
print(f"   New: {len(existing_artworks)}")
print(f"   Total: {total_cards}")

# Now augment
print(f"\nStarting augmentation...")
from augment_data_realistic import RealisticCardAugmentor

augmentor = RealisticCardAugmentor(
    images_dir=subset_dir,
    output_dir='data/augmented_subset_new'
)

augmentor.augment_dataset(
    num_augmentations=15,
    max_cards=None,
    num_threads=8
)

print("\nAugmentation complete!")

# Train model
print(f"\nStarting training...")
from train_model_improved import ImprovedCardRecognizer

trainer = ImprovedCardRecognizer(
    data_dir='data/augmented_subset_new',
    model_save_path='models/card_recognition_subset_v2.pth'
)

trainer.train(
    epochs=30,
    learning_rate=0.001,
    batch_size=32
)

print("\nComplete!")
print(f"\nNext steps:")
print(f"   1. Update app.py to use new model:")
print(f"      model_path='models/card_recognition_subset_v2.pth'")
print(f"      data_dir='data/augmented_subset_new'")
print(f"   2. Restart app.py")
print(f"   3. Test with new cards!")
