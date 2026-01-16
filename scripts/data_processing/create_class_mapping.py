#!/usr/bin/env python3
"""
Create class mapping file from trained model
Maps class indices to card names
"""

import json
import os
import pandas as pd

def create_class_mapping():
    """Create class_to_idx.json from card_id_mapping.json and cards.csv"""
    
    # Load card ID mapping
    mapping_path = 'models/card_id_mapping.json'
    if not os.path.exists(mapping_path):
        print(f"Error: {mapping_path} not found!")
        print("   Please train the model first.")
        return
    
    with open(mapping_path, 'r') as f:
        id_mapping = json.load(f)
    
    # Load cards database
    cards_csv = 'data/cards.csv'
    if not os.path.exists(cards_csv):
        print(f"Error: {cards_csv} not found!")
        return
    
    cards_df = pd.read_csv(cards_csv)
    
    # Create card_id to name mapping
    card_id_to_name = {}
    for _, row in cards_df.iterrows():
        card_id_to_name[row['id']] = row['name']
    
    # Create class_to_idx mapping (card_name -> class_index)
    class_to_idx = {}
    idx_to_class = id_mapping['idx_to_card_id']
    
    for idx_str, card_id in idx_to_class.items():
        if card_id in card_id_to_name:
            card_name = card_id_to_name[card_id]
            class_to_idx[card_name] = int(idx_str)
    
    # Save to data directory (where CNN recognizer expects it)
    output_dir = 'data/augmented_images'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'class_to_idx.json')
    with open(output_path, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    print(f"✓ Created class mapping with {len(class_to_idx)} cards")
    print(f"✓ Saved to: {output_path}")
    
    # Show sample
    print("\nSample mappings:")
    for i, (card_name, idx) in enumerate(list(class_to_idx.items())[:5]):
        print(f"   {card_name} -> {idx}")

if __name__ == "__main__":
    create_class_mapping()
