"""
Download card images from URLs in the dataset with multi-threading
"""
import pandas as pd
import requests
import os
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Thread-safe counters
class Counter:
    def __init__(self):
        self.value = 0
        self.lock = Lock()
    
    def increment(self):
        with self.lock:
            self.value += 1
    
    def get(self):
        with self.lock:
            return self.value

def download_single_image(row, images_dir, force=False):
    """Download a single card image"""
    card_id = row['id']
    image_url = row['image_url']
    
    if pd.isna(image_url):
        return 'failed', None
    
    image_path = os.path.join(images_dir, f"{card_id}.jpg")
    
    # Skip if already exists
    if os.path.exists(image_path) and not force:
        return 'skipped', None
    
    try:
        # Download image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Save image
        with open(image_path, 'wb') as f:
            f.write(response.content)
        
        return 'downloaded', None
        
    except Exception as e:
        return 'failed', str(e)

def download_card_images(max_cards=1000, force=False, num_threads=10):
    """
    Download card images from the dataset using multi-threading
    
    Args:
        max_cards: Maximum number of cards to download (default 1000 for testing)
        force: If True, re-download existing images
        num_threads: Number of parallel download threads (default 10)
    """
    # Load dataset
    print("Loading card database...")
    df = pd.read_csv('data/cards.csv')
    print(f"Total cards in database: {len(df)}")
    
    # Create images directory
    images_dir = 'data/card_images'
    os.makedirs(images_dir, exist_ok=True)
    
    # Limit number of cards if specified
    if max_cards and max_cards < len(df):
        df = df.head(max_cards)
        print(f"Downloading first {max_cards} cards with {num_threads} threads...")
    else:
        print(f"Downloading all {len(df)} cards with {num_threads} threads...")
    
    # Counters
    downloaded = Counter()
    skipped = Counter()
    failed = Counter()
    
    # Download images in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = {
            executor.submit(download_single_image, row, images_dir, force): idx 
            for idx, row in df.iterrows()
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(futures), desc="Downloading images") as pbar:
            for future in as_completed(futures):
                status, error = future.result()
                
                if status == 'downloaded':
                    downloaded.increment()
                elif status == 'skipped':
                    skipped.increment()
                elif status == 'failed':
                    failed.increment()
                
                pbar.update(1)
    
    # Final stats
    print(f"\nDownload complete!")
    print(f"  Downloaded: {downloaded.get()}")
    print(f"  Skipped (already exists): {skipped.get()}")
    print(f"  Failed: {failed.get()}")
    
    total_images = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    print(f"  Total images in folder: {total_images}")
    
    return images_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Yu-Gi-Oh! card images (multi-threaded)')
    parser.add_argument('--max', type=int, default=1000, 
                       help='Maximum number of cards to download (default: 1000, use 0 for all)')
    parser.add_argument('--force', action='store_true',
                       help='Re-download existing images')
    parser.add_argument('--threads', type=int, default=10,
                       help='Number of parallel download threads (default: 10)')
    
    args = parser.parse_args()
    
    max_cards = None if args.max == 0 else args.max
    download_card_images(max_cards=max_cards, force=args.force, num_threads=args.threads)
