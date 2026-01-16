"""
Script to download Yu-Gi-Oh! card database with images from Kaggle
"""
import kagglehub
import os
import shutil

def download_dataset():
    """Download the Yu-Gi-Oh! card database with images"""
    print("Downloading Yu-Gi-Oh! card database with images from Kaggle...")
    
    # Download latest version
    path = kagglehub.dataset_download("ioexception/yugioh-cards")
    
    print(f"Path to dataset files: {path}")
    
    # Copy to local data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Copy all files from downloaded path to data directory
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(data_dir, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"Copied {item} to {data_dir}")
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"Copied directory {item} to {data_dir}")
    
    print(f"\nDataset copied to: {data_dir}")
    
    # List contents
    print("\nDataset contents:")
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            num_files = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
            print(f"  {item}/ ({num_files} files)")
        else:
            size = os.path.getsize(item_path) / (1024 * 1024)  # MB
            print(f"   {item} ({size:.2f} MB)")
    
    return data_dir

if __name__ == "__main__":
    download_dataset()
