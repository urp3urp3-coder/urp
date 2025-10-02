
import pandas as pd
import os
import argparse
from tqdm import tqdm

def find_all_images(root_dir):
    """Finds all image files in the given root directory and its subdirectories."""
    image_paths = {}
    print(f"Searching for all images under '{root_dir}'...")
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Extract the ID from the filename (e.g., "UID_H01_01_1_1_1.jpg" -> "UID_H01_01_1_1_1")
                image_id = os.path.splitext(filename)[0]
                # Store the absolute path
                image_paths[image_id] = os.path.abspath(os.path.join(dirpath, filename))
    print(f"Found {len(image_paths)} images.")
    return image_paths

def main(args):
    """
    Prepares the dataset for training the conditional diffusion model.
    It combines image paths with their hemoglobin values from the metadata file.
    """
    print("--- Starting Data Preparation ---")
    
    # 1. Load metadata
    metadata_path = os.path.join(args.data_root, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found at '{metadata_path}'")
        return
        
    print(f"Loading metadata from '{metadata_path}'")
    metadata_df = pd.read_csv(metadata_path)
    
    # The ID in metadata might not have extensions, so ensure we handle that.
    # The find_all_images function already returns IDs without extension.
    metadata_df['ID'] = metadata_df['ID'].astype(str)

    # 2. Find all image paths
    image_paths_map = find_all_images(args.data_root)

    # 3. Map image paths to metadata
    print("Mapping image paths to metadata entries...")
    
    # Create a new column for the absolute path
    # We use .map() which is faster than iterating
    metadata_df['image_path'] = metadata_df['ID'].map(image_paths_map)

    # 4. Filter out entries where no image was found and report
    original_count = len(metadata_df)
    prepared_df = metadata_df.dropna(subset=['image_path', 'Hemoglobina']).copy()
    new_count = len(prepared_df)
    
    if new_count < original_count:
        print(f"Warning: Dropped {original_count - new_count} entries from metadata because the corresponding image files were not found.")

    # 5. Select and rename columns for clarity
    prepared_df = prepared_df[['image_path', 'Hemoglobina']]
    prepared_df = prepared_df.rename(columns={'Hemoglobina': 'hemoglobin'})

    # 6. Save the prepared data to a new CSV file
    output_path = os.path.join(args.output_dir, 'diffusion_data.csv')
    prepared_df.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully prepared data for {new_count} images.")
    print(f"Output saved to: '{output_path}'")
    print("--- Data Preparation Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for conditional diffusion model training.')
    
    # Get the parent directory of the current script to set default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_root = os.path.abspath(os.path.join(script_dir, 'Diff-Mix', 'real-data'))
    default_output_dir = os.path.abspath(script_dir)

    parser.add_argument('--data-root', type=str, default=default_data_root, 
                        help='Root directory containing metadata.csv and image subfolders (train, validation, test).')
    parser.add_argument('--output-dir', type=str, default=default_output_dir, 
                        help='Directory to save the output CSV file.')
    
    args = parser.parse_args()
    main(args)
