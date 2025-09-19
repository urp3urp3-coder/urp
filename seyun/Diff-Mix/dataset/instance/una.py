import os
import pandas as pd
from PIL import Image
from ..base import HugFewShotDataset
import numpy as np
from sklearn.model_selection import train_test_split

# --- Helper function to find image paths robustly ---
def find_full_path(row, data_root, label2class):
    possible_splits = ['train', 'validation', 'test']
    class_folder = label2class.get(row['label'])
    if class_folder is None:
        return None
    
    # Use 'ID' column and append '.jpg' as discovered
    file_id = row.get('ID')
    if not isinstance(file_id, str):
        return None
    file_name = f"{file_id}.jpg"

    for split_folder in possible_splits:
        path_to_check = os.path.join(data_root, split_folder, class_folder, file_name)
        if os.path.exists(path_to_check):
            return path_to_check
    return None


class UnaDataset(HugFewShotDataset):
    def __init__(self, split="train", *args, **kwargs):
        # FIX 1: Corrected data path
        self.data_root = "C:/Users/parkj/OneDrive - 성균관대학교/바탕 화면/urp/augmentation/urp/seyun/Diff-Mix/una-001-output"
        super().__init__(split=split, *args, **kwargs)
        
        self.split = split
        metadata_path = os.path.join(self.data_root, "metadata.csv")
        self.meta_df = pd.read_csv(metadata_path)

        self.meta_df = self.meta_df.dropna(subset=['UNAS'])
        self.meta_df = self.meta_df[self.meta_df['UNAS'].str.endswith('.jpg')]

        # FIX 2: Changed 'Moderada' to 'Moderado'
        self.class_names = ['Normal', 'Leve', 'Moderado']
        self.num_classes = len(self.class_names)
        self.class2label = {name: i for i, name in enumerate(self.class_names)}
        self.label2class = {i: name for i, name in enumerate(self.class_names)}

        conditions = [
            self.meta_df['Normal'] == 1,
            self.meta_df['Leve'] == 1,
            self.meta_df['Moderado'] == 1 # FIX 2
        ]
        choices = [self.class2label['Normal'], self.class2label['Leve'], self.class2label['Moderado']] # FIX 2
        self.meta_df['label'] = np.select(conditions, choices, default=-1)
        self.meta_df = self.meta_df[self.meta_df['label'] != -1]

        # FIX 3: Verify file existence and skip if not found
        print("Verifying dataset file paths...")
        self.meta_df['full_path'] = self.meta_df.apply(
            lambda row: find_full_path(row, self.data_root, self.label2class), 
            axis=1
        )
        
        original_len = len(self.meta_df)
        self.meta_df = self.meta_df.dropna(subset=['full_path'])
        new_len = len(self.meta_df)
        if original_len > new_len:
            print(f"Removed {original_len - new_len} missing file entries.")
        print(f"Found {new_len} valid file entries.")
        # End of FIX 3

        train_indices, val_indices = train_test_split(
            self.meta_df.index,
            test_size=0.2,
            random_state=42,
            stratify=self.meta_df['label']
        )

        if split == 'train':
            self.meta_df = self.meta_df.loc[train_indices]
        else:
            self.meta_df = self.meta_df.loc[val_indices]
        
        self.meta_df = self.meta_df.reset_index(drop=True)
        
        self.label_to_indices = {
            label: np.where(self.meta_df["label"] == label)[0]
            for label in range(self.num_classes)
        }

    def __len__(self):
        return len(self.meta_df)

    def get_image_by_idx(self, idx: int) -> Image.Image:
        # FIX 3: Use the pre-verified full path
        image_path = self.meta_df.iloc[idx]['full_path']
        return Image.open(image_path).convert("RGB")

    def get_label_by_idx(self, idx: int) -> int:
        return self.meta_df.iloc[idx]['label']

    def get_metadata_by_idx(self, idx: int) -> dict:
        label_name = self.label2class[self.get_label_by_idx(idx)]
        return {'name': label_name}
