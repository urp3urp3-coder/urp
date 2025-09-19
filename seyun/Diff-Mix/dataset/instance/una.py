import os
import pandas as pd
from PIL import Image
from ..base import HugFewShotDataset
import numpy as np

class UnaDataset(HugFewShotDataset):
    def __init__(self, split="train", *args, **kwargs):
        self.data_root = "una-001-output" 
        super().__init__(split=split, *args, **kwargs)
        
        self.split = split

        metadata_path = os.path.join(self.data_root, "metadata.csv")
        self.meta_df = pd.read_csv(metadata_path)

        self.meta_df = self.meta_df.dropna(subset=['UNAS'])
        self.meta_df = self.meta_df[self.meta_df['UNAS'].str.endswith('.jpg')]

        self.class_names = ['Normal', 'Leve', 'Moderado']
        self.num_classes = len(self.class_names)
        self.class2label = {name: i for i, name in enumerate(self.class_names)}
        self.label2class = {i: name for i, name in enumerate(self.class_names)}

        conditions = [
            self.meta_df['Normal'] == 1,
            self.meta_df['Leve'] == 1,
            self.meta_df['Moderado'] == 1
        ]
        choices = [self.class2label['Normal'], self.class2label['Leve'], self.class2label['Moderado']]
        self.meta_df['label'] = np.select(conditions, choices, default=-1)

        # 파일명 없는 거 제거
        print("[dataset file paths....]")
        def file_full_path(row):
            possible_splits = ['train', 'validation', 'test']
            class_folder = self.label2class.get(row['label'])
            if class_folder is None:
                return None
            
            for split_folder in possible_splits:
                path_to_check = os.path.join(self.data_root, split_folder, class_folder, row['UNAS'])
                if os.path.exists(path_to_check):
                    return path_to_check
            return None
        
        self.meta_df['full_path'] = self.meta_df.apply(file_full_path, axis=1)

        # dopping rows
        original_len = len(self.meta_df)
        self.meta_df = self.meta_df.dropna(subset=['full_path'])\
        new_len = len(self.meta_df)
        print(f'Dropped {original_len - new_len} rows without valid image paths.')


        from sklearn.model_selection import train_test_split
        
        self.meta_df = self.meta_df[self.meta_df['label'] != -1]

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
        # row = self.meta_df.iloc[idx]
        # label_idx = row['label']
        # class_folder = self.label2class[label_idx]

        # possible_splits = ['train', 'validation', 'test']
        # image_path = None
        # for split_folder in possible_splits:
        #     path_to_check = os.path.join(self.data_root, split_folder, class_folder, row['UNAS'])
        #     if os.path.exists(path_to_check):
        #         image_path = path_to_check
        #         break
        
        # if image_path is None:
        #     raise FileNotFoundError(f"Image {row['UNAS']} not found in any split/class folder.")
        image_path = self.meta_df.iloc[idx]['full_path']

        return Image.open(image_path).convert("RGB")

    def get_label_by_idx(self, idx: int) -> int:
        return self.meta_df.iloc[idx]['label']

    def get_metadata_by_idx(self, idx: int) -> dict:
        label_name = self.label2class[self.get_label_by_idx(idx)]
        return {'name': label_name}