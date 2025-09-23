import pandas as pd
import torch
import timm
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

class HbRegressionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        all_metadata = pd.read_csv(csv_file)
        
        # Pre-filter the metadata to include only entries with valid, findable images.
        # This avoids errors during training and is more efficient.
        valid_indices = []
        image_paths = []
        print("Verifying dataset integrity... This might take a moment.")
        for idx, row in all_metadata.iterrows():
            img_name = row['ID']
            # _find_img_path is a new robust method to find images
            img_path = self._find_img_path(self.img_dir, str(img_name))
            if img_path:
                valid_indices.append(idx)
                image_paths.append(img_path)
            # else:
            #     print(f"Warning: Image for ID '{img_name}' not found. Excluding from dataset.")

        self.metadata = all_metadata.loc[valid_indices].reset_index(drop=True)
        # Store the full, verified path to avoid searching again in __getitem__
        self.metadata['image_path'] = image_paths
        
        print(f"Dataset ready. Found {len(self.metadata)} valid image entries out of {len(all_metadata)}.")

    def __len__(self):
        return len(self.metadata)

    def _find_img_path(self, root, img_name):
        """
        Robustly finds an image file in a directory tree,
        matching against the name with and without extension.
        """
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                name_part, _ = os.path.splitext(filename)
                # Case 1: The ID from CSV matches the filename without extension (e.g., 'ID265' -> 'ID265.jpg')
                if name_part == img_name:
                    return os.path.join(dirpath, filename)
                # Case 2: The ID from CSV matches the full filename (e.g., 'ID265.jpg' -> 'ID265.jpg')
                if filename == img_name:
                    return os.path.join(dirpath, filename)
        return None

    def __getitem__(self, idx):
        # Use the pre-verified path for efficiency
        img_path = self.metadata.loc[idx, 'image_path']
        hb_value = self.metadata.loc[idx, 'Hemoglobina']
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        hb_value = torch.tensor(hb_value, dtype=torch.float32)

        return image, hb_value

# 최종 실험 때는 경로 수정해야 함. 
# csv_file: seyun\Diff-Mix\una-001-output\metadata.csv
# img_dir: seyun\Diff-Mix\una-001-output\train

# 전처리는 timm의 swin_tiny_patch4_window7_224 모델에 맞춰야 함!!!!!!!! 
data_config = timm.data.resolve_data_config({}, model='swin_tiny_patch4_window7_224')
transform = timm.data.create_transform(**data_config)

ROOT_DATA_DIR = 'seyun/Diff-Mix/una-001-output/'
CSV_FILE_PATH = os.path.join(ROOT_DATA_DIR, 'metadata.csv')
dataset = HbRegressionDataset(csv_file=CSV_FILE_PATH, img_dir=ROOT_DATA_DIR, transform=transform)
# NOTE: num_workers > 0 can cause issues on Windows. Set to 0 for debugging, especially on different hardware.
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# Swin Transformer
model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,
    num_classes=1 
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 일단 가장 기본적인 걸로 (실험하면서 바꿀 예정)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
num_epochs = 10

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        # Squeeze model output to match label shape (B, 1) -> (B)
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:    # 매 10 미니배치마다 출력
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')

    print(f'Epoch [{epoch+1}/{num_epochs}] finished, Average Loss: {running_loss/len(dataloader):.4f}')
print('Finished Training')

torch.save(model.state_dict(), 'hb_predictor_swin_transformer.pth')
