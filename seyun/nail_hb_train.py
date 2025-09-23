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
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.metadata.loc[idx, 'ID']
        hb_value = self.metadata.loc[idx, 'Hemoglobina']

        img_path = os.path.join(self.img_dir, img_name)
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        hb_value = torch.tensor(hb_value, dtype=torch.float32)

        return image, hb_value
    
    def find_img_path(self, root, img_name):
        for dirpath, _, filenames in os.walk(root):
            if img_name in filenames:
                return os.path.join(dirpath, img_name)
        return None

# 최종 실험 때는 경로 수정해야 함. 
# csv_file: seyun\Diff-Mix\una-001-output\metadata.csv
# img_dir: seyun\Diff-Mix\una-001-output\train

# 전처리는 timm의 swin_tiny_patch4_window7_224 모델에 맞춰야 함!!!!!!!! 
data_config = timm.data.resolve_data_config({}, model='swin_tiny_patch4_window7_224')
transform = timm.data.create_transform(**data_config)

ROOT_DATA_DIR = 'seyun/Diff-Mix/una-001-output/'
CSV_FILE_PATH = os.path.join(ROOT_DATA_DIR, 'metadata.csv')
dataset = HbRegressionDataset(csv_file=CSV_FILE_PATH, img_dir=ROOT_DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

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
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:    # 매 10 미니배치마다 출력
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')

    print(f'Epoch [{epoch+1}/{num_epochs}] finished, Average Loss: {running_loss/len(dataloader):.4f}')
print('Finished Training')

torch.save(model.state_dict(), 'hb_predictor_swin_transformer.pth')

