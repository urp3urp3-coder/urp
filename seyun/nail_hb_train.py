import pandas as pd
import torch
import timm
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class HbRegressionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        all_metadata = pd.read_csv(csv_file)
        
        valid_indices = []
        image_paths = []
        print(f"Verifying dataset for split: '{os.path.basename(img_dir)}'... This might take a moment.")
        for idx, row in all_metadata.iterrows():
            img_name = row['ID']
            img_path = self._find_img_path(self.img_dir, str(img_name))
            if img_path:
                valid_indices.append(idx)
                image_paths.append(img_path)

        self.metadata = all_metadata.loc[valid_indices].reset_index(drop=True)
        self.metadata['image_path'] = image_paths
        
        print(f"Dataset for '{os.path.basename(img_dir)}' ready. Found {len(self.metadata)} valid image entries.")

    def __len__(self):
        return len(self.metadata)

    def _find_img_path(self, root, img_name):
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                name_part, _ = os.path.splitext(filename)
                if name_part == img_name:
                    return os.path.join(dirpath, filename)
                if filename == img_name:
                    return os.path.join(dirpath, filename)
        return None

    def __getitem__(self, idx):
        img_path = self.metadata.loc[idx, 'image_path']
        hb_value = self.metadata.loc[idx, 'Hemoglobina']
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        hb_value = torch.tensor(hb_value, dtype=torch.float32)

        return image, hb_value

def evaluate(model, dataloader, criterion_mse, criterion_mae, device, accuracy_tolerance=1.0):
    model.eval()
    total_mse_loss = 0.0
    total_mae = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            
            mse_loss = criterion_mse(outputs, labels)
            mae = criterion_mae(outputs, labels)
            
            total_mse_loss += mse_loss.item() * images.size(0)
            total_mae += mae.item() * images.size(0)
            
            # For regression, we define 'accuracy' as predictions within a certain tolerance.
            correct_predictions += (torch.abs(outputs - labels) < accuracy_tolerance).sum().item()
            total_samples += images.size(0)

    avg_mse_loss = total_mse_loss / total_samples
    avg_mae = total_mae / total_samples
    accuracy = (correct_predictions / total_samples) * 100
    return avg_mse_loss, avg_mae, accuracy

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Setup
    writer = SummaryWriter('runs/hb_predictor_experiment')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Paths and Transformations
    ROOT_DATA_DIR = 'seyun/Diff-Mix/una-001-output/'
    CSV_FILE_PATH = os.path.join(ROOT_DATA_DIR, 'metadata.csv')
    data_config = timm.data.resolve_data_config({}, model='swin_tiny_patch4_window7_224')
    transform = timm.data.create_transform(**data_config)

    # 3. Data Loading
    train_dataset = HbRegressionDataset(csv_file=CSV_FILE_PATH, img_dir=os.path.join(ROOT_DATA_DIR, 'train'), transform=transform)
    val_dataset = HbRegressionDataset(csv_file=CSV_FILE_PATH, img_dir=os.path.join(ROOT_DATA_DIR, 'validation'), transform=transform)
    test_dataset = HbRegressionDataset(csv_file=CSV_FILE_PATH, img_dir=os.path.join(ROOT_DATA_DIR, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 4. Model, Loss, Optimizer
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=1)
    model.to(device)

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    num_epochs = 10

    # 5. Training & Validation Loop
    best_val_loss = float('inf')
    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion_mse(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/train_step', loss.item(), step)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Validation
        val_loss, val_mae, val_accuracy = evaluate(model, val_loader, criterion_mse, criterion_mae, device)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('MAE/validation', val_mae, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

        print(f'\nEpoch [{epoch+1}/{num_epochs}] Validation - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Accuracy: {val_accuracy:.2f}%\n')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_hb_predictor.pth')
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

    writer.close()
    print('--- Finished Training ---\n')

    # 6. Final Testing
    print("--- Loading best model for final testing ---")
    model.load_state_dict(torch.load('best_hb_predictor.pth'))
    test_loss, test_mae, test_accuracy = evaluate(model, test_loader, criterion_mse, criterion_mae, device)
    print(f"Final Test Results -> Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, Accuracy (tolerance 1.0): {test_accuracy:.2f}%")
