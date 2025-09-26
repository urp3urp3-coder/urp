import pandas as pd
import torch
import timm
import os
import math
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# -----------------------
# Config
# -----------------------
IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS_STAGE1 = 64
EPOCHS_STAGE2 = 500
LR_HEAD = 3e-4
LR_BACKBONE = 3e-5
WEIGHT_DECAY = 5e-2
# 실험용 - 72+ 18 = 90장 
# ROOT_DATA_DIR = 'Diff-Mix/una-001-output/'
# 실제 데이터 train 619장 
ROOT_DATA_DIR = 'Diff-Mix/real-data/'
CSV_FILE_PATH = os.path.join(ROOT_DATA_DIR, 'metadata.csv')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACCURACY_TOLERANCE = 1.0
NUM_WORKERS = 0 

# -----------------------
# Dataset
# -----------------------
class HbRegressionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        all_metadata = pd.read_csv(csv_file)
        valid_indices = []
        image_paths = []
        print(f"Verifying dataset for split: '{os.path.basename(img_dir)}'...")
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
                if name_part == img_name or filename == img_name:
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

# -----------------------
# Transforms
# -----------------------
# 데이터 증강(augmentation)은 Diff-Mix로 따로 할거임. 
# 기본적인 리사이즈, 텐서 변환, 정규화만 적용
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# -----------------------
# Evaluation
# -----------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_outputs = []
    all_labels = []
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    val_loss = running_loss / len(dataloader.dataset)
    mae = nn.L1Loss()(all_outputs, all_labels).item()
    correct_predictions = (torch.abs(all_outputs - all_labels) < ACCURACY_TOLERANCE).sum().item()
    accuracy = (correct_predictions / len(all_labels)) * 100
    
    ss_tot = torch.sum((all_labels - torch.mean(all_labels)) ** 2)
    ss_res = torch.sum((all_labels - all_outputs) ** 2)
    r2_score = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

    return val_loss, mae, accuracy, r2_score

# -----------------------
# Main Execution
# -----------------------
def main():
    writer = SummaryWriter('runs/hb_predictor_finetune_experiment')
    print(f"Using device: {DEVICE}")

    # Data
    train_dataset = HbRegressionDataset(csv_file=CSV_FILE_PATH, img_dir=os.path.join(ROOT_DATA_DIR, 'train'), transform=transform)
    val_dataset = HbRegressionDataset(csv_file=CSV_FILE_PATH, img_dir=os.path.join(ROOT_DATA_DIR, 'validation'), transform=transform)
    test_dataset = HbRegressionDataset(csv_file=CSV_FILE_PATH, img_dir=os.path.join(ROOT_DATA_DIR, 'test'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = timm.create_model(
        "swin_small_patch4_window7_224.ms_in22k",
        pretrained=True,
        num_classes=1,
        img_size = IMG_SIZE,
    )
    model = model.to(DEVICE)

    criterion = nn.SmoothL1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))
    
    def param_groups(m):
        head, back = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad: continue
            if "head" in n or "norm" in n or "stages.3" in n:
                head.append(p)
            else:
                back.append(p)
        return [{"params": head, "lr": LR_HEAD}, {"params": back, "lr": LR_BACKBONE}]

    total_epochs = EPOCHS_STAGE1 + EPOCHS_STAGE2
    iters_per_epoch = len(train_loader)
    best_val_loss = float('inf')

    # --- Stage 1 ---
    print("\n--- Starting Stage 1: Training head and deep layers ---")
    for name, p in model.named_parameters():
        if any(k in name for k in ["stages.0", "stages.1"]):
            p.requires_grad = False
    
    optimizer = torch.optim.AdamW(param_groups(model), weight_decay=WEIGHT_DECAY)
    
    total_steps = total_epochs * iters_per_epoch
    def lr_lambda(step):
        return 0.5 * (1 + math.cos(math.pi * step / total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(EPOCHS_STAGE1):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            step = epoch * iters_per_epoch + i
            writer.add_scalar('Loss/train_step', loss.item(), step)
            if len(optimizer.param_groups) > 1:
                writer.add_scalar('LR/head', optimizer.param_groups[0]['lr'], step)
                writer.add_scalar('LR/backbone', optimizer.param_groups[1]['lr'], step)

        val_loss, val_mae, val_accuracy, val_r2 = evaluate(model, val_loader, criterion, DEVICE)
        print(f'\nEpoch [{epoch+1}/{total_epochs}] (Stage 1) Validation -> Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Acc: {val_accuracy:.2f}%, R2: {val_r2:.4f}\n')
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('MAE/validation', val_mae, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        writer.add_scalar('R2/validation', val_r2, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "swinS_in22k_hb_best.pth")
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

    # --- Stage 2 ---
    print("\n--- Starting Stage 2: Unfreezing all layers ---")
    for p in model.parameters():
        p.requires_grad = True
    
    # Optimizer를 다시 만들어 모든 파라미터를 포함
    optimizer = torch.optim.AdamW(param_groups(model), weight_decay=WEIGHT_DECAY)
    
    # 스케줄러도 새로 만들어 현재 step에 맞게 상태를 조정
    current_step = EPOCHS_STAGE1 * iters_per_epoch
    def lr_lambda_stage2(step):
        actual_step = step + current_step
        return 0.5 * (1 + math.cos(math.pi * actual_step / total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_stage2)

    for epoch in range(EPOCHS_STAGE1, total_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            step = epoch * iters_per_epoch + i
            writer.add_scalar('Loss/train_step', loss.item(), step)
            if len(optimizer.param_groups) > 1:
                writer.add_scalar('LR/head', optimizer.param_groups[0]['lr'], step)
                writer.add_scalar('LR/backbone', optimizer.param_groups[1]['lr'], step)

        val_loss, val_mae, val_accuracy, val_r2 = evaluate(model, val_loader, criterion, DEVICE)
        print(f'\nEpoch [{epoch+1}/{total_epochs}] (Stage 2) Validation -> Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Acc: {val_accuracy:.2f}%, R2: {val_r2:.4f}\n')
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('MAE/validation', val_mae, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        writer.add_scalar('R2/validation', val_r2, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "swinS_in22k_hb_best.pth")
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

    writer.close()
    print('--- Finished Training ---\n')
    
    # --- Final Test ---
    print("--- Loading best model for final testing ---")
    model.load_state_dict(torch.load("swinS_in22k_hb_best.pth", weights_only=True))
    test_loss, test_mae, test_accuracy, test_r2 = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Final Test Results -> Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, Accuracy (tolerance {ACCURACY_TOLERANCE}): {test_accuracy:.2f}%, R-squared: {test_r2:.4f}")

if __name__ == '__main__':
    main()