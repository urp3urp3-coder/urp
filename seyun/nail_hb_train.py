import pandas as pd
import torch
import timm
import os
import math
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
# Evaluation
# -----------------------
def evaluate(model, dataloader, criterion, device, accuracy_tolerance):
    model.eval()
    all_outputs = []
    all_labels = []
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    val_loss = running_loss / len(dataloader.dataset)
    mae = nn.L1Loss()(all_outputs, all_labels).item()
    correct_predictions = (torch.abs(all_outputs - all_labels) < accuracy_tolerance).sum().item()
    accuracy = (correct_predictions / len(all_labels)) * 100
    
    ss_tot = torch.sum((all_labels - torch.mean(all_labels)) ** 2)
    ss_res = torch.sum((all_labels - all_outputs) ** 2)
    r2_score = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

    return val_loss, mae, accuracy, r2_score

# -----------------------
# Main Execution
# -----------------------
def main(args):
    # -----------------------
    # Config
    # -----------------------
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    EPOCHS_STAGE1 = args.epochs_stage1
    EPOCHS_STAGE2 = args.epochs_stage2
    LR_HEAD = args.lr_head
    LR_BACKBONE = args.lr_backbone
    WEIGHT_DECAY = args.weight_decay
    ROOT_DATA_DIR = args.root_data_dir
    CSV_FILE_PATH = os.path.join(ROOT_DATA_DIR, 'metadata.csv')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ACCURACY_TOLERANCE = args.accuracy_tolerance
    NUM_WORKERS = args.num_workers
    
    run_name = f"run_LRH_{LR_HEAD}_LRB_{LR_BACKBONE}_WD_{WEIGHT_DECAY}_BS_{BATCH_SIZE}"
    writer = SummaryWriter(f'runs/{run_name}')
    print(f"Using device: {DEVICE}")
    print(f"Starting run: {run_name}")

    # -----------------------
    # Transforms
    # -----------------------
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Data
    train_dataset = HbRegressionDataset(csv_file=CSV_FILE_PATH, img_dir=os.path.join(ROOT_DATA_DIR, 'train'), transform=transform)
    val_dataset = HbRegressionDataset(csv_file=CSV_FILE_PATH, img_dir=os.path.join(ROOT_DATA_DIR, 'validation'), transform=transform)
    test_dataset = HbRegressionDataset(csv_file=CSV_FILE_PATH, img_dir=os.path.join(ROOT_DATA_DIR, 'test'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    print(f"Creating model: {args.model_name}")
    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=1,
    )
    model = model.to(DEVICE)

    criterion = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE=="cuda"))
    
    def get_param_groups(m, head_lr, backbone_lr):
        try:
            classifier_params = m.get_classifier().parameters()
        except AttributeError:
            # Fallback for models that don't have get_classifier()
            # Assumes the last nn.Linear module is the classifier
            classifier_name = [n for n, _ in m.named_modules() if isinstance(_, nn.Linear)][-1]
            classifier_params = m.get_submodule(classifier_name).parameters()

        classifier_param_ids = {id(p) for p in classifier_params}
        
        backbone_params = [p for p in m.parameters() if id(p) not in classifier_param_ids and p.requires_grad]
        head_params = [p for p in m.parameters() if id(p) in classifier_param_ids and p.requires_grad]
        
        print(f"Found {len(head_params)} parameters in head and {len(backbone_params)} in backbone.")
        
        return [
            {"params": head_params, "lr": head_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]

    total_epochs = EPOCHS_STAGE1 + EPOCHS_STAGE2
    iters_per_epoch = len(train_loader)
    best_val_loss = float('inf')
    
    model_save_path = f"{args.model_name.replace('/', '_')}_hb_best_{run_name}.pth"

    # --- Stage 1 ---
    print("\n--- Starting Stage 1: Training head only ---")
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the classifier head
    try:
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    except AttributeError:
        classifier_name = [n for n, _ in model.named_modules() if isinstance(_, nn.Linear)][-1]
        for param in model.get_submodule(classifier_name).parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(get_param_groups(model, LR_HEAD, LR_BACKBONE), weight_decay=WEIGHT_DECAY)
    
    total_steps = total_epochs * iters_per_epoch
    def lr_lambda(step):
        return 0.5 * (1 + math.cos(math.pi * step / total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(EPOCHS_STAGE1):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            step = epoch * iters_per_epoch + i
            writer.add_scalar('Loss/train_step', loss.item(), step)
            if len(optimizer.param_groups) > 1:
                writer.add_scalar('LR/head', optimizer.param_groups[0]['lr'], step)
                writer.add_scalar('LR/backbone', optimizer.param_groups[1]['lr'], step)

        val_loss, val_mae, val_accuracy, val_r2 = evaluate(model, val_loader, criterion, DEVICE, ACCURACY_TOLERANCE)
        print(f'\nEpoch [{epoch+1}/{total_epochs}] (Stage 1) Validation -> Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Acc: {val_accuracy:.2f}%, R2: {val_r2:.4f}\n')
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('MAE/validation', val_mae, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        writer.add_scalar('R2/validation', val_r2, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

    # --- Stage 2 ---
    print("\n--- Starting Stage 2: Unfreezing all layers ---")
    for p in model.parameters():
        p.requires_grad = True
    
    # Optimizer를 다시 만들어 모든 파라미터를 포함시킵니다.
    optimizer = torch.optim.AdamW(get_param_groups(model, LR_HEAD, LR_BACKBONE), weight_decay=WEIGHT_DECAY)
    
    # 스케줄러도 새로 만들어 현재 step에 맞게 상태를 조정합니다.
    current_step = EPOCHS_STAGE1 * iters_per_epoch
    def lr_lambda_stage2(step):
        actual_step = step + current_step
        return 0.5 * (1 + math.cos(math.pi * actual_step / total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_stage2)

    for epoch in range(EPOCHS_STAGE1, total_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            step = epoch * iters_per_epoch + i
            writer.add_scalar('Loss/train_step', loss.item(), step)
            if len(optimizer.param_groups) > 1:
                writer.add_scalar('LR/head', optimizer.param_groups[0]['lr'], step)
                writer.add_scalar('LR/backbone', optimizer.param_groups[1]['lr'], step)

        val_loss, val_mae, val_accuracy, val_r2 = evaluate(model, val_loader, criterion, DEVICE, ACCURACY_TOLERANCE)
        print(f'\nEpoch [{epoch+1}/{total_epochs}] (Stage 2) Validation -> Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Acc: {val_accuracy:.2f}%, R2: {val_r2:.4f}\n')
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('MAE/validation', val_mae, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        writer.add_scalar('R2/validation', val_r2, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

    writer.close()
    print('--- Finished Training ---\n')
    
    # --- Final Test ---
    print("--- Loading best model for final testing ---")
    # For security, it's recommended to use weights_only=True if you are loading a checkpoint from an untrusted source.
    # Here we trust the source as we just saved it.
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    test_loss, test_mae, test_accuracy, test_r2 = evaluate(model, test_loader, criterion, DEVICE, ACCURACY_TOLERANCE)
    print(f"Final Test Results -> Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, Accuracy (tolerance {ACCURACY_TOLERANCE}): {test_accuracy:.2f}%, R-squared: {test_r2:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a hemoglobin prediction model.')
    parser.add_argument('--model-name', type=str, default='swin_small_patch4_window7_224.ms_in22k', help='Name of the model to use from timm library')
    parser.add_argument('--img-size', type=int, default=384, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs-stage1', type=int, default=8, help='Number of epochs for stage 1 training')
    parser.add_argument('--epochs-stage2', type=int, default=25, help='Number of epochs for stage 2 training')
    parser.add_argument('--lr-head', type=float, default=3e-4, help='Learning rate for the model head')
    parser.add_argument('--lr-backbone', type=float, default=3e-5, help='Learning rate for the model backbone')
    parser.add_argument('--weight-decay', type=float, default=5e-2, help='Weight decay for the optimizer')
    parser.add_argument('--root-data-dir', type=str, default='una-001-output', help='Root directory of the dataset')
    # C:\Users\parkj\Desktop\urp\augmentation\urp\seyun\una-001-output
    parser.add_argument('--accuracy-tolerance', type=float, default=1.0, help='Accuracy tolerance for evaluation')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for DataLoader')
    
    args = parser.parse_args()
    main(args)
