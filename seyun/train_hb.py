# train_hemoglobin.py
import argparse, os, random, time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets, models

from sklearn.metrics import classification_report, confusion_matrix

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_dataloaders(data_dir, img_size=224, val_ratio=0.2, batch_size=32, num_workers=4):
    # ImageNet 통계 (사전학습 백본과 일치)
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225])

    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.RandomResizedCrop(img_size, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        normalize,
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    full = datasets.ImageFolder(data_dir, transform=train_tf)
    # 클래스별 인덱스 모아 stratified split 비슷하게 만들기
    indices_by_class = {c: [] for c in range(len(full.classes))}
    for idx, (_, y) in enumerate(full.samples):
        indices_by_class[y].append(idx)

    train_indices, val_indices = [], []
    for c, idxs in indices_by_class.items():
        n = len(idxs); nv = max(1, int(n*val_ratio))
        random.shuffle(idxs)
        val_indices += idxs[:nv]
        train_indices += idxs[nv:]

    train_ds = Subset(datasets.ImageFolder(data_dir, transform=train_tf), train_indices)
    val_ds   = Subset(datasets.ImageFolder(data_dir, transform=val_tf),   val_indices)

    # 클래스 가중치(불균형 대응)
    counts = np.zeros(len(full.classes), dtype=np.int64)
    for _, y in np.array(full.samples)[train_indices]:
        counts[y] += 1
    class_weights = counts.sum() / np.maximum(counts, 1)
    class_weights = class_weights / class_weights.sum() * len(full.classes)  # 스케일 정규화
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, full.classes, class_weights

def build_model(num_classes=3):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feat, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_classes)
    )
    return model

def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds, all_trues = [], []
    loss_meter = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            loss_meter += loss.item() * x.size(0)
            all_trues.append(y.cpu().numpy())
            all_preds.append(logits.softmax(1).argmax(1).cpu().numpy())
    all_trues = np.concatenate(all_trues)
    all_preds = np.concatenate(all_preds)
    avg_loss = loss_meter / len(loader.dataset)
    acc = (all_trues == all_preds).mean()
    print("\n[Validation] loss: %.4f  acc: %.4f" % (avg_loss, acc))
    print("\nClassification report:")
    print(classification_report(all_trues, all_preds, target_names=class_names, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(all_trues, all_preds))
    return avg_loss, acc

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    train_loader, val_loader, class_names, class_weights = build_dataloaders(
        args.data_dir, args.img_size, args.val_ratio, args.batch_size, args.num_workers
    )
    print("Classes:", class_names)
    print("Train batches:", len(train_loader), " Val batches:", len(val_loader))

    model = build_model(num_classes=len(class_names)).to(device)

    # 손실: 클래스 가중치 적용(불균형일 때 도움)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    best_acc, best_path = 0.0, Path(args.out_dir)/"best_model.pt"
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss, running_correct, nseen = 0.0, 0, 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.softmax(1).argmax(1) == y).sum().item()
            nseen += x.size(0)

        scheduler.step()
        train_loss = running_loss / nseen
        train_acc = running_correct / nseen

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"time {time.time()-t0:.1f}s")

        # Validation & checkpoint
        val_loss, val_acc = evaluate(model, val_loader, device, class_names)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "classes": class_names,
                        "img_size": args.img_size}, best_path)
            print(f"  ✅ Best updated: acc={best_acc:.4f}  -> saved to {best_path}")

    print("\nTraining done. Best val acc:", best_acc)
    print("Best model path:", best_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="dataset root with Leve/Moderada/Normal")
    p.add_argument("--out_dir",  type=str, default="./outputs")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)
