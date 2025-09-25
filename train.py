import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from unet import UNet
from datasets import ImageFolderDataset, ConjAnemiaDataset  # ✅ stage1/2 모두 import
import segmentation_models_pytorch as smp

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


# -----------------------------
# ImageNet 정규화 (ResNet 인코더와 궁합)
# -----------------------------
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
    return (x - mean) / std

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__(); self.s = smooth
    def forward(self, logits, targets):
        p = torch.sigmoid(logits); t = targets
        inter = (p*t).sum()
        union = p.sum() + t.sum() - inter
        iou = (inter + self.s) / (union + self.s)
        return 1 - iou

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__(); self.a, self.b, self.s = alpha, beta, smooth
    def forward(self, logits, targets):
        p = torch.sigmoid(logits); t = targets
        TP = (p*t).sum(); FP = (p*(1-t)).sum(); FN = ((1-p)*t).sum()
        tv = (TP + self.s) / (TP + self.a*FP + self.b*FN + self.s)
        return 1 - tv

def estimate_pos_weight(dataloader, device):
    pos = neg = 0.0
    for imgs, masks in dataloader:
        m = masks.to(device)
        pos += m.sum().item()
        neg += (1 - m).sum().item()
    if pos <= 0: return torch.tensor(1.0, device=device)
    return torch.tensor(max(1.0, neg/(pos+1e-6)), device=device)

# -----------------------------
# Dice Loss 정의
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # 로짓 → 확률
        preds = preds.view(-1)
        targets = targets.view(-1)
        inter = (preds * targets).sum()
        dice = (2. * inter + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.a, self.b, self.g, self.s = alpha, beta, gamma, smooth
    def forward(self, logits, targets):
        p = torch.sigmoid(logits).view(-1)
        t = targets.view(-1)
        TP = (p*t).sum(); FP = (p*(1-t)).sum(); FN = ((1-p)*t).sum()
        tv = (TP + self.s) / (TP + self.a*FP + self.b*FN + self.s)
        return (1 - tv)**self.g

# -----------------------------
# 간단 후처리(노이즈 제거 + 최대 성분 유지)
# -----------------------------
def post_process(mask_bin: np.ndarray) -> np.ndarray:
    # mask_bin: [H,W] uint8 {0,1}
    k3 = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, k3)   # 점 노이즈 제거
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3)         # 작은 구멍 메움
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.argmax(areas)
        m = (labels == keep).astype(np.uint8)
    return m


# -----------------------------
# Stage1: ROI Segmentation 학습
# -----------------------------
def train_stage1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolderDataset(  # ✅ mask_suffix를 정확히 전달
        args.img_root, args.mask_root,
        img_size=args.img_size,
        mask_suffix=getattr(args, 'mask_suffix', '')
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = UNet(in_c=3, out_c=1).to(device)  # 흑백 출력 (로짓)

        # --- 손실 구성 ---
    if args.loss == "bce":
        posw = estimate_pos_weight(loader, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=posw)
    elif args.loss == "bce_dice":
        posw = estimate_pos_weight(loader, device)
        bce   = nn.BCEWithLogitsLoss(pos_weight=posw)
        dice  = DiceLoss()
        criterion = lambda pred, tgt: bce(pred, tgt) + 0.5*dice(pred, tgt)
    elif args.loss == "bce_jaccard":
        posw = estimate_pos_weight(loader, device)
        bce   = nn.BCEWithLogitsLoss(pos_weight=posw)
        jac   = JaccardLoss()
        criterion = lambda pred, tgt: bce(pred, tgt) + jac(pred, tgt)
    elif args.loss == "ft_bce":  # ⚡ FocalTversky + BCE(pos_weight)
        posw = estimate_pos_weight(loader, device)
        ft   = FocalTverskyLoss(alpha=getattr(args,'tversky_alpha',0.7),
                                beta=getattr(args,'tversky_beta',0.3),
                                gamma=0.75)
        bce  = nn.BCEWithLogitsLoss(pos_weight=posw)
        criterion = lambda pred, tgt: 0.6*ft(pred, tgt) + 0.4*bce(pred, tgt)
    else:  # tversky (default)
        criterion = TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta)

    '''
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # ✅ 선행연구 방향: BCE + Jaccard(IoU) 조합
    bce  = nn.BCEWithLogitsLoss()
    jacc = smp.losses.JaccardLoss(mode='binary', from_logits=True)
    def criterion(logits, masks):
        return 0.5 * bce(logits, masks) + 0.5 * jacc(logits, masks)'''

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)                 # 로짓
            loss = criterion(preds, masks)      # BCEWithLogitsLoss에 로짓 입력

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Stage1][Ep {epoch+1:03d}] Loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.out)
            print(f"Stage-1 best model saved: {args.out}")

        scheduler.step()
    


# -----------------------------
# Stage2: ROI Fine-tuning (BCE + Dice, 증강 포함)
# -----------------------------
def finetune_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ConjAnemiaDataset(  # ✅ stage2는 증강 dataset
        args.img_root, args.mask_root,
        mask_suffix=args.mask_suffix,
        img_size=args.img_size,
        augment=True
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = UNet(in_c=3, out_c=1).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    def criterion(preds, masks):
        return bce(preds, masks) + dice(preds, masks)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Stage2][Ep {epoch+1:03d}] Loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.out)
            print(f"Stage-2 best model saved: {args.out}")


# -----------------------------
# Stage2-AE: Autoencoder Fine-tuning (CP-AnemiC ROI 데이터셋)
# -----------------------------
class AutoencoderDataset(Dataset):
    def __init__(self, root, img_size=256):
        self.img_files = []
        for subdir, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.img_files.append(os.path.join(subdir, f))
        self.img_size = img_size
        print(f"[DEBUG] Found {len(self.img_files)} images under {root}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, img  # 입력 = 타겟


def finetune_stage2_autoencoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AutoencoderDataset(args.img_root, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = UNet(in_c=3, out_c=3).to(device)  # AE는 RGB 재구성

    # Stage1 checkpoint 로드 (out 레이어 제외)
    state_dict = torch.load(args.ckpt, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("out.")}
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    print("Loaded from ckpt. Missing keys:", missing, "Unexpected keys:", unexpected)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device)

            recons = model(imgs)
            loss = criterion(recons, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Stage2-AE][Ep {epoch+1:03d}] Loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.out)
            print(f"Stage-2 AE best model saved: {args.out}")


# -----------------------------
# Inference (IoU / Dice 계산 포함)
# -----------------------------
def run_inference(args):
    import os, cv2, torch, numpy as np
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋
    dataset = ImageFolderDataset(
        img_root=args.img_root,
        mask_root=args.mask_root,
        img_size=args.img_size,
        mask_suffix=args.mask_suffix,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 모델 로드
    model = UNet(in_c=3, out_c=1).to(device)
    try:
        state = torch.load(args.ckpt, map_location=device, weights_only=True)  # PyTorch 2.4+
    except TypeError:
        state = torch.load(args.ckpt, map_location=device)                    # 하위버전 호환
    model.load_state_dict(state)
    model.eval()

    # 출력 디렉토리
    os.makedirs(args.out_dir, exist_ok=True)
    overlay_dir = os.path.join(args.out_dir, "overlay")
    heatmap_dir = os.path.join(args.out_dir, "heatmap")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    # 안전한 ImageNet 정규화 (내부 정의)
    def imagenet_norm(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
        return (x - mean) / std

    # 동적 임계값
    def dynamic_thr(prob_map: np.ndarray) -> float:
        t = np.percentile(prob_map, 95.0)
        return float(np.clip(t, 0.35, 0.60))

    # 후처리 (노이즈 제거 + 최대 성분 유지 + 최소 면적)
    

    def post_process(mask_bin: np.ndarray, rgb_for_rule: np.ndarray) -> np.ndarray:
        """
        mask_bin: (H,W) uint8 {0,1} – 모델 이진 예측
        rgb_for_rule: (H,W,3) uint8 – 모델 입력(리사이즈된) RGB
        목표: 선→면 채움, 공막/상단 억제, 하단(palpebral) 성분 선택
        """
        m = mask_bin.astype(np.uint8)
        H, W = m.shape

        # 1) 상단 억제: palpebral 하단만 남기기 (0.40~0.50 사이 조절 가능)
        band_top = int(H * 0.40)
        m[:band_top, :] = 0

        # 2) 공막(흰자) 억제: 낮은 채도 & 높은 밝기 제거
        hsv = cv2.cvtColor(rgb_for_rule, cv2.COLOR_RGB2HSV)
        _, S, V = cv2.split(hsv)
        sclera_like = ((S < 40) & (V > 180)).astype(np.uint8)   # 필요시 S<50, V>170로 완화
        sclera_like = cv2.dilate(sclera_like, np.ones((3,3), np.uint8), iterations=1)
        m[sclera_like == 1] = 0

        # 3) 선분들을 '가로'로 붙여 면으로: closing → dilate
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))   # 가로 방향 강화
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=1)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)), iterations=1)

        # 4) 연결성분 분석: 아래쪽 + 큰 + 가로로 긴 성분만 남기기
        num, lbl, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        keep_mask = np.zeros_like(m)
        if num > 1:
            cands = []
            for lab in range(1, num):
                x, y, w, h, area = stats[lab]
                cy = cent[lab][1]
                if area < H * W * 0.001:      # 너무 작은 건 제거(0.1%)
                    continue
                if h > 0 and (w / h) < 1.5:   # 가로로 긴 띠 선호(필요시 2.0~2.5로 강화)
                    continue
                if cy < H * 0.45:             # 아래쪽 성분 선호(필요시 0.5로 강화)
                    continue
                score = area * (1.0 + 0.003 * (cy - H * 0.5))  # 아래쪽일수록 가산
                cands.append((score, lab))
            if cands:
                _, keep = max(cands)
                keep_mask = (lbl == keep).astype(np.uint8)
        else:
            keep_mask = m

        if keep_mask.sum() == 0:
            return keep_mask

        # 5) 컨투어 채우기(면화) – 들뜬 가장자리 매끈하게
        cnts, _ = cv2.findContours(keep_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(keep_mask)
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [c], -1, 1, thickness=cv2.FILLED)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8), iterations=1)

        # 6) 얇으면 한 번 더 보강
        out = cv2.dilate(out, np.ones((5,5), np.uint8), iterations=1)
        return out.astype(np.uint8)

    
    use_tta   = bool(getattr(args, "tta", False))
    use_dyn   = bool(getattr(args, "dyn_thr", False))
    fixed_thr = float(getattr(args, "thr", 0.45))

    iou_scores, dice_scores = [], []

    for i, (img, mask_gt) in enumerate(loader):
        img, mask_gt = img.to(device), mask_gt.to(device)

        with torch.no_grad():
            logits = model(img)

            # TTA: 좌우 flip 평균
            if use_tta:
                logits_f = model(torch.flip(img, dims=[-1]))
                logits = 0.5 * (logits + torch.flip(logits_f, dims=[-1]))

            probs = torch.sigmoid(logits)
            pmap  = probs[0, 0].cpu().numpy()
            pmap = cv2.GaussianBlur(pmap, (0, 0), sigmaX=1.0)  # 점‧실선 예측을 살짝 펴주기

        thr = dynamic_thr(pmap) if use_dyn else fixed_thr
        pred_bin = (pmap > thr).astype(np.uint8)
        img_vis = (img[0].cpu().permute(1,2,0).numpy() * 255.0).clip(0,255).astype(np.uint8)
        out_np = post_process(pred_bin, img_vis)

        # 저장 파일명: 원본 이미지 파일명 사용
        orig_path = dataset.img_files[i]
        stem = os.path.splitext(os.path.basename(orig_path))[0]

        # 1) 이진 마스크 저장
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_roi.png"), out_np * 255)

        # 2) 오버레이 저장 (모델 입력 크기 기준)
        img_vis = (img[0].cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        color = np.zeros_like(img_vis); color[..., 1] = 255  # 초록
        overlay = cv2.addWeighted(img_vis, 1.0, color, 0.4, 0)
        overlay[out_np == 0] = img_vis[out_np == 0]
        cv2.imwrite(os.path.join(overlay_dir, f"{stem}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 3) 확률 히트맵 저장 (pmap)
        heat = (pmap * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(heatmap_dir, f"{stem}_heat.png"), heat)

        # 4) IoU / Dice
        gt = mask_gt[0, 0].cpu().numpy().astype(np.uint8)
        inter = (out_np & gt).sum()
        union = ((out_np | gt) > 0).sum()
        iou = inter / union if union > 0 else 0.0
        dice = 2 * inter / (out_np.sum() + gt.sum() + 1e-6)
        iou_scores.append(iou)
        dice_scores.append(dice)

    if iou_scores:
        print(f"Mean IoU: {np.mean(iou_scores):.4f}")
        print(f"Mean Dice: {np.mean(dice_scores):.4f}")
    else:
        print("⚠️ No ground truth masks were found for evaluation.")