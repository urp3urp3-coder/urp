from __future__ import annotations
import argparse, random
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

from v3_roi_pipeline import (
    RoiConfig, letterbox_bgr, preprocess_for_mask,
    slic_conjunctiva_mask, unletterbox_mask
)

def binarize_mask(m):
    if m.ndim == 3: m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return (m > 127).astype(np.uint8)

def iou_score(pred01, gt01):
    pred = pred01.astype(bool); gt = gt01.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter / union) if union > 0 else 1.0

def predict_mask(bgr, cfg: RoiConfig):
    small, meta = letterbox_bgr(bgr, size=cfg.resize)
    small_p = preprocess_for_mask(small, cfg)
    m_small = slic_conjunctiva_mask(small_p, cfg)
    return unletterbox_mask(m_small, meta)

def find_mask_for(image_path: Path, masks_root: Path, mask_suffix: str, mask_exts=(".png",".jpg",".jpeg")) -> Path | None:
    stem = image_path.stem
    rel = image_path.relative_to(image_root)
    # 1) 같은 상대경로 + suffix
    cand = [masks_root / rel.with_suffix("").as_posix() + mask_suffix + ext for ext in mask_exts]
    for c in cand:
        p = Path(c)
        if p.exists(): return p
    # 2) 마스크 폴더에서 같은 stem(확장자만 다름)
    for ext in mask_exts:
        p = masks_root / rel.with_suffix(ext).name
        if p.exists(): return p
    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Tune ROI params by IoU on CP-AnemiC GT")
    ap.add_argument("--images", required=True, help="CP-AnemiC images root")
    ap.add_argument("--masks", required=True, help="CP-AnemiC GT masks root")
    ap.add_argument("--mask-suffix", default="_mask", help="suffix added to image stem to get GT mask name")
    ap.add_argument("--sample", type=int, default=200, help="number of images to sample for quick tuning")
    ap.add_argument("--trials", type=int, default=60, help="random trials (>=30 추천)")
    ap.add_argument("--gate", choices=["fixed","shemo"], default="shemo")
    args = ap.parse_args()

    image_root = Path(args.images); masks_root = Path(args.masks)
    imgs = sorted([p for p in image_root.rglob("*") if p.suffix.lower() in {".jpg",".jpeg"}])
    if args.sample and args.sample < len(imgs):
        random.seed(0); imgs = random.sample(imgs, args.sample)

    # 검색 공간(가볍게)
    seg_choices = [280, 320, 360, 420]
    comp_choices = [8.0, 12.0, 16.0]
    aperc_choices = [74.0, 80.0, 86.0]
    fixed_gate_choices = [0.45, 0.50, 0.55]
    offset_choices = [0.06, 0.08, 0.10]  # shemo gate offset
    lr_choices = [1/6, 1/7, 1/5]

    best = None
    for t in range(args.trials):
        nseg = random.choice(seg_choices)
        comp = random.choice(comp_choices)
        aperc = random.choice(aperc_choices)
        if args.gate == "fixed":
            lth = random.choice(fixed_gate_choices); off=None; lr=None
        else:
            lth=None; off = random.choice(offset_choices); lr = random.choice(lr_choices)

        cfg = RoiConfig(n_segments=nseg, compactness=comp, a_percentile=aperc,
                        gate=args.gate,
                        lower_half_threshold=lth if lth is not None else 0.5,
                        gate_offset_frac=off if off is not None else 0.08,
                        gate_lr_margin_frac=lr if lr is not None else 1/6)

        ious = []
        for ip in tqdm(imgs, desc=f"trial {t+1}/{args.trials}", leave=False):
            mp = find_mask_for(ip, masks_root, args.mask_suffix)
            if mp is None: continue
            gt = binarize_mask(cv2.imread(str(mp), cv2.IMREAD_UNCHANGED))
            bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            if bgr is None or gt is None: continue
            pred = predict_mask(bgr, cfg)
            ious.append(iou_score(pred, gt))
        if not ious: continue
        mean_iou = float(np.mean(ious))
        cand = (mean_iou, nseg, comp, aperc, lth, off, lr)
        if (best is None) or (mean_iou > best[0]): best = cand
        print(f"[trial {t+1}] mIoU={mean_iou:.4f}  nseg={nseg}  comp={comp}  a%={aperc}  "
              f"gate={args.gate}  fixed={lth}  offset={off}  lr={lr}")

    if best:
        print("\n=== BEST ===")
        print(f"mIoU={best[0]:.4f}  n_segments={best[1]}  compactness={best[2]}  a_percentile={best[3]}  "
              f"gate={args.gate}  lower_half={best[4]}  offset_frac={best[5]}  lr_margin_frac={best[6]}")
    else:
        print("No results (check mask naming/paths).")
