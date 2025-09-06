from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from skimage import color, segmentation, measure, morphology, util

# ---------------- Config ----------------
@dataclass
class RoiConfig:
    resize: int = 320

    gaussian_ksize: int = 5
    gaussian_sigma: float = 1.0
    use_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    n_segments: int = 360
    compactness: float = 12.0
    a_percentile: float = 80.0

    suppress_specular: bool = True
    specular_v_th: float = 0.90
    specular_s_th: float = 0.25

    # 게이트
    gate: str = "shemo"                 # {"fixed","shemo"}
    lower_half_threshold: float = 0.50  # fixed 모드
    gate_offset_frac: float = 0.08      # shemo 모드
    gate_lr_margin_frac: float = 1/6    # shemo 모드

    border: int = 6
    min_component_area: int = 500
    min_area_frac: float = 0.02
    max_area_frac: float = 0.40

    close_radius: int = 3
    open_radius: int = 2

    save_mask: bool = True
    save_overlay: bool = True
    save_preprocessed: bool = False


# ---------- letterbox / unletterbox ----------
def letterbox_bgr(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    top = (size - nh) // 2; left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    meta = {"scale": scale, "left": left, "top": top, "orig_w": w, "orig_h": h, "box": size}
    return canvas, meta

def unletterbox_mask(mask_small: np.ndarray, meta: dict) -> np.ndarray:
    s = meta["scale"]; left = meta["left"]; top = meta["top"]
    w = meta["orig_w"]; h = meta["orig_h"]
    hh = int(round(h * s)); ww = int(round(w * s))
    crop = mask_small[top:top + hh, left:left + ww]
    return cv2.resize(crop.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

# ---------- 전처리 ----------
def _clahe_L_only(bgr: np.ndarray, clip: float, grid: int) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, a, b]), cv2.COLOR_LAB2BGR)

def preprocess_for_mask(bgr_base: np.ndarray, cfg: RoiConfig) -> np.ndarray:
    out = bgr_base.copy()
    if cfg.use_clahe:
        out = _clahe_L_only(out, cfg.clahe_clip, cfg.clahe_grid)
    out = cv2.GaussianBlur(out, (cfg.gaussian_ksize, cfg.gaussian_ksize), cfg.gaussian_sigma)
    return out

# ---------- 스펙큘러 ----------
def _specular_mask(bgr: np.ndarray, v_th: float, s_th: float) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    spec = (v >= int(v_th * 255)) & (s <= int(s_th * 255))
    spec = morphology.binary_dilation(spec, morphology.disk(1))
    return spec.astype(bool)

# ---------- sHEMO 게이트 ----------
def _auto_canny(gray: np.ndarray, sigma: float = 0.55):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def shemo_lower_gate(img_bgr_small: np.ndarray, cfg: RoiConfig) -> np.ndarray:
    H, W = img_bgr_small.shape[:2]
    gray = cv2.cvtColor(img_bgr_small, cv2.COLOR_BGR2GRAY)
    edges = _auto_canny(gray, sigma=0.55)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        cy = int(0.5 * H)
    else:
        areas = [cv2.contourArea(c) for c in cnts]
        c = cnts[int(np.argmax(areas))]
        M = cv2.moments(c)
        cy = int(M["m01"] / (M["m00"] + 1e-6))

    gate_y = min(H - 1, int(cy + cfg.gate_offset_frac * H))
    gate = np.zeros((H, W), np.uint8)
    xl = int(W * cfg.gate_lr_margin_frac); xr = int(W * (1 - cfg.gate_lr_margin_frac))
    gate[gate_y:, xl:xr] = 1
    return gate

# ---------- SLIC ----------
def slic_conjunctiva_mask(img_bgr: np.ndarray, cfg: RoiConfig) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_f = util.img_as_float(img_rgb)
    H, W = img_rgb.shape[:2]

    lab = color.rgb2lab(img_rgb_f)
    a = lab[..., 1]

    labels = segmentation.slic(
        img_rgb_f, n_segments=cfg.n_segments, compactness=cfg.compactness,
        start_label=0, channel_axis=-1, enforce_connectivity=True,
    )

    props = measure.regionprops(labels + 1, intensity_image=a)
    mean_a = {p.label - 1: p.mean_intensity for p in props}
    th = np.percentile(list(mean_a.values()), cfg.a_percentile)
    keep = {k for k, v in mean_a.items() if v >= th}
    candidate = np.isin(labels, list(keep))

    if cfg.suppress_specular:
        spec = _specular_mask(img_bgr, cfg.specular_v_th, cfg.specular_s_th)
        candidate &= ~spec

    if cfg.gate == "fixed":
        Y = np.arange(H)[:, None]
        candidate &= (Y >= int(cfg.lower_half_threshold * H))
    else:
        gate = shemo_lower_gate(img_bgr, cfg)
        candidate &= gate.astype(bool)

    if cfg.border > 0:
        candidate[:cfg.border, :] = False
        candidate[-cfg.border:, :] = False
        candidate[:, :cfg.border] = False
        candidate[:, -cfg.border:] = False

    candidate = morphology.remove_small_objects(candidate, cfg.min_component_area)
    candidate = morphology.binary_closing(candidate, morphology.disk(cfg.close_radius))
    candidate = morphology.binary_opening(candidate, morphology.disk(cfg.open_radius))

    # 연결 성분 선택
    labeled = measure.label(candidate)
    if labeled.max() == 0:
        return np.zeros_like(candidate, dtype=np.uint8)
    regions = measure.regionprops(labeled)

    # 후보 중 "눈 중심 근처에 있는 것"만 남기기
    H, W = img_bgr.shape[:2]
    eye_cy = H // 2
    valid_regions = []
    for r in regions:
        cy = r.centroid[0]
        if cy < eye_cy * 1.2:  # 화면 전체 높이의 0.6~1.2배 범위 이내
            valid_regions.append(r)

    if not valid_regions:
        regions.sort(key=lambda r: r.area, reverse=True)
        target = regions[0]
    else:
        valid_regions.sort(key=lambda r: r.area, reverse=True)
        target = valid_regions[0]

    mask = (labeled == target.label)
    return mask.astype(np.uint8)


# ---------- ROI ----------
def overlay_mask(img_bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    color = np.zeros_like(img_bgr); color[..., 2] = 255
    # 전체 프레임 단위로 한 번에 블렌딩
    blended = cv2.addWeighted(img_bgr, 1 - alpha, color, alpha, 0)

    # 마스크로 최종 합성
    out = img_bgr.copy()
    m = (mask01 > 0)
    out[m] = blended[m]
    return out

def extract_roi_crop(img_bgr: np.ndarray, mask01: np.ndarray, pad: int = 6):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_bgr, (0, 0, img_bgr.shape[1] - 1, img_bgr.shape[0] - 1)
    x1, x2 = xs.min(), xs.max(); y1, y2 = ys.min(), ys.max()
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(img_bgr.shape[1] - 1, x2 + pad); y2 = min(img_bgr.shape[0] - 1, y2 + pad)
    crop = img_bgr[y1:y2 + 1, x1:x2 + 1]
    return crop, (int(x1), int(y1), int(x2), int(y2))

def roi_only(img_bgr: np.ndarray, mask01: np.ndarray, bg_val: int = 0) -> np.ndarray:
    out = img_bgr.copy(); out[mask01 == 0] = bg_val; return out

def roi_only_crop(img_bgr: np.ndarray, mask01: np.ndarray, pad: int = 6, bg_val: int = 0) -> np.ndarray:
    full = roi_only(img_bgr, mask01, bg_val=bg_val)
    crop, _ = extract_roi_crop(full, mask01, pad=pad); return crop

# ---------- 메인 ----------
def run_single_image(img_path: Path, out_dir: Path, cfg: RoiConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr_orig = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr_orig is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    small_bgr, meta = letterbox_bgr(bgr_orig, size=cfg.resize)
    small_for_mask = preprocess_for_mask(small_bgr, cfg)
    mask_small = slic_conjunctiva_mask(small_for_mask, cfg)
    mask01 = unletterbox_mask(mask_small, meta).astype(np.uint8)

    crop, bbox = extract_roi_crop(bgr_orig, mask01)
    over = overlay_mask(bgr_orig, mask01)
    roi_full = roi_only(bgr_orig, mask01, bg_val=0)
    roi_cropped = roi_only_crop(bgr_orig, mask01, pad=6)

    stem = img_path.stem
    crop_path = out_dir / f"{stem}_crop.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    mask_path = out_dir / f"{stem}_mask.png"
    roi_path = out_dir / f"{stem}_roi.png"
    roi_crop_path = out_dir / f"{stem}_roi_crop.png"

    cv2.imwrite(str(crop_path), crop)
    if cfg.save_overlay: cv2.imwrite(str(overlay_path), over)
    if cfg.save_mask:    cv2.imwrite(str(mask_path), (mask01 * 255))
    cv2.imwrite(str(roi_path), roi_full)
    cv2.imwrite(str(roi_crop_path), roi_cropped)

    return {
        "input": str(img_path),
        "crop": str(crop_path),
        "overlay": str(overlay_path) if cfg.save_overlay else "",
        "mask": str(mask_path) if cfg.save_mask else "",
        "roi": str(roi_path),
        "roi_crop": str(roi_crop_path),
        "bbox": bbox,
    }
