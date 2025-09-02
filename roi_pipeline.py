from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from skimage import color, segmentation, measure, morphology, util


@dataclass
class RoiConfig:
    resize: int = 200
    gaussian_ksize: int = 5
    gaussian_sigma: float = 1.0
    n_segments: int = 220
    compactness: float = 12.0
    a_percentile: float = 80.0
    min_component_area: int = 500
    relax_steps: int = 2

    # 저장 옵션 (원본색 유지)
    save_mask: bool = True
    save_overlay: bool = True
    save_preprocessed: bool = False  # 디버깅용, 기본 비저장

    # 마스크 품질 보강 옵션
    use_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    suppress_specular: bool = True
    specular_v_th: float = 0.92
    specular_s_th: float = 0.20
    lower_half_only: bool = True
    lower_half_threshold: float = 0.35
    border: int = 6
    min_area_frac: float = 0.02
    max_area_frac: float = 0.40


# ---------- 기본 유틸: 리사이즈+패딩(색 보정 없음) ----------
def resize_and_pad(bgr: np.ndarray, target: int) -> tuple[np.ndarray, dict]:
    h, w = bgr.shape[:2]
    scale = target / max(h, w)
    resized = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pad_h = target - resized.shape[0]
    pad_w = target - resized.shape[1]
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT)
    tf = {"scale": scale, "top": top, "left": left, "H": out.shape[0], "W": out.shape[1]}
    return out, tf


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
# ------------------------------------------------------------


def _specular_mask(bgr: np.ndarray, v_th: float, s_th: float):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    spec = (v >= int(v_th * 255)) & (s <= int(s_th * 255))
    spec = morphology.binary_dilation(spec, morphology.disk(1))
    return spec.astype(bool)


def slic_conjunctiva_mask(img_bgr: np.ndarray, cfg: RoiConfig) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_float = util.img_as_float(img_rgb)
    H, W = img_rgb.shape[:2]

    lab = color.rgb2lab(img_rgb_float)
    L, a, _ = lab[..., 0], lab[..., 1], lab[..., 2]

    labels = segmentation.slic(
        img_rgb_float,
        n_segments=cfg.n_segments,
        compactness=cfg.compactness,
        start_label=0,
        channel_axis=-1,
        enforce_connectivity=True,
    )

    props_a = measure.regionprops(labels + 1, intensity_image=a)
    mean_a = {p.label - 1: p.mean_intensity for p in props_a}

    th = np.percentile(list(mean_a.values()), cfg.a_percentile)
    keep = {k for k, v in mean_a.items() if v >= th}
    candidate = np.isin(labels, list(keep))

    if cfg.suppress_specular:
        spec = _specular_mask(img_bgr, cfg.specular_v_th, cfg.specular_s_th)
        candidate &= ~spec

    if cfg.lower_half_only:
        Y = np.arange(H)[:, None]
        candidate &= (Y >= int(cfg.lower_half_threshold * H))

    if cfg.border > 0:
        candidate[:cfg.border, :] = False
        candidate[-cfg.border:, :] = False
        candidate[:, :cfg.border] = False
        candidate[:, -cfg.border:] = False

    candidate = morphology.remove_small_objects(candidate, cfg.min_component_area)
    candidate = morphology.binary_closing(candidate, morphology.disk(3))
    candidate = morphology.binary_opening(candidate, morphology.disk(2))

    frac = candidate.sum() / (H * W)
    if frac < cfg.min_area_frac or frac > cfg.max_area_frac:
        sorted_labels = [k for k, _ in sorted(mean_a.items(), key=lambda kv: kv[1], reverse=True)]
        tmp = np.zeros_like(candidate, dtype=bool)
        for k in sorted_labels:
            tmp |= (labels == k)
            t = tmp.copy()
            if cfg.suppress_specular:
                t &= ~spec
            if cfg.lower_half_only:
                Y = np.arange(H)[:, None]
                t &= (Y >= int(cfg.lower_half_threshold * H))
            if cfg.border > 0:
                t[:cfg.border, :] = False; t[-cfg.border:, :] = False
                t[:, :cfg.border] = False; t[:, -cfg.border:] = False
            t = morphology.binary_closing(t, morphology.disk(3))
            t = morphology.binary_opening(t, morphology.disk(2))
            f = t.sum() / (H * W)
            if cfg.min_area_frac <= f <= cfg.max_area_frac:
                candidate = t
                break

    labeled = measure.label(candidate)
    if labeled.max() == 0:
        return np.zeros_like(candidate, dtype=np.uint8)
    regions = measure.regionprops(labeled)
    regions.sort(key=lambda r: r.area, reverse=True)
    mask = (labeled == regions[0].label)

    return mask.astype(np.uint8)


def extract_roi_crop(img_bgr: np.ndarray, mask01: np.ndarray, pad: int = 6) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_bgr, (0, 0, img_bgr.shape[1], img_bgr.shape[0])
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(img_bgr.shape[1] - 1, x2 + pad); y2 = min(img_bgr.shape[0] - 1, y2 + pad)
    crop = img_bgr[y1:y2+1, x1:x2+1]
    bbox = (int(x1), int(y1), int(x2), int(y2))
    return crop, bbox


def overlay_mask(img_bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = img_bgr.copy()  # 원본 색
    color_mask = np.zeros_like(img_bgr)
    color_mask[..., 2] = (mask01 * 255)
    return cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0.0)


def run_single_image(img_path: Path, out_dir: Path, cfg: RoiConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr_orig = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr_orig is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    # 1) 원본 색 유지 버전(리사이즈+패딩만)과
    bgr_base, _tf = resize_and_pad(bgr_orig, cfg.resize)
    # 2) 마스크 계산용 전처리 버전
    bgr_for_mask = preprocess_for_mask(bgr_base, cfg)

    # 3) 마스크 계산은 전처리 이미지로, 크롭/오버레이는 원본색 이미지로
    mask01 = slic_conjunctiva_mask(bgr_for_mask, cfg)
    crop, bbox = extract_roi_crop(bgr_base, mask01)

    stem = img_path.stem
    crop_path = out_dir / f"{stem}_crop.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    mask_path = out_dir / f"{stem}_mask.png"
    preproc_path = out_dir / f"{stem}_preprocessed.png"

    cv2.imwrite(str(crop_path), crop)  # <-- 원본 색 유지 (중요!)

    if cfg.save_overlay:
        cv2.imwrite(str(overlay_path), overlay_mask(bgr_base, mask01))
    if cfg.save_mask:
        cv2.imwrite(str(mask_path), (mask01 * 255))
    if cfg.save_preprocessed:
        cv2.imwrite(str(preproc_path), bgr_for_mask)

    return {
        "input": str(img_path),
        "crop": str(crop_path),
        "overlay": str(overlay_path) if cfg.save_overlay else "",
        "mask": str(mask_path) if cfg.save_mask else "",
        "bbox": bbox,
    }
