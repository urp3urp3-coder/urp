from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import argparse, itertools

import cv2, numpy as np
from skimage import color, segmentation, measure, morphology, util

from v3_roi_pipeline import RoiConfig, run_single_image

# ---------------- File utils ----------------
def collect_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg"}
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: (str(p.parent).lower(), p.name.lower())
    )

def choose_last_per_folder(paths: List[Path], sort_by: str) -> List[Path]:
    result = []
    for leaf, group in itertools.groupby(paths, key=lambda p: p.parent):
        items = list(group)
        if sort_by == "mtime":
            items.sort(key=lambda p: p.stat().st_mtime)
        else:
            items.sort(key=lambda p: p.name.lower())
        result.append(items[-1])
    return result

# ---------------- Main ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--out", default="./v3_1_roi_out")
    ap.add_argument("--take-last", action="store_true")
    ap.add_argument("--sort-by", default="name", choices=["name", "mtime"])
    ap.add_argument("--dry-run", action="store_true")

    # ROI 파라미터 -> RoiConfig에 주입
    ap.add_argument("--gate", choices=[None, "shemo"], default=None)
    ap.add_argument("--segments", type=int, default=160)
    ap.add_argument("--compactness", type=float, default=10.0)
    ap.add_argument("--a_percentile", type=float, default=80)

    # (옵션) 게이트 추가 파라미터도 받기
    ap.add_argument("--gate-offset-frac", type=float, default=None)
    ap.add_argument("--gate-lr-frac", type=float, default=None)

    ap.add_argument("--no-skin-suppress", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    inp = Path(args.input)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    paths = collect_images(inp)
    if args.take_last:
        paths = choose_last_per_folder(paths, args.sort_by)

    # ------- RoiConfig에 CLI 파라미터 주입 -------
    cfg = RoiConfig()
    cfg.use_skin_suppress = not args.no_skin_suppress

    # v3_roi_pipeline.RoiConfig의 필드명에 맞춰 주입
    if hasattr(cfg, "n_segments"):   cfg.n_segments   = args.segments
    if hasattr(cfg, "compactness"):  cfg.compactness  = args.compactness
    if hasattr(cfg, "a_percentile"): cfg.a_percentile = args.a_percentile
    if args.gate is not None and hasattr(cfg, "gate"):
        cfg.gate = args.gate
    if args.gate_offset_frac is not None and hasattr(cfg, "gate_offset_frac"):
        cfg.gate_offset_frac = args.gate_offset_frac
    if args.gate_lr_frac is not None and hasattr(cfg, "gate_lr_margin_frac"):
        cfg.gate_lr_margin_frac = args.gate_lr_frac
    # -------------------------------------------

    if args.dry_run:
        for p in paths: print(str(p))
        print(f"\n(dry-run) total: {len(paths)} file(s)")
        return

    for p in paths:
        # ❗ run_single_image는 cfg만 넘깁니다 (키워드 인자 X)
        print(run_single_image(p, out_dir, cfg))

if __name__ == "__main__":
    main()
