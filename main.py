from __future__ import annotations
import argparse
from pathlib import Path
import os
import re

from roi_pipeline import RoiConfig, run_single_image


def parse_args():
    ap = argparse.ArgumentParser(description="Conjunctiva ROI extractor (SLIC-based)")
    ap.add_argument("--input", "-i", type=str, required=True, help="Image file or directory (root)")
    ap.add_argument("--out", "-o", type=str, default="./roi_out", help="Output directory")
    ap.add_argument("--segments", type=int, default=220, help="SLIC n_segments")
    ap.add_argument("--compactness", type=float, default=12.0, help="SLIC compactness")
    ap.add_argument("--a_percentile", type=float, default=80.0, help="Percentile on a* to select conjunctiva candidates")

    # 패턴 기반 필터(원래 방식)
    ap.add_argument("--only-targets", action="store_true",
                    help="Use curated name patterns (India/Italy) to filter files")

    # ⬇ 새 옵션: 폴더마다 '마지막 파일'만 선택
    ap.add_argument("--take-last", action="store_true",
                    help="Select only the last image in each numeric subfolder (or in the given folder if it has images)")
    ap.add_argument("--sort-by", choices=["name", "mtime"], default="name",
                    help="Criterion for 'last' when --take-last (default: name)")

    # 실행 모드
    ap.add_argument("--dry-run", action="store_true", help="List files that would be processed and exit")
    return ap.parse_args()


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------- 기존 패턴(원하면 같이 사용 가능) ----------------
RE_INDIA_FLEX = re.compile(
    r"""
    ^\d{8}_\d{6}            # 날짜_시간
    (?:_\d{1,4})?           # _카메라번호(옵션)
    (?:_                    # 접미사 블록(옵션)
        (?:
            palpebral
          | fornic(?:e)?al(?:_palpebral)?   # forniceal, fornical, forniceal_palpebral
        )
    )?
    \.(?:png|jpe?g)$
    """, re.IGNORECASE | re.VERBOSE
)
RE_ITALY_NUMERIC   = re.compile(r"^\d+\.(?:jpe?g|png)$", re.IGNORECASE)
RE_ITALY_T_PATTERN = re.compile(r"^T_\d+_\d{8}_\d{6}(?:_(?:palpebral|forniceal|forniceal_palpebral))?\.(?:png|jpe?g)$",
                                re.IGNORECASE)

def matches_curated_patterns(p: Path) -> bool:
    n = p.name
    return bool(
        RE_INDIA_FLEX.fullmatch(n) or
        RE_ITALY_NUMERIC.fullmatch(n) or
        RE_ITALY_T_PATTERN.fullmatch(n)
    )
# ------------------------------------------------------------------


# 자연 정렬 키(1,2,10 순으로 정렬)
_num_re = re.compile(r"(\d+)")
def natural_key(p: Path):
    parts = _num_re.split(p.name.lower())
    return [int(t) if t.isdigit() else t for t in parts]


def pick_last_in_dir(d: Path, sort_by: str) -> Path | None:
    """디렉터리 d 안에서 마지막 이미지를 1개 선택."""
    files = [f for f in d.iterdir() if f.is_file() and is_image(f)]
    if not files:
        return None
    if sort_by == "mtime":
        files.sort(key=lambda x: x.stat().st_mtime)  # 수정시각 오름차순
    else:  # name
        files.sort(key=natural_key)
    return files[-1]


def select_last_files(root: Path, sort_by: str) -> list[Path]:
    """
    숫자 폴더(예: India/1.., Italy/1..)마다 마지막 이미지를 하나씩 선택.
    숫자 폴더가 없다면 root 디렉터리 자체에서 마지막 1개를 선택.
    """
    numeric_dirs = sorted(
        {d for d in root.rglob("*") if d.is_dir() and d.name.isdigit()},
        key=lambda x: int(x.name)
    )
    selected: list[Path] = []

    if numeric_dirs:
        for d in numeric_dirs:
            last = pick_last_in_dir(d, sort_by)
            if last:
                selected.append(last)
    else:
        # 바로 아래에 이미지가 있다면 그 폴더에서 1개 선택
        last = pick_last_in_dir(root, sort_by)
        if last:
            selected.append(last)
    return selected


def main():
    args = parse_args()
    inp = Path(args.input)
    out_dir = Path(args.out)

    cfg = RoiConfig(
        n_segments=args.segments,
        compactness=args.compactness,
        a_percentile=args.a_percentile,
    )

    # 파일 하나 입력이면 그대로 처리
    if inp.is_file():
        if args.only_targets and not matches_curated_patterns(inp):
            print(f"(skip) Not matching curated patterns: {inp.name}")
            return
        if args.dry_run:
            print(str(inp))
            return
        info = run_single_image(inp, out_dir, cfg)
        print(info)
        return

    # 디렉터리 입력
    to_process: list[Path] = []

    if args.take_last:
        # '마지막 파일' 전략 (패턴 무시)
        to_process = select_last_files(inp, args.sort_by)
    else:
        # 기존 방식: 전체 재귀 + (옵션) 패턴 필터
        for p in inp.rglob("*"):
            if p.is_file() and is_image(p):
                if (not args.only_targets) or matches_curated_patterns(p):
                    to_process.append(p)

    if args.dry_run:
        if to_process:
            for p in to_process:
                print(str(p))
            print(f"\n(dry-run) total: {len(to_process)} file(s)")
        else:
            print("(dry-run) No matched files.")
        return

    if not to_process:
        raise ValueError(f"No images selected in {inp}")

    for p in to_process:
        info = run_single_image(p, out_dir, cfg)
        print(info)


if __name__ == "__main__":
    main()

