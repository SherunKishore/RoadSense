"""Guided-filter detail enhancement (optionally with CLAHE) for road images.

The image is split into a smooth base layer and a detail layer with the guided filter
(He et al., "Guided Image Filtering", ECCV 2010). The detail layer, which holds crack
edges, is amplified by k and added back. Only the LAB luminance channel is changed.

The same enhance() must be used for training data and at inference.

Examples:
    python src/preprocess.py --src data/rdd4 --dst data/rdd4_gde --yaml configs/rdd4_gde.yaml
    python src/preprocess.py --src data/rdd4 --dst data/rdd4_clahe_gde --clahe --yaml configs/rdd4_clahe_gde.yaml
    python src/preprocess.py --sheet 6 --src data/rdd4 --out preview.jpg   # before/after preview only
"""
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png"}


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Grayscale guided filter. I (guide) and p (input) are float32 images in [0, 1]."""
    ksize = (2 * r + 1, 2 * r + 1)

    def box(x):
        return cv2.boxFilter(x, -1, ksize, borderType=cv2.BORDER_REFLECT)

    mean_I, mean_p = box(I), box(p)
    var_I = box(I * I) - mean_I * mean_I
    cov_Ip = box(I * p) - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return box(a) * I + box(b)


def enhance(img_bgr: np.ndarray, r: int = 8, eps: float = 0.01, k: float = 2.0, clahe: bool = False) -> np.ndarray:
    """Boost fine luminance detail (crack edges) with a self-guided filter."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    if clahe:
        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    Lf = L.astype(np.float32) / 255.0
    base = guided_filter(Lf, Lf, r, eps)
    out = np.clip(base + k * (Lf - base), 0.0, 1.0)
    lab[..., 0] = (out * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def process_tree(src: Path, dst: Path, workers: int = 8, **kw) -> int:
    """Mirror every images/ dir under src into dst (enhanced), copying the sibling labels/ dir."""
    jobs = []
    for img_dir in sorted(p for p in src.rglob("images") if p.is_dir()):
        rel = img_dir.relative_to(src)
        out_img = dst / rel
        out_img.mkdir(parents=True, exist_ok=True)
        lbl_dir = img_dir.parent / "labels"
        if lbl_dir.is_dir():
            out_lbl = dst / rel.parent / "labels"
            out_lbl.mkdir(parents=True, exist_ok=True)
            for f in lbl_dir.glob("*.txt"):
                shutil.copy2(f, out_lbl / f.name)
        jobs += [(f, out_img / f.name) for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXT]

    def run(job):
        s, d = job
        cv2.imwrite(str(d), enhance(cv2.imread(str(s)), **kw), [cv2.IMWRITE_JPEG_QUALITY, 95])

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, jobs))
    return len(jobs)


def write_yaml(base_yaml: Path, dst: Path, out_yaml: Path, note: str) -> None:
    cfg = yaml.safe_load(base_yaml.read_text())
    cfg["path"] = dst.resolve().as_posix()
    out_yaml.write_text(f"# {note}\n" + yaml.safe_dump(cfg, sort_keys=False))


def preview_sheet(src: Path, out: Path, n: int, **kw) -> None:
    """Side-by-side original | enhanced for n random train images."""
    rng = np.random.default_rng(0)
    files = sorted((src / "train" / "images").glob("*.jpg"))
    rows = []
    for f in rng.choice(files, n, replace=False):
        im = cv2.imread(str(f))
        rows.append(np.hstack([im, enhance(im, **kw)]))
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "data" / "rdd4"))
    ap.add_argument("--dst", default=None)
    ap.add_argument("--yaml", default=None, help="write a dataset yaml for dst (names copied from configs/rdd4.yaml)")
    ap.add_argument("--r", type=int, default=8, help="guided filter radius (px)")
    ap.add_argument("--eps", type=float, default=0.01, help="guided filter regularisation")
    ap.add_argument("--k", type=float, default=2.0, help="detail gain")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--sheet", type=int, default=0, help="only write a before/after preview of N images")
    ap.add_argument("--out", default="preview.jpg")
    args = ap.parse_args()

    kw = dict(r=args.r, eps=args.eps, k=args.k, clahe=args.clahe)
    src = Path(args.src)
    if args.sheet:
        preview_sheet(src, Path(args.out), args.sheet, **kw)
        print(f"Preview: {args.out}")
        return
    if not args.dst:
        raise SystemExit("--dst is required unless --sheet is used")
    dst = Path(args.dst)
    n = process_tree(src, dst, **kw)
    print(f"Enhanced {n} images -> {dst}  ({kw})")
    if args.yaml:
        write_yaml(ROOT / "configs" / "rdd4.yaml", dst, Path(args.yaml), f"{dst.name}: guided-filter detail enhancement {kw}")
        print(f"Config: {args.yaml}")


if __name__ == "__main__":
    main()
