"""Add RDD2022 Japan images that contain transverse cracks (D10), converted from Pascal VOC XML to YOLO txt.

Every Japan train image with at least one D10 box is selected. All four damage classes in those
images are kept (D00->0, D10->1, D20->2, D40->3); dropping them would teach the model that visible
damage is background. Other RDD classes (D43, D44, D50, ...) are dropped.

Reads straight from downloads/RDD2022_Japan.zip; only the selected images are extracted.

Examples:
    python src/add_japan_d10.py
    python src/add_japan_d10.py --sheet japan_d10_crops.jpg
"""
import argparse
import io
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "downloads" / "RDD2022_Japan.zip"
DST = ROOT / "data" / "japan_d10"
CLASS_MAP = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
NAMES = ["D00", "D10", "D20", "D40"]
D10 = CLASS_MAP["D10"]


def xml_size(root: ET.Element) -> tuple[float, float]:
    size = root.find("size")
    if size is None:
        return 0.0, 0.0
    return float(size.findtext("width") or 0), float(size.findtext("height") or 0)


def voc_to_yolo(xml_bytes: bytes, img_size=None):
    """Convert one Pascal VOC annotation to YOLO lines.

    Returns (lines, pixel_boxes, skipped, dropped_names). Coordinates are normalised by img_size
    (the real image w, h) when given, otherwise by the <size> in the XML.
    """
    root = ET.fromstring(xml_bytes)
    w, h = img_size if img_size else xml_size(root)
    lines, boxes, skipped, dropped = [], [], 0, Counter()
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_MAP:
            dropped[name] += 1
            continue
        b = obj.find("bndbox")
        try:
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (AttributeError, TypeError, ValueError):
            skipped += 1
            continue
        x1, x2 = max(0.0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            skipped += 1
            continue
        c = CLASS_MAP[name]
        lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        boxes.append((c, x1, y1, x2, y2))
    return lines, boxes, skipped, dropped


def crop_sheet(dst: Path, out: Path, cls: int = D10, n: int = 24, t: int = 170) -> None:
    """Draw class `cls` boxes from the written YOLO txt files (checks the conversion visually)."""
    random.seed(0)
    lbls = sorted((dst / "labels").glob("*.txt"))
    random.shuffle(lbls)
    sheet = Image.new("RGB", (6 * t, 4 * t), "white")
    k = 0
    for lf in lbls:
        for line in lf.read_text().splitlines():
            p = line.split()
            if int(p[0]) != cls:
                continue
            im = Image.open(dst / "images" / f"{lf.stem}.jpg").convert("RGB")
            W, H = im.size
            x, y, bw, bh = map(float, p[1:])
            x1, y1, x2, y2 = (x - bw / 2) * W, (y - bh / 2) * H, (x + bw / 2) * W, (y + bh / 2) * H
            pad = 0.25 * max(x2 - x1, y2 - y1) + 10
            ox, oy = max(0, x1 - pad), max(0, y1 - pad)
            c = im.crop((ox, oy, min(W, x2 + pad), min(H, y2 + pad)))
            ImageDraw.Draw(c).rectangle([x1 - ox, y1 - oy, x2 - ox, y2 - oy], outline=(255, 0, 0), width=3)
            c.thumbnail((t, t))
            sheet.paste(c, ((k % 6) * t, (k // 6) * t))
            k += 1
            break
        if k >= n:
            break
    sheet.save(out, quality=90)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=str(ZIP))
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--sheet", default=None, help="write a crop sheet of converted D10 boxes to this path")
    args = ap.parse_args()

    dst = Path(args.dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(args.zip)
    xmls = sorted(n for n in z.namelist() if n.startswith("Japan/train/annotations/xmls/") and n.endswith(".xml"))
    boxes_written, imgs_with, skipped_total, dropped_total = Counter(), Counter(), 0, Counter()
    selected = size_mismatch = 0
    for x in xmls:
        stem = Path(x).stem
        xml_bytes = z.read(x)
        root = ET.fromstring(xml_bytes)
        if not any((o.findtext("name") or "").strip() == "D10" for o in root.iter("object")):
            continue
        img_bytes = z.read(f"Japan/train/images/{stem}.jpg")
        W, H = Image.open(io.BytesIO(img_bytes)).size
        if xml_size(root) != (W, H):
            size_mismatch += 1  # normalise by the real image size, not the XML header
        lines, boxes, skipped, dropped = voc_to_yolo(xml_bytes, img_size=(W, H))
        if not any(b[0] == D10 for b in boxes):  # every D10 box was malformed
            continue
        (dst / "images" / f"{stem}.jpg").write_bytes(img_bytes)
        (dst / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        selected += 1
        skipped_total += skipped
        dropped_total += dropped
        for b in boxes:
            boxes_written[b[0]] += 1
        for c in {b[0] for b in boxes}:
            imgs_with[c] += 1

    print(f"Japan train XML files scanned: {len(xmls)}")
    print(f"Images with >=1 D10 (selected): {selected}")
    print("Boxes written (boxes / images):")
    for c, n in enumerate(NAMES):
        print(f"  {n}: {boxes_written[c]:5d} / {imgs_with[c]}")
    print(f"Boxes skipped (malformed or zero-area): {skipped_total}")
    print(f"XML <size> different from the real image size: {size_mismatch}")
    print(f"Boxes of other classes dropped: {dict(dropped_total)}")
    print(f"Output: {dst}")
    if args.sheet:
        crop_sheet(dst, Path(args.sheet))
        print(f"Crop sheet: {args.sheet}")


if __name__ == "__main__":
    main()
