"""Module 2 - Evaluate a trained model on the held-out test split.

Examples:
    python src/evaluate.py                                                   # yolo11l_v2 best.pt on test
    python src/evaluate.py --weights runs/yolo11l_v2/weights/best.pt --split val

Writes Ultralytics plots (PR curves, confusion matrix, sample predictions) and a
metrics.json to runs/<run>_<split>/.
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

import bsw_yolo  # noqa: F401  (registers BSW-YOLO11 classes so their checkpoints can be loaded)
import gc_yolo  # noqa: F401  (same for the YOLO11 + GIA + CDSA classes)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "runs" / "yolo11l_v2" / "weights" / "best.pt"))
    ap.add_argument("--data", default=str(ROOT / "configs" / "rdd4.yaml"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="GPU index, or 'cpu'")
    ap.add_argument("--out", default=None, help="output folder (default runs/<run>_<split>)")
    args = ap.parse_args()

    weights = Path(args.weights).resolve()
    run_name = weights.parents[1].name  # runs/<run>/weights/best.pt
    out_dir = Path(args.out).resolve() if args.out else ROOT / "runs" / f"{run_name}_{args.split}"

    metrics = YOLO(str(weights)).val(
        data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
        # conf left unset: mAP uses conf=0.001 and the confusion matrix uses 0.25.
        # Passing conf=0.001 would also apply it to the confusion matrix and inflate it.
        iou=0.7, device=args.device, plots=True,
        project=str(out_dir.parent), name=out_dir.name, exist_ok=True,
    )
    box = metrics.box

    # Confidence threshold that maximises mean F1 over classes; used as the default for inference.
    f1_mean = box.f1_curve.mean(0)
    best_conf = float(box.px[f1_mean.argmax()])

    per_class = []
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap5095 = box.class_result(i)
        per_class.append({
            "class": metrics.names[c],
            "images": int(metrics.nt_per_image[c]),
            "instances": int(metrics.nt_per_class[c]),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "mAP50": round(float(ap50), 4),
            "mAP50-95": round(float(ap5095), 4),
        })

    report = {
        "weights": str(weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50-95": round(float(box.map), 4),
        },
        "best_f1_conf": round(best_conf, 3),
        "best_mean_f1": round(float(f1_mean.max()), 4),
        "per_class": per_class,
        "speed_ms_per_image": {k: round(v, 2) for k, v in metrics.speed.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'class':24s} {'imgs':>5s} {'inst':>5s} {'P':>6s} {'R':>6s} {'mAP50':>6s} {'mAP50-95':>8s}")
    for row in per_class:
        print(f"{row['class']:24s} {row['images']:5d} {row['instances']:5d} {row['precision']:6.3f} "
              f"{row['recall']:6.3f} {row['mAP50']:6.3f} {row['mAP50-95']:8.3f}")
    o = report["overall"]
    print(f"{'all':24s} {'':5s} {'':5s} {o['precision']:6.3f} {o['recall']:6.3f} {o['mAP50']:6.3f} {o['mAP50-95']:8.3f}")
    print(f"\nBest mean-F1 {report['best_mean_f1']:.3f} at conf={report['best_f1_conf']}")
    print(f"Saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
