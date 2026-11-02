"""Module 1 - Train a YOLO11 detector on the 4-class RDD dataset.

Examples:
    python src/train.py                                  # full baseline (yolo11l)
    python src/train.py --epochs 1 --fraction 0.1 --name smoke_l   # quick pipeline test
    python src/train.py --resume runs/yolo11l_v2/weights/last.pt   # continue an interrupted run
    python src/train.py --model yolo11s.pt --snake --bra --wiou --optimizer SGD --lrf 0.01 ...  # BSW-YOLO11
    python src/train.py --model yolo11s.pt --gia --cdsa --optimizer SGD --lrf 0.01 ...          # YOLO11 + GIA + CDSA
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from bsw_yolo import BSWDetectionModel, make_trainer  # also needed to unpickle BSW checkpoints
from gc_yolo import GCDetectionModel, make_gc_trainer  # also needed to unpickle GIA/CDSA checkpoints

ROOT = Path(__file__).resolve().parents[1]


def cap_gpu_memory(margin_mb: int = 400, min_gb: float = 3.0) -> None:
    """Limit PyTorch to the VRAM that is actually free.

    On Windows the NVIDIA driver silently spills to system RAM when VRAM runs out,
    which made training ~10x slower with no error. With a cap we get a clear OOM instead.
    Refuses to start when another app (e.g. Steam, a game, a crashed run) holds the GPU,
    since training then only spirals through out-of-memory retries.
    """
    free, total = torch.cuda.mem_get_info()
    cap_gb = (free / 2**30) - margin_mb / 1024
    if cap_gb < min_gb:
        raise SystemExit(f"Only {cap_gb:.2f} GB of GPU memory is free (need {min_gb} GB). "
                         "Close GPU-heavy apps (games, Steam, video) or wait for a crashed run to exit.")
    torch.cuda.set_per_process_memory_fraction((free - margin_mb * 2**20) / total)
    print(f"GPU memory cap: {cap_gb:.2f} GB of {total / 2**30:.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11l.pt", help="pretrained weights, e.g. yolo11m.pt / yolo11l.pt")
    ap.add_argument("--data", default=str(ROOT / "configs" / "rdd4.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30, help="stop if val mAP does not improve for N epochs")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=None,
                    help="default 6 (yolo11l fits batch 6 on a 6 GB GPU); with --resume, overrides the saved batch")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set to use")
    ap.add_argument("--name", default="yolo11l_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="path to last.pt of an interrupted run")
    # Training-recipe knobs used by the improvement experiments (Ultralytics defaults otherwise)
    ap.add_argument("--cos-lr", action="store_true", help="cosine learning-rate schedule")
    ap.add_argument("--close-mosaic", type=int, default=10, help="turn mosaic off for the last N epochs")
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=0.5, help="random scale augmentation gain")
    ap.add_argument("--multi-scale", type=float, default=0.0, help="multi-scale range as a fraction of imgsz")
    ap.add_argument("--cls-pw", type=float, default=0.0,
                    help="inverse class-frequency weights on the cls loss, 0=off .. 1=full (0.5 recommended)")
    ap.add_argument("--optimizer", default="auto",
                    help="auto (Ultralytics picks AdamW or MuSGD by run length) or e.g. SGD as in the base paper")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.01, help="final LR = lr0 * lrf (base paper: 0.01 -> 0.0001)")
    # Base paper (BSW-YOLO) modules
    ap.add_argument("--snake", action="store_true", help="SnakeConv (Dynamic Snake Convolution) before each head")
    ap.add_argument("--bra", action="store_true", help="BRA-DropKey (bi-level routing attention) before SPPF")
    ap.add_argument("--wiou", action="store_true", help="Wise-IoU v3 box loss instead of CIoU")
    ap.add_argument("--gia", action="store_true", help="GIA (gradient-guided image attention) after layers 4 and 6")
    ap.add_argument("--cdsa", action="store_true", help="CDSA (contextual dilated spatial attention) before each head")
    args = ap.parse_args()
    bsw = {"snake": args.snake, "bra": args.bra, "wiou": args.wiou}
    gc = {"gia": args.gia, "cdsa": args.cdsa}
    if any(gc.values()) and any(bsw.values()):
        ap.error("--gia/--cdsa build the YOLO11 + GIA + CDSA model; don't combine them with --snake/--bra/--wiou")

    cap_gpu_memory()
    if args.resume:
        # custom-layer checkpoints must resume with the matching trainer, or those layers would be dropped
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        m = ck["ema"] if ck.get("ema") is not None else ck.get("model")
        trainer = make_trainer(**dict(m.bsw)) if isinstance(m, BSWDetectionModel) else \
            make_gc_trainer(**dict(m.gc)) if isinstance(m, GCDetectionModel) else None
        # Ultralytics lets batch change on resume (to recover from an out-of-memory crash)
        YOLO(args.resume).train(resume=True, **({"trainer": trainer} if trainer else {}),
                                **({"batch": args.batch} if args.batch else {}))
        return
    args.batch = args.batch or 6

    trainer = make_gc_trainer(**gc) if any(gc.values()) else make_trainer(**bsw) if any(bsw.values()) else None
    model = YOLO(args.model)
    model.train(
        **({"trainer": trainer} if trainer else {}),
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        fraction=args.fraction,
        device=0,
        seed=args.seed,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=False,
        plots=True,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        scale=args.scale,
        multi_scale=args.multi_scale,
        cls_pw=args.cls_pw,
    )


if __name__ == "__main__":
    main()
