"""Build notebooks/RoadSense_Source_Code.ipynb: the complete RoadSense source code as one Jupyter notebook.

The code cells are the project's source files, unchanged, split at each top-level function or class and
ordered module by module. Each script's `if __name__ == "__main__":` block is replaced by a run cell:
quick steps run a small demo on CPU, slow ones (building datasets, downloads, training, the web server)
show the exact command as a comment.

    python scripts/build_notebook.py             # rebuild after editing src/ (no outputs)
    python scripts/build_notebook.py --execute   # also run every cell on CPU and save the outputs
"""
import argparse
import ast
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "RoadSense_Source_Code.ipynb"
_ids = iter(range(1, 10_000))


def _src(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": f"c{next(_ids):03d}", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": f"c{next(_ids):03d}", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(text)}


def split_source(path: Path) -> tuple[list[str], str]:
    """The file's code split at top-level functions/classes (with their decorators and leading comments),
    and its `if __name__ == "__main__":` block separately."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    body = ast.parse(text).body
    end = len(lines)
    if body and isinstance(body[-1], ast.If) and "__name__" in ast.unparse(body[-1].test):
        end, body = body[-1].lineno - 1, body[:-1]
    starts = [0]
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            while s > 0 and lines[s - 1].lstrip().startswith("#"):
                s -= 1
            starts.append(s)
    starts.append(end)
    chunks = [c for c in ("\n".join(lines[a:b]).strip("\n") for a, b in zip(starts, starts[1:])) if c.strip()]
    guard = "\n".join(lines[end:]).strip("\n")
    # the notebook must hold the file exactly: same non-blank lines in the same order
    rebuilt = [l for c in chunks + [guard] for l in c.splitlines() if l.strip()]
    assert rebuilt == [l for l in lines if l.strip()], f"{path} does not split cleanly"
    return chunks, guard


def source_file(rel: str, purpose: str, run: str | None = None, prelude: str = "") -> list[dict]:
    path = ROOT / rel
    chunks, guard = split_source(path)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    cells = [md(f"### `{rel}` ({n_lines} lines)\n\n{purpose}")]
    if "__file__" in path.read_text(encoding="utf-8"):
        cells.append(code(f'__file__ = str(PROJECT / "{rel}")  '
                          "# notebook only: the code below finds the project folder from its own path" + prelude))
    cells += [code(c) for c in chunks]
    if guard and not run:
        raise ValueError(f"{rel} is a script: give it a run cell")
    if run:
        cells.append(code(run))
    return cells


def text_file(rel: str, purpose: str, lang: str) -> dict:
    body = (ROOT / rel).read_text(encoding="utf-8").rstrip()
    assert "```" not in body
    return md(f"### `{rel}`\n\n{purpose}\n\n```{lang}\n{body}\n```")


# ----------------------------------------------------------------------------------------------- content

INTRO = r"""
# RoadSense: road damage detection, severity, repair priority, Grad-CAM and GPS mapping

This notebook holds the complete source code of the final-year project, module by module. Every code cell is
taken unchanged from the project's own files (split at each function or class), so the notebook matches `src/`.

**What the system does**
1. Detects road damage in photos and dashcam video and classifies it into four types: D00 longitudinal crack,
   D10 transverse crack, D20 alligator crack, D40 pothole.
2. Draws bounding boxes with confidence scores.
3. Estimates the severity of each damage (Low / Medium / High).
4. Recommends a repair priority (P1 within 48 hours … P4 next maintenance cycle) and a treatment.
5. Explains each detection with Grad-CAM (HiResCAM).
6. Maps the damage from photo EXIF GPS or a GPX track (interactive map + GeoJSON).

**Contents**

| Part | Source files |
|---|---|
| Setup | notebook helpers |
| Module 0: Datasets | `prepare_data.py`, `add_japan_d10.py`, `preprocess.py`, `merge_japan_split.py`, `make_trainset.py`, `add_china.py` |
| Module 1: Model and training | `bsw_modules.py`, `bsw_yolo.py`, `train.py`, `scripts/train_watchdog.sh`; alternative model `gc_modules.py`, `gc_yolo.py` |
| Module 2: Evaluation | `evaluate.py`, `postprocess.py`, `eval_pipeline.py` |
| Module 3: Inference | `detector.py`, `predict.py` |
| Module 4: Severity | `severity.py`, `configs/severity.json` |
| Module 5: Repair priority | `priority.py`, `configs/priority.json` |
| Module 6: Explainability | `gradcam.py` |
| Module 7: GPS mapping | `geo_map.py` |
| Pipeline | `roadsense.py` (Modules 3–7 in one command) |
| Dashboard | `dashboard/app.py`, `dashboard/index.html` |
| Ensemble engine | `ensemble.py`, `scripts/tune_ensemble.py`, `configs/ensemble.json` |
| Website | `website/` (FastAPI back end, SQLite storage, accounts, GPS, surveys; HTML/CSS/JS front end) |

**Results** (held-out test splits, each evaluated once)

| Test set | Model | mAP50 | mAP50-95 |
|---|---|---|---|
| India + Japan | BSW-YOLO11s (final model) | 0.498 | 0.207 |
| India only | BSW-YOLO11s | 0.365 | 0.154 |
| India + Japan | RoadSense ensemble (website engine, deployment path) | 0.500 | 0.195 |
| RDD-China (base-paper setting) | YOLO11s | 0.860 | 0.564 |

The base paper (BSW-YOLO, Wang et al., *Eng. Appl. of AI* 163, 2026) reports 0.905 mAP50 on RDD-China.

**Running this notebook.** Open it from the project folder (or `notebooks/`) with the project's Python
environment and run all cells. The demos run on the CPU and take a few minutes. Cells that rebuild datasets,
download data or train are commented out: they take hours, their results are already in `data/` and `runs/`,
and each comment shows the exact command used.
"""

SETUP = r'''
# Notebook setup: find the project folder, make src/ importable, and work from the project root so relative
# paths such as data/... behave exactly as on the command line.
import os
import sys
from pathlib import Path

from IPython.display import Markdown, display

PROJECT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "detector.py").exists())
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))
DEVICE = "cpu"                                                   # demos never compete with training for the GPU
DEMO_IMAGE = "data/india_japan/test/images/India_006851.jpg"     # unseen test photo used by Modules 3-6


def run_cli(main, *args):
    """Run a script's main() as if from a terminal: python src/<script>.py *args"""
    saved = sys.argv
    sys.argv = ["notebook", *map(str, args)]
    try:
        main()
    finally:
        sys.argv = saved


def latest(pattern):
    """Newest run folder matching pattern (the scripts add 2, 3, ... rather than overwrite an old run)."""
    return max(PROJECT.glob(pattern), key=lambda p: p.stat().st_mtime)


def show(path, width=900):
    from IPython.display import Image as ImageOutput  # local name: some source files import PIL's Image
    display(ImageOutput(filename=str(path), width=width))


def show_bgr(img, width=900):
    import cv2
    from IPython.display import Image as ImageOutput
    display(ImageOutput(data=cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), width=width))


print("Project folder:", PROJECT)
'''

M0 = r"""
## Module 0: Datasets

- **Data:**
  - RDD2022 India (from `RDD_SPLIT/`, which is never modified);
  - the RDD2022 Japan images that contain transverse cracks (D10, the class India has fewest of);
  - RDD-China, for the base-paper comparison.
- **Label fix:** `RDD_SPLIT` has class ids 3 and 4 swapped (3 = faded lane markings, 4 = potholes). `prepare_data.py` fixes this.
- **Format:** Japan and China labels are Pascal VOC XML and are converted to YOLO txt.
- **Leak-free split:** Japan images are split 70:15:15 in blocks of consecutive image IDs, so near-duplicate photos of the same road cannot leak from train into test.
- **Guided-filter detail enhancement** (`preprocess.py`) makes the `*_gde` copies used for training. The same `enhance()` runs at inference.
- **Disk space:** images are hard-linked, so each variant takes no extra space.

Build order (already done; results in `data/`):
```bash
python src/prepare_data.py                     # RDD_SPLIT -> data/rdd4 (India, 4 classes)
python src/add_japan_d10.py                    # Japan D10 images, VOC XML -> YOLO txt -> data/japan_d10
python src/preprocess.py --src data/rdd4 --dst data/rdd4_gde --yaml configs/rdd4_gde.yaml
python src/preprocess.py --src data/japan_d10 --dst data/japan_d10_gde
python src/merge_japan_split.py --india data/rdd4 --japan data/japan_d10 --dst data/india_japan --japan-dst data/japan_d10_split
python src/merge_japan_split.py                # guided-filter variant: data/india_japan_gde (training data)
python src/add_china.py                        # RDD-China: data/rdd_china (8:1:1 like the paper)
```
"""

RUN_PREPARE = """
# Command line (already done):  python src/prepare_data.py
# run_cli(main)
"""

RUN_JAPAN = """
# Command line (already done; reads downloads/RDD2022_Japan.zip):  python src/add_japan_d10.py
# run_cli(main)
"""

RUN_PREPROCESS = """
# Command line (already done):  python src/preprocess.py --src data/rdd4 --dst data/rdd4_gde --yaml configs/rdd4_gde.yaml
# Demo: an unseen test photo before (left) and after (right) guided-filter detail enhancement
import cv2
import numpy as np

photo = cv2.imread(DEMO_IMAGE)
photo = cv2.resize(photo, (720, round(720 * photo.shape[0] / photo.shape[1])))
show_bgr(np.hstack([photo, enhance(photo)]))
"""

RUN_MERGE = """
# Command line (already done):
#   python src/merge_japan_split.py --india data/rdd4 --japan data/japan_d10 --dst data/india_japan --japan-dst data/japan_d10_split
#   python src/merge_japan_split.py
# run_cli(main)
"""

RUN_TRAINSET = """
# Used for the earlier "India train + all Japan D10" experiments, before the 70:15:15 Japan split:
#   python src/make_trainset.py --india data/rdd4_gde --japan data/japan_d10_gde --name rdd4_gde_jp
# run_cli(main, "--india", "data/rdd4_gde", "--japan", "data/japan_d10_gde", "--name", "rdd4_gde_jp")
"""

RUN_CHINA = """
# Command line (already done; reads the RDD2022 China zips in downloads/):  python src/add_china.py
# run_cli(main)
"""

DATA_STATS = r'''
# Demo: what the prepared datasets contain (images, and boxes per class, in each split)
import pandas as pd
from collections import Counter


def dataset_table(root):
    classes = ["D00 longitudinal", "D10 transverse", "D20 alligator", "D40 pothole"]
    rows = {}
    for split in ("train", "val", "test"):
        labels = (PROJECT / root / split / "labels").glob("*.txt")
        boxes = Counter(int(l.split()[0]) for f in labels for l in f.read_text().splitlines() if l.strip())
        rows[split] = {"images": sum(1 for _ in (PROJECT / root / split / "images").iterdir()),
                       **{c: boxes[i] for i, c in enumerate(classes)}}
    return pd.DataFrame(rows).T


display(Markdown("**India + Japan** (`data/india_japan`, used for the final model)"), dataset_table("data/india_japan"))
display(Markdown("**RDD-China** (`data/rdd_china`, the base paper's random 8:1:1 split)"), dataset_table("data/rdd_china"))
'''

M1 = r"""
## Module 1: Model and training (BSW-YOLO11)

The detector is **YOLO11s** with the three changes from the base paper (BSW-YOLO, Wang et al. 2026), re-implemented for YOLO11:

| Change | Where | What it does |
|---|---|---|
| **Dynamic Snake Convolution** (`SnakeConvBlock`) | the three neck outputs (layers 16, 19, 22) | kernels that bend along thin, winding cracks |
| **BRA-DropKey** (bi-level routing attention + DropKey) | after SPPF (layer 8) | each region attends only to its most relevant regions |
| **Wise-IoU v3** box loss | the loss | down-weights low-quality (noisy) boxes |

- Every new block starts as an exact identity, so the pretrained YOLO11s weights still work at the start of training.
- **Ablation** on India + Japan val, mAP50:

  | Configuration | mAP50 |
  |---|---|
  | Baseline | 0.488 |
  | + Wise-IoU | 0.490 |
  | + SnakeConv | 0.481 |
  | + BRA | 0.480 |
  | **All three** | **0.497** |
- **Test:** 0.498 vs 0.484 for the plain YOLO11s, the same +1.4 points the paper reports.
"""

RUN_BSW_MODULES = """
# Demo: both new blocks start as an identity (output == input), so they can be inserted into a pretrained YOLO11
import torch

x = torch.randn(1, 128, 40, 40)
for block in (SnakeConvBlock(128).eval(), BRADropKey(128).eval()):
    with torch.no_grad():
        print(f"{block.__class__.__name__:15s} output equals input at initialisation: {torch.allclose(block(x), x)}")
"""

RUN_BSW_YOLO = """
# Demo: build BSW-YOLO11s from plain YOLO11s weights and check that, before training, it predicts exactly the same
import torch
from ultralytics.nn.tasks import DetectionModel

torch.manual_seed(0)
plain = DetectionModel("yolo11s.yaml", nc=4, verbose=False).eval()
bsw = build_bsw_model("yolo11s.yaml", nc=4, ch=3, weights=plain, snake=True, bra=True, wiou=True, verbose=False).eval()
x = torch.rand(1, 3, 640, 640)
with torch.no_grad():
    same = torch.allclose(plain(x)[0], bsw(x)[0], atol=1e-5)
million = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
print(f"YOLO11s {million(plain):.2f} M parameters -> BSW-YOLO11s {million(bsw):.2f} M")
print(f"Same predictions before training: {same}")
print("Wrapped layers:", [f"{i}: {m.type}" for i, m in enumerate(bsw.model) if isinstance(m, Wrapped)])
"""

RUN_TRAIN = """
# Command line (already done; GPU, ~12 h). Final India + Japan model, BSW-YOLO11s (best epoch 87):
#   python src/train.py --model yolo11s.pt --data configs/india_japan_gde.yaml --epochs 150 --patience 50 --batch 12 \\
#       --optimizer SGD --lr0 0.01 --lrf 0.01 --close-mosaic 15 --snake --bra --wiou --name final_bsw_s
# Base-paper setting on RDD-China, YOLO11s (best epoch 263):
#   python src/train.py --model yolo11s.pt --data configs/rdd_china.yaml --imgsz 512 --batch 16 --epochs 300 \\
#       --patience 50 --optimizer SGD --lr0 0.01 --lrf 0.01 --name china_c0_base
# Resume an interrupted run (keeps the BSW blocks):  python src/train.py --resume runs/final_bsw_s/weights/last.pt
#
# Saved result of the final run: validation metrics per epoch (best.pt = best mAP50-95)
import pandas as pd

hist = pd.read_csv("runs/final_bsw_s/results.csv")
best = hist.loc[hist["metrics/mAP50-95(B)"].idxmax()]
print(f"{len(hist)} epochs trained; best epoch {int(best['epoch'])}: precision {best['metrics/precision(B)']:.3f}, "
      f"recall {best['metrics/recall(B)']:.3f}, mAP50 {best['metrics/mAP50(B)']:.3f}, "
      f"mAP50-95 {best['metrics/mAP50-95(B)']:.3f} (India + Japan val)")
show("runs/final_bsw_s/results.png")
"""

WATCHDOG = ("Training helper used on the 6 GB laptop GPU. It waits until enough GPU memory is free (for example "
            "after a game closes), starts or resumes the run, retries with a smaller batch after out-of-memory "
            "errors, and cleans up leftover worker processes.")

M2 = r"""
## Module 2: Evaluation

- **`evaluate.py`:** Ultralytics validation on a split, producing PR curves, a confusion matrix and `metrics.json`.
- **`eval_pipeline.py`:** re-implements the validator's maths so that pre- and post-processing can be scored too. It also tunes the per-class confidence thresholds on the val split; the detector uses these.
- **`postprocess.py`:** test-time augmentation, weighted box fusion and crack-fragment merging. All three were tested and did not improve the final model, so the detector does not use them.
"""

RUN_EVALUATE = """
# Command line (already done; GPU, ~1 min). Final model on the held-out India + Japan test split:
#   python src/evaluate.py --weights runs/final_bsw_s/weights/best.pt --data configs/india_japan_gde.yaml --split test
# Saved result (runs/final_bsw_s_test/):
import json
import pandas as pd

metrics = json.loads(Path("runs/final_bsw_s_test/metrics.json").read_text())
print("Overall:", metrics["overall"])
display(pd.DataFrame(metrics["per_class"]).set_index("class"))
show("runs/final_bsw_s_test/BoxPR_curve.png", width=700)
"""

RUN_EVAL_PIPELINE = """
# Command line (already done). Per-class confidence thresholds used by the detector, tuned on the val split:
#   python src/eval_pipeline.py --weights runs/final_bsw_s/weights/best.pt --data-root data/india_japan_gde --split val \\
#       --tag final_bsw_s_val_combined --save-thresholds configs/thresholds_final_bsw_s.json
print(Path("configs/thresholds_final_bsw_s.json").read_text())
"""

CROSS = r"""
# Saved cross-dataset comparison: how each model does on data it has never seen
display(Markdown(Path("runs/cross_dataset_comparison.md").read_text(encoding="utf-8")))
"""

M3 = r"""
## Module 3: Inference

- **`detector.py`:** the shared inference core used by every later module and by the dashboard.
  - It knows how each model must be fed:
    - India + Japan: resize to 720 px, guided-filter enhancement, 640 px input, per-class thresholds;
    - China: raw 512 px.
  - It reads photo EXIF GPS.
- **`predict.py`:** the command-line tool for a photo, a folder or a video (every Nth frame). It writes:
  - annotated media;
  - `detections.json`, one record per image or frame with boxes, confidences and GPS, which is the input for Modules 4–7;
  - `detections.csv` and `summary.json`.
"""

RUN_DETECTOR = """
# Demo: the Detector API on an unseen test photo
det = Detector("india_japan", device=DEVICE)
for d in det.predict(read_image(DEMO_IMAGE)):
    print(f"{d.name:24s} confidence {d.conf:.2f}  box {[round(v) for v in d.box_xyxy]}")
"""

RUN_PREDICT = """
# Command line:  python src/predict.py --source <photo | folder | video> [--model china] [--name my_run]
run_cli(main, "--source", DEMO_IMAGE, "--device", DEVICE, "--name", "notebook_demo")
DEMO_RUN = latest("runs/predict/notebook_demo*")
show(DEMO_RUN / Path(DEMO_IMAGE).name)
"""

M4 = r"""
## Module 4: Severity

RDD2022 has no severity labels, so severity is a transparent, adjustable rule system in the spirit of
pavement-condition surveys (distress type × extent), not a trained model:

```
score = 100 × type_weight × (0.3 + 0.7 × extent) × (0.6 + 0.4 × confidence)      Low < 35 ≤ Medium < 60 ≤ High
```

- **Type weight:** pothole 1.00, alligator crack 0.85, transverse crack 0.70, longitudinal crack 0.65.
- **Extent:** box area (potholes, alligator cracks) or length (linear cracks), on a log scale relative to that damage type's size range in the training labels.
- **Perspective correction (dashcam model):** damage higher in the frame is farther away, so its size is scaled up.
- **Per image:** worst severity, counts per level, and a 0–100 condition index.
- **On the test set:** Low 42% / Medium 47% / High 11% of detections.

The rules live in `configs/severity.json`:
"""

SEV_CONFIG = 'print(Path("configs/severity.json").read_text())'

RUN_SEVERITY = """
# Command line:  python src/severity.py runs/predict/<name>
run_cli(main, DEMO_RUN)
import json
for d in json.loads((DEMO_RUN / "severity.json").read_text())[0]["detections"]:
    print(f"{d['name']:24s} confidence {d['conf']:.2f} -> severity {d['severity']['level']:6s} (score {d['severity']['score']})")
"""

M5 = r"""
## Module 5: Repair priority

1. **Group** detections into locations:
   - GPS photos and frames within 25 m (leader clustering);
   - else 5-second video segments;
   - else one location per photo.
2. **Score** each location: worst severity, plus a bonus per extra Medium/High damage, plus a pothole safety bonus, times a road-class factor.
   - The same damage seen in several frames counts once.
3. **Map to a level:**

   | Level | Deadline | Score |
   |---|---|---|
   | P1 Urgent | 48 h | ≥ 75 |
   | P2 High | 7 days | ≥ 55 |
   | P3 Medium | 30 days | ≥ 35 |
   | P4 Routine | next maintenance cycle | < 35 |

   Overrides: a High pothole is always P1.
4. **Attach a treatment** per damage type and severity.

The rules live in `configs/priority.json`:
"""

PRI_CONFIG = 'print(Path("configs/priority.json").read_text())'

RUN_PRIORITY = """
# Command line:  python src/priority.py runs/predict/<name> [--road-class highway|arterial|local]
run_cli(main, DEMO_RUN)
display(Markdown((DEMO_RUN / "repair_plan.md").read_text(encoding="utf-8")))
"""

M6 = r"""
## Module 6: Explainability (Grad-CAM / HiResCAM)

- **How it works:** for each detection, the class score of the raw prediction behind that box is back-propagated to the three feature maps the detection head reads. The result is mapped back onto the photo.
- **HiResCAM (default)** weights gradients element-wise, so it explains the individual box.
- **Classic Grad-CAM** (`--method gradcam`) averages gradients per channel, which for a detector gives a class-level map.

Measured on 70 test detections:

| Method | Peak inside the box | Heat concentration in the box | Score drop when its top-5% pixels are blurred (random 5%: ~0.015) |
|---|---|---|---|
| Classic Grad-CAM | 16% | ×2.2 | 0.260 |
| **HiResCAM** | **100%** | **×46.6** | **0.471** |
"""

RUN_GRADCAM = """
# Command line:  python src/gradcam.py --source <photo | folder> [--method gradcam] [--top 3]
run_cli(main, "--source", DEMO_IMAGE, "--device", DEVICE, "--name", "notebook_demo")
show(latest("runs/gradcam/notebook_demo*") / f"{Path(DEMO_IMAGE).stem}_gradcam.jpg")
"""

M7 = r"""
## Module 7: GPS mapping

- **Photos:** positions come from EXIF GPS (read in Module 3).
- **Video:** frames are positioned from a **GPX track**. Each frame's time (video start + offset) is interpolated between track points; frames outside the track get no position.
- **Output:**
  - `damage_map.html`: a self-contained Leaflet/OpenStreetMap map, with locations coloured by priority and popups showing the damage, actions and photo;
  - `damage_locations.geojson`: for GIS tools.
"""

RUN_GEO = """
# Command line:  python src/geo_map.py runs/<name> [--gpx track.gpx --video-start 2026-10-01T08:15:00Z]
# Demo: positions of video frames from the demo GPX track (synthetic, 8 m/s)
track = read_gpx("data/demo/drive/demo_drive_synthetic.gpx")
t0 = track[0][0]
print(f"{len(track)} track points from {t0:%H:%M:%S} to {track[-1][0]:%H:%M:%S}")
for s in (0, 2.5, 10):
    print(f"frame at +{s:>4} s ->", interpolate(track, t0 + timedelta(seconds=s)))
print("1 s before the track starts ->", interpolate(track, t0 - timedelta(seconds=1)))
"""

PIPE = r"""
## Pipeline: Modules 3–7 in one command

`roadsense.py` runs the whole chain:
- detection;
- GPS for video, from a GPX track;
- severity;
- repair priority;
- HiResCAM explanations for the most urgent locations;
- the map;
- a report.

Below it runs on 40 test photos tagged with **synthetic** GPS positions (`data/demo/`), 15 m apart along a street.
"""

RUN_PIPE = """
# Command line:  python src/roadsense.py --source <photos | folder | video> [--gpx track.gpx --video-start ...] --name my_run
run_cli(main, "--source", "data/demo/survey_synthetic_gps", "--device", DEVICE, "--explain", 2, "--name", "notebook_demo_survey")
PIPE_RUN = latest("runs/roadsense/notebook_demo_survey*")
display(Markdown((PIPE_RUN / "report.md").read_text(encoding="utf-8")))
show(sorted((PIPE_RUN / "explanations").glob("*.jpg"))[0])
print("Interactive map (open in a browser):", PIPE_RUN / "damage_map.html")
"""

M1_GC = r"""
### Alternative model: YOLO11s + GDGF + GIA + CDSA (no BSW parts)

Trained with the same guided-filter data, optimiser and 150-epoch schedule as BSW-YOLO11s, but with two different
attention blocks instead of SnakeConv, BRA-DropKey and Wise-IoU:
- **GIA (Gradient-guided Image Attention),** after backbone layers 4 and 6: fixed Sobel filters find strong edges (cracks are thin, high-gradient lines), and spatial and channel gates built from that gradient re-weight the features.
- **CDSA (Contextual Dilated Spatial Attention),** after the neck outputs 16, 19, 22: dilated depthwise convolutions (dilation 1, 2, 4) plus global context give each position a wide view along long cracks.

Both start as an identity. These blocks were designed for this project from their names; no published reference implementation was found.

| Model | Val | India + Japan test | India test | Japan test |
|---|---|---|---|---|
| YOLO11s, plain (A0, 60 epochs) | 0.488 / 0.204 | 0.484 / 0.201 | 0.354 / 0.146 | 0.453 / 0.196 |
| **YOLO11s + GIA + CDSA** (best epoch 81) | 0.488 / 0.204 | 0.493 / 0.209 | 0.359 / 0.155 | 0.466 / 0.206 |
| BSW-YOLO11s (final model) | 0.502 / 0.211 | 0.498 / 0.207 | 0.365 / 0.154 | 0.456 / 0.195 |

(mAP50 / mAP50-95.) It's close to BSW with fewer parameters (9.68 M vs 11.49 M) and faster training (160 vs 291 s/epoch). BSW stays the final model because it's better on val.
"""

RUN_GC_MODULES = """
# Demo: both blocks start as an identity (output == input), so they can be inserted into a pretrained YOLO11
import torch

x = torch.randn(1, 256, 40, 40)
for block in (GIA(256).eval(), CDSA(256).eval()):
    with torch.no_grad():
        print(f"{block.__class__.__name__:5s} output equals input at initialisation: {torch.equal(block(x), x)}")
"""

RUN_GC_YOLO = """
# Command line (already done; GPU, early-stopped at epoch 131, best epoch 81):
#   python src/train.py --model yolo11s.pt --data configs/india_japan_gde.yaml --epochs 150 --patience 50 --batch 12 --optimizer SGD --lr0 0.01 --lrf 0.01 --close-mosaic 15 --gia --cdsa --name gdgf_gia_cdsa_s
# Demo: build YOLO11s + GIA + CDSA from plain YOLO11s weights; before training it predicts exactly the same
import torch
from ultralytics.nn.tasks import DetectionModel

torch.manual_seed(0)
plain = DetectionModel("yolo11s.yaml", nc=4, verbose=False).eval()
gc_model = build_gc_model("yolo11s.yaml", nc=4, ch=3, weights=plain, gia=True, cdsa=True, verbose=False).eval()
x = torch.rand(1, 3, 640, 640)
with torch.no_grad():
    same = torch.allclose(plain(x)[0], gc_model(x)[0], atol=1e-5)
million = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
print(f"YOLO11s {million(plain):.2f} M parameters -> YOLO11s + GIA + CDSA {million(gc_model):.2f} M")
print(f"Same predictions before training: {same}")
print("Wrapped layers:", [f"{i}: {m.type.split('.')[-1]}" for i, m in enumerate(gc_model.model) if isinstance(m, GCWrapped)])
show("runs/gdgf_gia_cdsa_s/results.png")
"""

DASH = r"""
## Dashboard: three models side by side

A local web page that runs every photo (an upload, or a random unseen India + Japan or China test image) through
**three models**, each in its own column:
- BSW-YOLO11s (the final model);
- YOLO11s + GIA + CDSA;
- YOLO11s trained on China.

Each column shows:
- detections with confidence and severity;
- found / missed / false alarms against the ground truth;
- the repair-priority banner;
- optional HiResCAM heatmaps.

A summary table compares the models: detections per class, priorities, precision, recall and speed. Thresholds can be each model's own (tuned on its val split) or one shared slider.

The server is FastAPI (`app.py`), and the page is a single HTML file (`index.html`).
"""

RUN_DASH = """
# Command line (keeps running; then open http://127.0.0.1:8765 in a browser):
#   python dashboard/app.py                          (all three models)
#   python dashboard/app.py --models india_japan     (one model; the notebook loads only this one, to stay quick)
# Quick check without starting the server: the models the page's /info request describes
[{k: m[k] for k in ("short", "weights", "epoch", "thresholds", "metrics")} for m in info()["models"]]
"""


ENS = r"""
## Ensemble engine (used by the website)

The website runs three India + Japan YOLO11s models together: BSW-YOLO11s, YOLO11s + GIA + CDSA and plain
YOLO11s. The photo is guided-filtered once and each model predicts. Weighted boxes fusion then averages overlapping
boxes; a box only one model sees keeps a third of its score.

`scripts/tune_ensemble.py` picks the fusion IoU, the per-class thresholds and a confidence calibration on the
validation split, then scores the test split once. Test results (deployment code path, India + Japan):

| | Box P | Box R | Box F1 | mAP50 | Photos judged right |
|---|---|---|---|---|---|
| BSW-YOLO11s alone | 0.545 | 0.479 | 0.506 | 0.473 | 89.8 % |
| Ensemble | 0.559 | 0.521 | 0.537 | 0.500 | 89.4 % |

The shown confidence is calibrated: on test photos it differs from the real hit rate by 2.8 points on average.
"""

RUN_ENSEMBLE = """
det = EnsembleDetector(device=DEVICE)
img = read_image(DEMO_IMAGE)
for d in det.predict(img):
    print(f"{d.name:24s} calibrated {d.conf:.2f}  fused score {d.score:.3f}  models agreeing {d.agree}/3")
print("Photo quality:", det.quality(img))
"""

RUN_TUNE = """
# Command line (GPU, ~3 min each; member predictions are cached in runs/ensemble/):
#   python scripts/tune_ensemble.py --split val     # tune -> configs/ensemble.json
#   python scripts/tune_ensemble.py --split test    # score once -> runs/ensemble/test.json
import json
t = json.loads(Path("runs/ensemble/test.json").read_text())
for k in ("combined", "india", "japan"):
    for m in ("ensemble", "bsw_alone"):
        r = t[k][m]
        print(f"{k:9s} {m:10s} F1 {r['at_thresholds']['f1']:.3f}  mAP50 {r['mAP50']:.3f}  photos right {r['image_level']['photo_accuracy']:.1%}")
"""

WEB_MD = r"""
## Website

`website/` is the finished application built on Modules 3–7:
- **New report:** a photo goes through the ensemble, severity and repair priority. Its location comes from the
  photo's EXIF GPS, the phone's live GPS (accuracy stored), or a map pin, and gets a street address from
  OpenStreetMap.
- **Damage map and Reports:** filters, status workflow (open, in progress, repaired, rejected), history, and CSV
  or GeoJSON export.
- **Surveys:** batches of geotagged photos, or dash-cam video with a GPX track, processed in the background and
  grouped into 25 m repair locations.
- **Model comparison and About:** the dashboard's side-by-side view, and the measured accuracy.
- **Accounts:** inspector and admin roles, scrypt-hashed passwords, and HTTPS for phones.

Run it with `python website/app.py`, then open http://localhost:8000. For phones, run
`bash scripts/make_cert.sh` once, then `python website/app.py --https`.
"""

RUN_WEB = """
# Command line (keeps running):  python website/app.py   ->  http://localhost:8000
# Quick check without a browser: sign in and analyse a photo through the HTTP API (temporary data folder).
from fastapi.testclient import TestClient
import auth as web_auth
client = TestClient(app)
if not STORE.one("SELECT id FROM users WHERE username = 'notebook'"):
    web_auth.create_user(STORE, "notebook", "Notebook demo", "inspector", "notebook-demo-1")
H = {"X-RoadSense": "1"}
client.post("/api/login", json={"username": "notebook", "password": "notebook-demo-1"}, headers=H)
res = client.post("/api/analyze", files={"file": ("demo.jpg", Path(DEMO_IMAGE).read_bytes(), "image/jpeg")}, headers=H).json()
print("detections:", [(d["name"], round(d["conf"], 2), d["severity"]["level"]) for d in res["detections"]])
print("priority:", res["plan"]["priority"] if res["plan"] else "no damage", "| photo quality warnings:", res["quality"]["warnings"])
"""


def build() -> dict:
    cells = [md(INTRO), md("## Setup"), code(SETUP), md(M0)]
    cells += source_file("src/prepare_data.py", "RDD_SPLIT (India) → `data/rdd4`: keeps the 4 damage classes and fixes the swapped class ids.", RUN_PREPARE)
    cells += source_file("src/add_japan_d10.py", "Japan images with transverse cracks: Pascal VOC XML → YOLO txt.", RUN_JAPAN)
    cells += source_file("src/preprocess.py", "Guided-filter detail enhancement (He et al., *Guided Image Filtering*, ECCV 2010) on the LAB luminance channel.", RUN_PREPROCESS)
    cells += source_file("src/merge_japan_split.py", "Leak-free 70:15:15 Japan split merged into the India train/val/test splits.", RUN_MERGE)
    cells += source_file("src/make_trainset.py", "Training-list builder from the earlier India + Japan experiments.", RUN_TRAINSET)
    cells += source_file("src/add_china.py", "RDD-China (ChinaDrone + ChinaMotorBike) built as in the base paper: 3,853 images / 7,718 boxes.", RUN_CHINA)
    cells += [code(DATA_STATS), md(M1)]
    cells += source_file("src/bsw_modules.py", "The three BSW building blocks: Dynamic Snake Convolution, BRA-DropKey attention, Wise-IoU v3 loss.", RUN_BSW_MODULES)
    cells += source_file("src/bsw_yolo.py", "Inserts the blocks into YOLO11 (layers 8, 16, 19, 22), swaps in the Wise-IoU loss, and provides the trainer.", RUN_BSW_YOLO)
    cells += source_file("src/train.py", "Training entry point for the plain and BSW models (fresh or resumed).", RUN_TRAIN)
    cells.append(text_file("scripts/train_watchdog.sh", WATCHDOG, "bash"))
    cells.append(md(M1_GC))
    cells += source_file("src/gc_modules.py", "The two blocks: GIA (gradient-guided image attention) and CDSA (contextual dilated spatial attention).", RUN_GC_MODULES)
    cells += source_file("src/gc_yolo.py", "Inserts GIA after layers 4 and 6 and CDSA after 16, 19 and 22 of a pretrained YOLO11, and provides the trainer.", RUN_GC_YOLO)
    cells.append(md(M2))
    cells += source_file("src/evaluate.py", "Ultralytics validation of a trained model on a split.", RUN_EVALUATE)
    cells += source_file("src/postprocess.py", "Post-processing experiments: test-time augmentation, weighted box fusion, crack-fragment merging, best-F1 thresholds.")
    cells += source_file("src/eval_pipeline.py", "Validator-equivalent scoring of the full pipeline (preprocessing → model → post-processing).", RUN_EVAL_PIPELINE)
    cells += [code(CROSS), md(M3)]
    cells += source_file("src/detector.py", "Model registry, `Detector`, image/EXIF-GPS reading and drawing.", RUN_DETECTOR)
    cells += source_file("src/predict.py", "Command-line inference for photos, folders and videos.", RUN_PREDICT)
    cells.append(md(ENS))
    cells += source_file("src/ensemble.py", "Three-model WBF ensemble with tuned thresholds, calibrated confidence and a photo-quality check.", RUN_ENSEMBLE)
    cells += source_file("scripts/tune_ensemble.py", "Val tuning (fusion IoU, thresholds, calibration, quality limits) and the one-off test scoring.", RUN_TUNE)
    cells += [md(M4), code(SEV_CONFIG)]
    cells += source_file("src/severity.py", "Severity score and level for each detection, plus a per-image summary.", RUN_SEVERITY)
    cells += [md(M5), code(PRI_CONFIG)]
    cells += source_file("src/priority.py", "Locations, priority scores, P1–P4 levels, treatments and the repair plan.", RUN_PRIORITY)
    cells.append(md(M6))
    cells += source_file("src/gradcam.py", "HiResCAM / Grad-CAM for individual YOLO detections, with localisation checks.", RUN_GRADCAM)
    cells.append(md(M7))
    cells += source_file("src/geo_map.py", "GPX reading and interpolation, video geotagging, GeoJSON and the HTML map.", RUN_GEO)
    cells.append(md(PIPE))
    cells += source_file("src/roadsense.py", "The end-to-end pipeline and its report.", RUN_PIPE)
    cells.append(md(DASH))
    cells += source_file("dashboard/app.py", "FastAPI server: loads several models; `/predict?model=…`, `/explain?model=…` (HiResCAM), `/sample?set=…` (random unseen test image with ground truth), `/info`.", RUN_DASH,
                         prelude='\nsys.argv = ["app.py", "--models", "india_japan", "--device", DEVICE]  '
                                 "# notebook only: app.py reads its options at start-up")
    cells.append(text_file("dashboard/index.html", "The dashboard page (HTML + JavaScript). Its priority rules are a port of `priority.py` and read the same `configs/priority.json`.", "html"))
    cells += [md(WEB_MD), code('sys.path.insert(0, str(PROJECT / "website"))  # the website modules import each other by name')]
    cells += source_file("website/geo.py", "EXIF GPS from uploaded photo bytes, coordinate checks, distances and OpenStreetMap street addresses.")
    cells += source_file("website/store.py", "SQLite schema and queries: users, sessions, reports, detections, media, history, surveys, address cache.")
    cells += source_file("website/auth.py", "Accounts, scrypt password hashing, sessions and login rate limiting.")
    cells += source_file("website/engine.py", "Analysis engine: ensemble, severity, priority, annotated images, HiResCAM, comparison models.")
    cells += source_file("website/surveys.py", "Background survey jobs: geotagged photo batches and dash-cam video + GPX.")
    cells += source_file("website/app.py", "FastAPI app: security headers, accounts, analysis, reports, map, exports, stats, surveys, comparison.", RUN_WEB,
                         prelude='\nsys.argv = ["app.py", "--data", str(PROJECT / "runs" / "notebook_demo_website"), "--device", DEVICE]  '
                                 "# notebook only: a temporary data folder, not the real website/data")
    cells.append(text_file("website/manage.py", "Terminal administration: `create-user`, `list-users`, `set-password`, `seed-dev`, `backup-reset`.", "python"))
    cells.append(text_file("scripts/make_cert.sh", "Self-signed HTTPS certificate for phones on the same Wi-Fi.", "bash"))
    web_files = [("website/static/index.html", "Page shell", "html"),
                 ("website/static/css/app.css", "Design tokens, layout and components (light and dark)", "css"),
                 ("website/static/js/theme.js", "Theme applied before first paint", "javascript"),
                 ("website/static/js/app.js", "Session, layout and router", "javascript"),
                 ("website/static/js/api.js", "Fetch wrapper", "javascript"),
                 ("website/static/js/ui.js", "Shared UI helpers and icons", "javascript"),
                 ("website/static/js/viewer.js", "Photo viewer with boxes, model input and heat map", "javascript"),
                 ("website/static/js/mapkit.js", "Map helpers and the device-location fix", "javascript")]
    web_files += [(f"website/static/js/views/{v}.js", f"Page: {v}", "javascript") for v in
                  ("login", "overview", "newreport", "map", "reports", "report", "surveys", "lab", "about", "users", "account")]
    for rel, what, lang in web_files:
        cells.append(text_file(rel, what + ".", lang))
    cells.append(text_file("website/tests/test_website.py", "Website tests (run: `python website/tests/test_website.py`).", "python"))
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": platform.python_version()}},
        "nbformat": 4, "nbformat_minor": 5}


# ------------------------------------------------------------------------------------------- execution

def _tidy(text: str) -> str:
    """Keep what a terminal would show: drop progress-bar frames overwritten by carriage returns."""
    return "\n".join(line.rsplit("\r", 1)[-1] for line in text.replace("\r\n", "\n").split("\n"))


def execute(nb: dict, timeout: int = 3600) -> None:
    """Run every code cell in a fresh kernel (GPU hidden, so everything runs on CPU) and store the outputs."""
    from jupyter_client.manager import start_new_kernel

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    km, kc = start_new_kernel(kernel_name="python3", cwd=str(OUT.parent), env=env)
    try:
        cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        for k, cell in enumerate(cells, 1):
            msg_id = kc.execute("".join(cell["source"]))
            outputs = []
            while True:
                msg = kc.get_iopub_msg(timeout=timeout)
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                kind, c = msg["msg_type"], msg["content"]
                if kind == "status" and c["execution_state"] == "idle":
                    break
                if kind == "stream":
                    if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == c["name"]:
                        outputs[-1]["text"] += c["text"]
                    else:
                        outputs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
                elif kind in ("display_data", "execute_result"):
                    out = {"output_type": kind, "data": c["data"], "metadata": c["metadata"]}
                    if kind == "execute_result":
                        out["execution_count"] = c["execution_count"]
                    outputs.append(out)
                elif kind == "error":
                    outputs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"],
                                    "traceback": c["traceback"]})
                elif kind == "clear_output":
                    outputs = []
            while True:
                reply = kc.get_shell_msg(timeout=timeout)
                if reply["parent_header"].get("msg_id") == msg_id:
                    break
            for o in outputs:
                if o["output_type"] == "stream":
                    o["text"] = _tidy(o["text"])
            cell["outputs"], cell["execution_count"] = outputs, reply["content"].get("execution_count")
            first = "".join(cell["source"]).splitlines()[0][:70]
            print(f"[{k}/{len(cells)}] {reply['content']['status']:5s} {first}", flush=True)
            if reply["content"]["status"] != "ok":
                err = next((o for o in outputs if o["output_type"] == "error"), {})
                raise SystemExit(f"Cell {k} failed: {err.get('ename')}: {err.get('evalue')}")
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="run all cells on CPU and save their outputs")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    nb = build()
    if args.execute:
        execute(nb)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
    print(f"Saved {out} ({len(nb['cells'])} cells, {n_code} code; {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
