# RoadSense

Road damage detection, severity scoring and repair prioritisation from road photos and dash-cam video.
RoadSense finds four kinds of pavement damage, rates how severe each one is, groups them into repair locations
with a priority (P1 urgent to P4 routine), explains each detection with a heat map and puts everything on a map
in a web application.

The detector is **GGC-YOLO**: YOLO11s with guided-filter detail enhancement (GDGF) at the input, Gradient-guided
Image Attention (GIA) in the backbone and Contextual Dilated Spatial Attention (CDSA) in front of the detection head.
The website runs it in an ensemble with two other YOLO11s detectors.

Final-year project, Department of Information Technology, PSG College of Technology, Coimbatore.

## Damage classes

| Code | Damage |
|---|---|
| D00 | Longitudinal crack |
| D10 | Transverse crack |
| D20 | Alligator crack |
| D40 | Pothole |

## Results

Dataset: RDD2022 India plus the RDD2022 Japan images that contain transverse cracks (9,946 images).
Split 6,935 train / 1,507 validation / 1,504 test. The test split was evaluated once.

| Model | Params | GFLOPs | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | GPU (FP16) | CPU (FP32) |
|---|---|---|---|---|---|---|---|---|
| YOLO11s | 9.41 M | 21.4 | 0.546 | 0.481 | 0.484 | 0.201 | 7.2 ms | 120.6 ms |
| **GGC-YOLO** | 9.67 M | 22.3 | 0.539 | 0.493 | 0.494 | **0.209** | 9.5 ms | 137.0 ms |
| BSW-YOLO11s (our re-implementation of BSW-YOLO, Wang et al. 2026) | 11.48 M | 24.2 | 0.558 | 0.493 | 0.498 | 0.206 | 12.4 ms | 183.3 ms |

Latency: batch 1, 640×640, fused models, RTX 4050 Laptop GPU and a 6-thread laptop CPU.

The website combines the three models with weighted boxes fusion, per-class thresholds and calibrated
confidence. Measured through the website's own code on the same test split:

| Model | Box P | Box R | Box F1 | mAP@0.5 | Photos judged right | Right damage type |
|---|---|---|---|---|---|---|
| YOLO11s | 0.556 | 0.470 | 0.508 | 0.468 | 89.5% | 96.7% |
| GGC-YOLO | 0.549 | 0.464 | 0.502 | 0.466 | 89.1% | 96.7% |
| BSW-YOLO11s | 0.545 | 0.479 | 0.506 | 0.472 | 89.8% | 98.1% |
| **Ensemble** | **0.559** | **0.521** | **0.537** | **0.500** | 89.4% | 97.7% |

The ensemble's reported confidence is within 2.8% of the observed accuracy (expected calibration error over
1,964 test boxes). Full numbers: `runs/ensemble/*.json` and `runs/bench_models.json`.

## Repository layout

```
website/   FastAPI web application (reports, damage map, surveys, users) and its single-page front end
src/       detection, GDGF preprocessing, GIA/CDSA and BSW modules, ensemble, severity, priority,
           HiResCAM explanations, GPS mapping, training and evaluation scripts
configs/   ensemble settings, severity and priority rules, per-class thresholds, dataset YAMLs
scripts/   ensemble tuning, model benchmark, HTTPS certificate, notebook builder
runs/      the three trained models used by the website (best.pt) and the saved test results
```

## Run the website

Python 3.11 or newer.

```bash
pip install -r requirements.txt
python website/app.py
```

Open http://localhost:8000. On the first start the site asks you to create the admin account (only from the
computer running it). More accounts can be added from the Users page or with
`python website/manage.py create-user --username NAME --role inspector`.

Options: `--device auto` uses the GPU when there is one (default CPU). For phones on the same Wi-Fi, which need
HTTPS for the camera and GPS, run `bash scripts/make_cert.sh` once and start with `--https`, then open
`https://<computer-ip>:8443` on the phone.

## Command-line pipeline

```bash
python src/roadsense.py --source survey_photos/ --name ward12            # detect, score, prioritise, map, report
python src/roadsense.py --source dashcam.mp4 --gpx dashcam.gpx --video-start 2026-10-01T08:15:00Z
python src/predict.py --source photo.jpg                                  # detection only
```

Results go to `runs/roadsense/<name>/`: annotated media, severity and priority tables, a repair plan, heat maps and
an HTML damage map.

## Training from scratch

The datasets are not in this repository. Download RDD2022 (India and Japan) from the
[RDD2022 release](https://github.com/sekilab/RoadDamageDetector), then:

```bash
python src/prepare_data.py                                         # 4-class India dataset (fixes swapped class ids)
python src/add_japan_d10.py                                        # Japan images with transverse cracks
python src/preprocess.py --src data/rdd4 --dst data/rdd4_gde --yaml configs/rdd4_gde.yaml   # GDGF-enhanced copy
python src/make_trainset.py --india data/rdd4_gde --japan data/japan_d10_gde --name rdd4_gde_jp
python src/train.py --model yolo11s.pt --gia --cdsa --optimizer SGD --lrf 0.01 --data configs/india_japan_gde.yaml
python scripts/tune_ensemble.py                                    # ensemble thresholds and calibration
```

Training settings for GGC-YOLO: COCO-pretrained YOLO11s, SGD (lr 0.01 to 0.0001, momentum 0.937, weight decay
5e-4), batch 12, 640×640, up to 150 epochs with patience 50, mixed precision, seed 0. Best checkpoint at epoch 81.

## Limitations

- Each model was trained once (seed 0); differences of about one mAP point are within run-to-run variation.
- Images come from RDD2022 India and Japan only. Night, rain and other camera set-ups are untested.
- Severity and priority come from transparent hand-set rules (`configs/severity.json`, `configs/priority.json`),
  not from measured crack width or depth. An inspector should confirm P1 reports.

## Team

Arun V, Dhanusiyaa M, Ramana T, Sherun Kishore K. Guide: Ms. Suguna G, Department of Information Technology,
PSG College of Technology.

## Credits

[RDD2022](https://github.com/sekilab/RoadDamageDetector) dataset (Arya et al., 2024),
[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0),
weighted boxes fusion (Solovyev et al., 2021), HiResCAM (Draelos and Carin, 2020), Leaflet and OpenStreetMap.
