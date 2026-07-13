# Organism Detection — Model List & Dataset Analysis

**Author:** Pritam Thapa | AI Researcher | AISCS

## Organisms in Dataset

| Class | ID | Instances | Median BBox Area | Notes |
|---|---|---|---|---|
| Trichomonas vaginalis (TV) | 0 | 377 | ~1,184 px² (~32×37) | **Very small — hardest class** |
| Bacterial vaginosis flora shift (BV) | 1 | 969 | ~39,400 px² (~198×199) | Dominant class |
| Candida spp. | 2 | 259 | ~32,600 px² (~180×180) | High size variance |
| Actinomyces spp. | 3 | 42 | ~5,054 px² (~71×71) | Rare class |
| None / Unknown / Reactive changes | — | 742 | — | Exclude — treat as background |

**4-class detection task.** Class IDs above are used in YOLO label files.

---

## Dataset Summary

- **Total images:** 473 — train: 378 / val: 47 / test: 48 (all perfectly image-label paired)
- **Image size:** 1280×720 RGB JPEG, all uniform
- **Label format:** `.npy` per image → `data.item()` → dict:
  - `masks`: `(720, 1280)` int32, pixel value = instance ID (0 = background)
  - `bboxes`: list of `[x1, y1, x2, y2]` pixel coords
  - `instance_ids`: list of ints matched to bboxes (globally unique per case, not per-image sequential)
  - `labels`: Pap cytology result (NILM/ASC-US/ASC-H) — NOT the organism class
  - `attributes`: list of `{'Organisms': <name>}` — **this is the detection class**
- **Cohort prefixes:** `Cy-` and `EH-` (two scanner/lab sources)
- **Magnification:** `10X` suffix files are all Candida spp. at different magnification
- **Multi-organism images:** 22 images contain >1 organism type

---

## Critical Data Observations

### Small Object Problem — Trichomonas vaginalis
TV has a median bounding box of ~32×37 px in a 1280×720 image. At 640 input resolution this shrinks to ~16×19 px — below the reliable detection floor of most detectors. **Training must use 1280-input resolution.** YOLO26's STAL (Small object label maintenance) was built exactly for this.

### Class Imbalance
BV:TV:Candida:Actinomyces ≈ 23:9:6:1. Mitigation: copy-paste augmentation for Actinomyces (42 instances) and TV; focal loss is built into YOLO26's progressive loss schedule.

### Color Profile vs. ImageNet
LBC Papanicolaou stain is much brighter than natural images:
- Dataset mean (0–1 scale): R=0.749, G=0.805, B=0.689
- Dataset std: R=0.111, G=0.095, B=0.052
- Use **color jitter augmentation** (hue ±10°, saturation ×0.8–1.2) during training rather than offline stain normalization — the inter-slide variation in this dataset is narrow enough that augmentation covers it.

---

## Augmentation Strategy

| Augmentation | Setting | Reason |
|---|---|---|
| Horizontal + vertical flip | On | Organisms have no fixed orientation |
| Rotation | ±15° | Same reason |
| Mosaic | On (YOLO default) | Artificial scene diversity for 378 images |
| Copy-paste | On | Critical for Actinomyces (42 inst.) and TV |
| Color jitter | hue ±10°, sat ×0.8–1.2, bright ×0.9–1.1 | Stain variation simulation |
| Random scale | 0.5×–2.0× | Multi-magnification robustness |
| Elastic distortion | **Off** | Distorts cell morphology |
| Heavy blur | **Off** | Destroys TV fine structure |

---

## Models for Benchmarking — YOLO26 (n / s / m)

### Why YOLO26 over YOLO11 and YOLO12

| Feature | YOLO11 | YOLO12 | **YOLO26** |
|---|---|---|---|
| Small object handling | P2 head only | Area attention | **STAL — label maintenance for small obj.** |
| NMS at inference | Required | Required | **NMS-free (one-to-one head default)** |
| Optimizer | SGD/Adam | SGD/Adam | **MuSGD (hybrid SGD+Muon)** |
| Loss schedule | Standard | Standard | **Progressive Loss (train→inference aligned)** |
| COCO mAP (nano) | ~37.3 | ~40.6 | **40.9** |
| COCO mAP (small) | ~48.0 | ~48.4 | **48.6** |
| COCO mAP (medium) | ~51.5 | ~52.5 | **53.1** |

YOLO26's STAL directly addresses the TV small-object problem. NMS-free inference simplifies the clinical patch pipeline (no NMS threshold tuning for overlapping organisms). MuSGD gives better convergence on small datasets like ours (378 train images).

---

## The 3 Models

### YOLO26n — Speed Baseline
| Property | Value |
|---|---|
| Parameters | 2.4M |
| FLOPs | 5.4B |
| COCO mAP | 40.9 |
| T4 TensorRT latency | 1.7ms |
| Training input | 1280 |
| Batch size (8GB budget) | 8 |
| Est. VRAM | ~4GB |

Use case: real-time screening pipelines with strict latency constraints. TV recall may be lower than s/m — benchmark confirms.

### YOLO26s — Primary Recommendation
| Property | Value |
|---|---|
| Parameters | 9.5M |
| FLOPs | 20.7B |
| COCO mAP | 48.6 |
| T4 TensorRT latency | 2.5ms |
| Training input | 1280 |
| Batch size (8GB budget) | 8 |
| Est. VRAM | ~6GB |

Best speed/accuracy balance. Enough capacity to learn TV morphology at small scale. Expected primary winner for production deployment.

### YOLO26m — Accuracy Ceiling
| Property | Value |
|---|---|
| Parameters | 20.4M |
| FLOPs | 68.2B |
| COCO mAP | 53.1 |
| T4 TensorRT latency | 4.7ms |
| Training input | 1280 |
| Batch size (8GB budget) | 4 |
| Est. VRAM | ~8GB |

Maximum accuracy. Use as the reference ceiling — if YOLO26s reaches within 1–2 mAP points, deploy the s variant. If Actinomyces recall is still poor at s-scale, m is the fallback.

---

## Training Plan

All three models trained sequentially on a single RTX 4090 (24GB), budgeted to ~8GB VRAM each.

| Model | Epochs | Batch | Input | Est. time |
|---|---|---|---|---|
| YOLO26n | 150 | 8 | 1280 | ~45 min |
| YOLO26s | 150 | 8 | 1280 | ~1.5 hr |
| YOLO26m | 150 | 4 | 1280 | ~2.5 hr |

Total sequential run: **~4.5 hours**. Pretrained weights: COCO (Ultralytics default).

---

## Benchmark Metrics

Report for each model:

| Metric | What it measures |
|---|---|
| mAP@50 (all) | Overall detection quality |
| mAP@50-95 (all) | Strict localization quality |
| AP@50 per class | Per-organism detection (TV is the critical one) |
| AP_small | Small object performance (≤32² px) — TV proxy |
| Recall @ conf=0.5 per class | Clinical sensitivity per organism type |
| FPS (1280 input, T4) | Deployment feasibility |
| Inference latency (ms) | Per-patch pipeline speed |

---

## Project Structure

```
models/organism_det/
├── configs/
│   ├── dataset.yaml          # YOLO dataset config (paths + class names)
│   ├── yolo26n.yaml
│   ├── yolo26s.yaml
│   └── yolo26m.yaml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_npy_to_yolo_converter.ipynb   # .npy → YOLO txt labels
│   ├── 03_train_benchmark.ipynb         # trains n/s/m sequentially
│   └── 04_benchmark_results.ipynb       # PR curves, confusion matrix, AP tables
├── checkpoints/
└── results/
    ├── metrics/    # per-model CSV: AP tables
    ├── plots/      # PR curves, confusion matrices, sample predictions
    └── reports/    # final benchmark summary
```

---

## Data Conversion Notes (.npy → YOLO)

```
class_id  cx  cy  w  h   (all normalized 0–1, space-separated)
```

Class mapping: `{'Trichomonas vaginalis': 0, 'Bacterial vaginosis flora shift': 1, 'Candida spp.': 2, 'Actinomyces spp.': 3}`

Skip any instance where `attributes['Organisms']` is `'None'`, `'Unknown'`, or `'Reactive changes'`.

Bbox conversion from `[x1, y1, x2, y2]` pixels to YOLO normalized `cx cy w h`:
```
cx = (x1 + x2) / 2 / img_width
cy = (y1 + y2) / 2 / img_height
w  = (x2 - x1) / img_width
h  = (y2 - y1) / img_height
```

Image dimensions: width=1280, height=720 (all images uniform).

**Note on instance_ids:** values are globally unique per case (not 1-indexed per image). To extract the binary mask for instance `i`: `mask_i = (masks == instance_ids[i]).astype(uint8)`.
