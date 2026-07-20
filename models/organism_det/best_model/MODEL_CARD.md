---
license: other
license_name: wiseyak-proprietary
license_link: LICENSE
tags:
- object-detection
- medical-imaging
- cytology
- yolo
- ultralytics
library_name: ultralytics
pipeline_tag: object-detection
---

# organism-detector-yolo26n-v2

**Author:** Pritam Thapa | AI Researcher | AISCS

Detects 4 organism types in liquid-based cytology (LBC) Papanicolaou-stained image patches, as a
screening-assist aid for pathologist review. **Not a standalone diagnostic tool** — see Limitations.

**Weights file:** `organism-detector-yolo26n-v2.pt`
**Base architecture:** YOLO26n

## Model description

- **Architecture:** YOLO26n (2.4M params, 5.4B FLOPs), fine-tuned from COCO-pretrained weights.
- **Input:** 1280×720 RGB image patches, 1280px inference resolution (required — the smallest
  detected class has ~32×37px objects that fall below reliable detection size at 640px).
- **Output:** bounding boxes + class + confidence for 4 organism classes, NMS-free (one-to-one head).
- **Classes:**

  | ID | Class | Test AP@50 |
  |---|---|---|
  | 0 | Trichomonas vaginalis | 0.443 |
  | 1 | Bacterial vaginosis flora shift | 0.727 |
  | 2 | Candida spp. | 0.381 |
  | 3 | Actinomyces spp. | 0.995 |

## Intended use

- A **screening-assist aid**: flags candidate organisms for a pathologist to review, biased toward
  recall (a missed flag is worse than an extra glance).
- Every prediction is intended to be reviewed by a qualified pathologist before any clinical action.
  **This model has not been validated as a standalone diagnostic device.**
- Not intended for use outside the LBC Papanicolaou-stain imaging pipeline it was trained on
  (fixed 1280×720 patch format, specific scanner/lab sources — see Training data).

## Training data

- 473 LBC cytology image patches (378 train / 47 val / 48 test), 1280×720, from two
  scanner/lab sources (institutional, not a public dataset).
- Class instance counts: BV 969, TV 377, Candida 259, Actinomyces 42 — **significant class
  imbalance (~23:9:6:1)**, most severe for Actinomyces.
- Labels derived from pathologist-annotated segmentation masks, converted to bounding boxes.

## Training procedure

Fine-tuned from `yolo26n.pt` (COCO pretrained), 200 epochs, batch 8, image size 1280,
AdamW (auto-selected), cosine LR (`lrf=0.001`). Key augmentation beyond defaults:
`copy_paste=0.6` (rare-class instance pasting — this is what fixed Actinomyces' overfitting,
see Limitations), `mixup=0.15`, `multi_scale=0.5`, `degrees=15`, `hsv` jitter tuned for the
Papanicolaou stain's brighter-than-ImageNet color profile.

Full recipe and rationale: `models/organism_det/results/reports/best_model_report.md` and
`models/organism_det/notebooks/04_train_yolo26n_v2.ipynb`. Supporting plots (PR curves, confusion
matrix, sample predictions): `models/organism_det/results/plots/yolo26n_v2/`.

## Evaluation results

Test set (48 images, held out from all training/tuning decisions):

| Metric | Value |
|---|---|
| mAP@50 | 0.6365 |
| mAP@50-95 | 0.3734 |
| Precision | 0.6048 |
| Recall | 0.6210 |

Val→test mAP@50 gap: -0.0037 (0.6402 → 0.6365) — evaluated on a single fixed split, not
cross-validated (see Limitations).

## Limitations

- **Trichomonas vaginalis (the most clinically time-sensitive class) is this model's weakest
  class** (AP@50 0.443). Do not rely on this model to rule out TV; a negative prediction is not
  a negative finding.
- **Candida spp. is also weak** (AP@50 0.381), consistent across every recipe tried so far —
  likely a data property (high size variance in the source images), not yet solved by augmentation.
- **Actinomyces' near-perfect score (0.995 AP@50) is based on only 42 real training instances**
  and heavy synthetic augmentation (`copy_paste`). Treat this number with more caution than the
  others — it reflects strong performance on a narrow, augmentation-heavy training signal, not a
  richly-sampled class.
- **Single train/val/test split, not cross-validated.** All numbers above come from one 47-image
  val / 48-image test partition. A 3-fold cross-validation of a *different* training recipe on
  this same dataset showed metrics can swing by ±0.1-0.3 AP@50 across folds for the rarer classes
  — this model's own recipe has not yet been re-verified with that methodology.
- Trained on a specific institutional imaging pipeline (scanner sources, patch format, stain
  protocol). Performance on other LBC imaging setups is unknown.
- No formal clinical validation has been performed. This is a research/screening-assist model.

## How to use

```python
from ultralytics import YOLO

model = YOLO("organism-detector-yolo26n-v2.pt")
results = model.predict("path/to/patch.jpg", imgsz=1280, conf=0.25)
```

## License

**Proprietary — Copyright (c) 2026 Wiseyak Solutions Pvt Ltd. All rights reserved.** See the
repository `LICENSE` file. Not licensed for external use, reproduction, or distribution without
written permission from Wiseyak Solutions Pvt Ltd.

*Note:* this model is fine-tuned from Ultralytics YOLO26 base weights.

## Citation

Base architecture: Jocher et al., *Ultralytics YOLO26: Unified Real-Time End-to-End Vision
Models*, 2026. arXiv:2606.03748.
