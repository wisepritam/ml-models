# Cervical Cancer ML Models

**Author:** Pritam Thapa | AI Researcher | AISCS

Machine learning models for cervical cancer screening support. The active work in this repository
is organism detection in liquid-based cytology (LBC) Papanicolaou-stained image patches — flagging
candidate organisms for pathologist review.

## Best model

The current best model is `organism-detector-yolo26n-v2`, a fine-tuned YOLO26n checkpoint. See:

- [`models/organism_det/best_model/`](models/organism_det/best_model/) — model weights, model card, and license
- [`models/organism_det/results/reports/best_model_report.md`](models/organism_det/results/reports/best_model_report.md) — full results, rationale, and what to try next

## Repository layout

```
models/organism_det/
├── best_model/       # curated release: weights, MODEL_CARD.md, LICENSE
├── configs/          # dataset.yaml and training configs
├── notebooks/        # data conversion, training, and evaluation notebooks (00-07)
├── compare_models.py # benchmark comparison script
└── results/
    ├── metrics/      # per-run CSV metrics
    ├── plots/        # PR curves, confusion matrices, sample predictions
    └── reports/      # written analysis and model reports
```

## Reports

- [`training_report.md`](training_report.md) — v1/v2 training analysis and hyperparameter changes
- [`organism_list.md`](organism_list.md) — dataset composition and class notes
- [`models/organism_det/results/reports/yolo26_model_review.md`](models/organism_det/results/reports/yolo26_model_review.md) — why YOLO26 was chosen for this task
- [`models/organism_det/results/reports/best_model_report.md`](models/organism_det/results/reports/best_model_report.md) — current best model and next steps

## Setup

```
pip install -r requirements.txt
```

## License

Proprietary — Copyright (c) 2026 Wiseyak Solutions Pvt Ltd. All rights reserved. See [`LICENSE`](LICENSE).
