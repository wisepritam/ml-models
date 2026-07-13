# YOLO26 Organism Detection — Training Analysis Report

**Author:** Pritam Thapa | AI Researcher | AISCS
**Date:** 2026-06-05
**Models evaluated:** YOLO26n (92 epochs), YOLO26s (118 epochs)
**Dataset:** 378 train / 47 val / 48 test images, 4 classes, 1280px input

> v2 runs are queued. See Section 9 for the v2 hyperparameter changes and results.
> Notebooks: `04_train_yolo26n_v2.ipynb`, `05_train_yolo26s_v2.ipynb`

---

## 1. Overall Performance Summary

### Test set results

| Model | mAP@50 | mAP@50-95 | Precision | Recall | Params | Stopped at |
|---|---|---|---|---|---|---|
| YOLO26n | 0.5192 | 0.2884 | 0.4963 | 0.5700 | 2.4M | epoch 92 (patience 30) |
| YOLO26s | 0.5301 | 0.2876 | 0.5269 | 0.5928 | 9.5M | epoch 118 (patience 30) |

YOLO26s leads on mAP@50 and recall, while YOLO26n edges ahead on mAP@50-95 by a small margin (0.2884 vs 0.2876). The gap between the two models is narrow — about 1.1 percentage points on mAP@50 — which is a modest return for four times the parameter count. This points to the small dataset as the limiting factor, rather than model capacity.

---

## 2. Per-Class AP@50 Breakdown

| Class | YOLO26n (val) | YOLO26n (test) | YOLO26s (val) | YOLO26s (test) | Notes |
|---|---|---|---|---|---|
| Trichomonas vaginalis (TV) | 0.609 | 0.467 | 0.581 | 0.510 | Tiny objects, ~32×37px — the hardest class |
| Bacterial vaginosis (BV) | 0.714 | 0.677 | 0.704 | 0.684 | Largest objects, most stable |
| Candida spp. | 0.195 | 0.321 | 0.328 | 0.373 | High size variance, consistently weak |
| Actinomyces spp. | 0.955 | 0.612 | 0.972 | 0.553 | Only 42 training instances — overfitting |

### Key findings per class

**Actinomyces spp. — severe overfitting.** Validation AP sits at 0.95–0.97 but drops to 0.55–0.61 on the test set, a fall of roughly 0.40. With only 42 training instances, the model has effectively memorised them rather than learned a generalisable pattern. This is the most urgent issue to address.

**Trichomonas vaginalis — a real but expected val/test gap.** Validation AP of 0.58–0.61 drops to 0.47–0.51 on test, a decline of about 0.10. Some drop is expected for objects this small, but it remains a concern. STAL is helping the model find these objects at all, but the amount of training data is the binding constraint.

**Candida spp. — weak, though the larger model helps.** YOLO26s improves on YOLO26n by 0.052 AP@50 on the test set, suggesting additional capacity does help here. The class's high size variance is a likely source of the remaining inconsistency.

**Bacterial vaginosis — the most reliable class.** AP@50 holds steady around 0.68–0.71 across both models, helped by larger objects and more consistent annotations.

---

## 3. Training Dynamics

### YOLO26n (92 epochs)

Best validation mAP@50 reached about 0.639 at epoch 61. Training was noisy through the first 30 epochs, then improved steadily from 0.52 to 0.63 between epochs 30 and 70, before plateauing and triggering early stopping around epoch 92. The training loss declined in a healthy pattern (classification loss from 9.2 to 0.92, box loss from 1.55 to 1.25). Validation loss, however, oscillated without a clear downward trend after epoch 60, a sign of mild overfitting.

### YOLO26s (118 epochs)

Best validation mAP@50 reached about 0.646 at epoch 88. A spike in validation classification loss around epoch 18 (up to 7.5) was most likely caused by a difficult batch. Convergence stayed noisy through epochs 26–90, with several smaller spikes, and progress slowed considerably from epoch 90 onward. Despite running 26 epochs longer than YOLO26n, the improvement in validation mAP was only 0.013 — evidence that the model's extra capacity is not being put to use, most likely because of the limited amount of training data.

### Validation-to-test gap

Both models show a consistent drop of roughly 10–15% in mAP@50 between validation and test. Three factors likely contribute: the 47-image validation set may not fully represent the test distribution; the models show some overfitting to the validation distribution during training; and 378 training images is a genuinely small sample for four-class detection at 1280px resolution.

---

## 4. What Is Working

- STAL is helping TV detection: 32-pixel objects are being found at all, which is not guaranteed at this scale.
- The MuSGD optimiser is converging stably despite the small dataset.
- The NMS-free detection head simplifies deployment.
- BV detection is solid and could be deployed as-is.
- Both models reached a reasonable level of convergence within their training budgets.

---

## 5. Improvement Recommendations

Ranked by expected impact for this dataset and task.

### Priority 1 — Data (highest impact)

**A. Collect more Actinomyces samples.** The current 42 training instances should be expanded to at least 200. The collapse from 0.97 validation AP to 0.55 test AP is memorisation, plain and simple — no augmentation technique will substitute for more real data here.

**B. Tile-based training for Trichomonas.** The current approach trains on full 1280px images containing 32px TV objects. Extracting 640×640 crops centred on TV annotations, along with hard negative patches, would generate more TV-positive training samples from the images already available. SAHI (Slicing Aided Hyper Inference) is a reasonable tool for this:

```python
pip install sahi
```

`sahi.slicing.slice_coco` can build a tiled dataset to train alongside the existing full-image set.

**C. Tune mosaic and mixup.** The current configuration uses `mosaic=1.0`, `mixup=0.0`, and `copy_paste=0.3`. Raising `copy_paste` to 0.5–0.7 would paste more rare-class objects into training images, and enabling `mixup=0.15` should help with class-boundary learning on a dataset this small.

### Priority 2 — Training improvements

**D. Class-weighted loss.** Actinomyces and Candida are underrepresented and would benefit from an upweighted class loss — either through per-class weights on top of the current `cls: 0.5` gain, or a focal loss with a higher gamma for rare classes.

**E. Extend training with a cosine learning-rate schedule.** Both models stopped at `patience=30` while the learning rate was still meaningful. Switching to `cos_lr: true`, extending `patience` to 50 or `epochs` to 200, and lowering `lrf` to 0.001 should give the models more room to escape local minima and converge more smoothly.

**F. Multi-scale training.** Currently disabled (`multi_scale: 0.0`). Enabling `multi_scale: 0.5` (training at plus or minus 50% of the base image size) should help with Candida's size variance.

**G. Freeze the backbone for early epochs.** All layers currently train from epoch 1. Freezing the first 10 backbone layers for the first 10 epochs, then resuming without the freeze, would let the head adapt while preserving the pretrained features underneath.

### Priority 3 — Inference and evaluation

**H. Use SAHI for inference on TV.** Sliding-window inference at test time can meaningfully improve small-object recall without any retraining:

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=640, slice_width=640,
    overlap_height_ratio=0.2, overlap_width_ratio=0.2
)
```

**I. Per-class confidence thresholds.** A single global threshold is currently used across all classes. TV would likely benefit from a lower threshold (0.25–0.35) to maximise recall, while Actinomyces could use a higher threshold (0.50+) to reduce the false positives introduced by its overfitting. Thresholds should be tuned separately per class on the validation set.

**J. Test-time augmentation.** Ultralytics supports this natively (`model.val(data='configs/dataset.yaml', augment=True)`) and typically yields a 1–3% mAP improvement with no retraining cost.

---

## 6. Recommended Next Steps

| Step | Action | Effort | Expected gain | Status |
|---|---|---|---|---|
| 1 | Collect 150+ more Actinomyces images | High | +0.15–0.25 AP (Actinomyces) | Pending |
| 2 | Enable `copy_paste=0.6` and `mixup=0.15`, retrain | Low | +0.03–0.06 mAP overall | Done — v2 |
| 3 | Enable `cos_lr=true`, extend to 200 epochs | Low | +0.02–0.04 mAP | Done — v2 |
| 4 | SAHI inference for TV at test time | Low | +0.05–0.10 AP (TV) | Pending |
| 5 | Per-class confidence threshold tuning | Low | +0.02–0.04 clinical recall | Pending |
| 6 | Tile-based training for TV | Medium | +0.05–0.10 AP (TV) | Pending |
| 7 | Train YOLO26m once data is expanded | Medium | +0.03–0.06 overall | Pending |

---

## 7. Should YOLO26m Be Trained Next?

Not yet. With only 378 training images, the improvement from YOLO26n to YOLO26s was just 1.1 percentage points on mAP@50, and YOLO26m would likely show a similarly marginal gain over YOLO26s. Actinomyces would also be expected to overfit further, since more parameters generally means more capacity to memorise a small class. YOLO26m is worth training once the dataset — particularly Actinomyces — has been expanded and the augmentation changes above have been tested.

---

## 8. Summary

Both models are functional but limited by data volume. The two biggest bottlenecks are Actinomyces, with only 42 training instances, and Candida, whose high size variance makes it hard to learn consistently. YOLO26s is the marginally stronger candidate on recall, though the practical difference from YOLO26n is small. The most promising near-term improvements — SAHI inference, a higher copy-paste rate, and a cosine learning-rate schedule — are all low-effort changes that do not require new data.

---

## 9. Version 2 — Hyperparameter Changes

**Notebooks:** `04_train_yolo26n_v2.ipynb`, `05_train_yolo26s_v2.ipynb`
**Checkpoints:** `checkpoints/yolo26n_v2/`, `checkpoints/yolo26s_v2/`
**Metrics:** `results/metrics/yolo26n_v2_metrics.csv`, `results/metrics/yolo26s_v2_metrics.csv`
**Plots:** `results/plots/yolo26n_v2/`, `results/plots/yolo26s_v2/`

### Parameter changes (v1 to v2)

| Parameter | v1 | v2 | Addresses |
|---|---|---|---|
| `copy_paste` | 0.3 | 0.6 | Priority 1C — more Actinomyces and TV instances pasted per image |
| `mixup` | 0.0 | 0.15 | Priority 1C — class-boundary regularisation |
| `multi_scale` | 0.0 | 0.5 | Priority 2F — plus or minus 50% image size per step, helps Candida's size variance |
| `cos_lr` | False | True | Priority 2E — cosine schedule avoids plateau, smoother long-run convergence |
| `lrf` | 0.01 | 0.001 | Priority 2E — lower learning-rate floor for the cosine tail |
| `epochs` | 150 | 200 | Priority 2E — longer window for the cosine schedule to use |
| `patience` | 30 | 50 | Priority 2E — avoids a hair-trigger early stop under the cosine schedule |

### Expected v2 improvement (estimated before training)

| Change | Expected gain |
|---|---|
| copy_paste + mixup | +0.03–0.06 mAP@50 overall |
| cos_lr + extended epochs | +0.02–0.04 mAP@50 |
| multi_scale | +0.03–0.05 AP (Candida, from reduced size-variance sensitivity) |
| Combined | +0.05–0.10 mAP@50 overall |

### v2 test results

| Model | mAP@50 | mAP@50-95 | Precision | Recall | Stopped at |
|---|---|---|---|---|---|
| YOLO26n v2 | 0.6365 | 0.3734 | 0.6048 | 0.6210 | epoch 200 (batch size auto-reduced from 8 to 4 partway through — see note below) |
| YOLO26s v2 | — | — | — | — | Not completed |

| Class | v1 YOLO26n (test) | YOLO26n v2 (test) | Change | AP50-95 (v2) | AP50-to-AP50-95 gap |
|---|---|---|---|---|---|
| Trichomonas vaginalis | 0.4672 | 0.4430 | -0.0242 | 0.2294 | 0.2136 |
| Bacterial vaginosis | 0.6771 | 0.7274 | +0.0503 | 0.5040 | 0.2233 |
| Candida spp. | 0.3209 | 0.3808 | +0.0599 | 0.1542 | 0.2266 |
| Actinomyces spp. | 0.6117 | 0.9950 | +0.3833 | 0.6058 | 0.3892 |

### Analysis

`copy_paste=0.6` effectively resolved the Actinomyces overfitting problem: test AP@50 rose from 0.61 to 0.995, closing the earlier validation-to-test gap almost entirely. Its AP50-to-AP50-95 gap (0.389) is now the largest of any class, meaning recall is excellent but box localisation is still comparatively loose — expected, given that only 42 source instances are available to learn precise box regression from.

BV and Candida both improved as well (+0.05 and +0.06 respectively), plausibly because `multi_scale=0.5` is helping with Candida's known size variance.

TV regressed slightly (-0.024), which matters because it is the clinically critical class. The likely cause, confirmed by reading `ultralytics/models/yolo/detect/train.py`, is that `multi_scale=0.5` trains across a resolution range from `imgsz*(1-0.5)=640px` up to roughly 1952px. At the low end of that range, TV's native ~32×37px objects shrink to about 16×18px — likely below a reliably detectable size even with STAL — which dilutes TV's gradient signal across the epochs spent at low resolution.

Candida's AP50-to-AP50-95 gap (0.227) is not elevated relative to TV (0.214) or BV (0.223), which indicates its weakness is a recall problem — AP50 itself is low — rather than a bounding-box precision problem. This argues against oriented or rotated boxes as a fix for now; the filamentous shape of Candida does not yet show up as a localisation-precision bottleneck.

### Follow-ups to consider for v3

1. Cap the `multi_scale` downscale floor — for example 0.25 instead of 0.5, giving a minimum size of about 960px — to stop TV from training at sub-detectable resolutions, while retaining some scale diversity for Candida.
2. A manual multi-scale ensemble at inference time (not training time) for Candida: running `predict()` at, for example, 960, 1280, and 1600px and merging the results with NMS or weighted box fusion. Note that Ultralytics' built-in `augment=True` test-time augmentation is a no-op for YOLO26, confirmed via `model.model.end2end == True`, which causes `_predict_augment()` to log a warning and silently fall back to single-scale inference. Any multi-scale inference experiment needs to be done manually rather than through the `augment` flag.
3. SAHI tiled inference, reserved for TV specifically, where objects are genuinely lost to downsampling. Not recommended for Candida, where tiling risks cutting elongated filaments across tile boundaries.
4. Per-class confidence threshold retuning is now more urgent than before: Actinomyces is near-saturated while TV and Candida remain weak, so a single global threshold is even more miscalibrated than it was in v1.

Note on Priority 2G (backbone freeze): this was not included in the v2 parameter update. A freeze-then-unfreeze schedule across epochs 1–10 requires an `on_train_epoch_start` callback to restore `requires_grad` on the frozen layers partway through training. This was scheduled for v3, once the v2 results were in.
