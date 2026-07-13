# Best Model Report — YOLO26n v2

**Author:** Pritam Thapa | AI Researcher | AISCS

**Status:** current best model, single checkpoint, production candidate
**Checkpoint:** `models/organism_det/best_model/organism-detector-yolo26n-v2.pt`
**Metrics source:** `models/organism_det/results/metrics/yolo26n_v2_metrics.csv`
**Training notebook:** `models/organism_det/notebooks/04_train_yolo26n_v2.ipynb`

## Summary

Of every model trained so far — v1 YOLO26n, v1 YOLO26s, v2 YOLO26n, and a 3-fold-validated v3 recipe attempt — YOLO26n v2 is the best single model, and by a wide margin on overall mAP. It is also the model this project is currently pointed at for production. The two-model class-routed ensemble explored in notebook 06 was explicitly rejected in favour of a single model, and the v3 recipe changes tested on top of v2 (notebook 07) made results worse rather than better, as detailed in "What was already tried and didn't work" below. No experiment run to date has beaten this model.

## Results

| Split | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| Val | 0.6402 | 0.4076 | 0.5664 | 0.6432 |
| Test | 0.6365 | 0.3734 | 0.6048 | 0.6210 |

**Per-class AP@50 (test):**

| Class | AP@50 | Notes |
|---|---|---|
| Trichomonas vaginalis (TV) | 0.4430 | Weakest class, and the most clinically critical — see below |
| Bacterial vaginosis flora shift (BV) | 0.7274 | Strongest and most stable class |
| Candida spp. | 0.3808 | Weak, likely due to high size variance in the source data |
| Actinomyces spp. | 0.9950 | Near-saturated — see "Why this run wins" below |

**Comparison against every other completed run (test set):**

| Run | mAP@50 | TV | BV | Candida | Actinomyces |
|---|---|---|---|---|---|
| v1 YOLO26n | 0.5192 | 0.467 | 0.677 | 0.321 | 0.612 |
| v1 YOLO26s | 0.5301 | 0.510 | 0.684 | 0.373 | 0.553 |
| v2 YOLO26n | 0.6365 | 0.443 | 0.727 | 0.381 | 0.995 |
| v3 recipe (3-fold mean, YOLO26n proxy) | 0.478 ± 0.139 | 0.423 ± 0.081 | 0.645 ± 0.101 | 0.109 ± 0.081 | 0.735 ± 0.301 |

The v2 YOLO26s run was never completed — notebook 05 stalled before producing a metrics file. Finishing it is one of the suggestions below.

## Recipe

The base architecture is YOLO26n (2.4M params, 5.4B FLOPs), trained from COCO-pretrained weights, using STAL for small-object label maintenance, an NMS-free one-to-one head, the MuSGD optimiser, and progressive loss.

| Parameter | v1 | v2 (this model) | Rationale |
|---|---|---|---|
| `copy_paste` | 0.3 | 0.6 | More rare-class (Actinomyces, TV) instances pasted per image |
| `mixup` | 0.0 | 0.15 | Class-boundary regularisation on a small dataset |
| `multi_scale` | 0.0 | 0.5 | Training at ±50% image size, helps with Candida's size variance |
| `cos_lr` | False | True | Avoids plateau-triggered early stopping |
| `lrf` | 0.01 | 0.001 | Lower learning-rate floor for a smoother cosine tail |
| `epochs` | 150 | 200 | Longer window for the cosine schedule to use |
| `patience` | 30 | 50 | Matches the longer schedule |
| `imgsz` | 1280 | 1280 | Required — TV's ~32×37px boxes need full resolution |
| `batch` | 8 | 8 | ~4GB VRAM budget |

Unchanged augmentations: `flipud=0.5`, `fliplr=0.5`, `degrees=15`, `hsv_h=0.05`, `hsv_s=0.3`, `hsv_v=0.1`, `scale=0.5`.

The model was trained and evaluated on the original fixed split of 378 train, 47 validation, and 48 test images (473 total). Class instance counts are approximately BV 969, TV 377, Candida 259, and Actinomyces 42 — an imbalance ratio of roughly 23:9:6:1.

## Why this run wins

This model actually generalises, rather than partially memorising the validation set. The gap between validation and test mAP@50 is only -0.0037 (0.6402 to 0.6365), compared with a drop of roughly 10 to 12 points for both v1 YOLO26n (-0.0989) and v1 YOLO26s (-0.1161). Both v1 models showed signs of overfitting to the small, 47-image validation split; v2's gap is close to what noise alone would produce.

The main driver is `copy_paste=0.6`, which fixed Actinomyces' catastrophic overfitting. With only 42 real training instances, v1 had memorised them outright — validation AP of 0.95 to 0.97 collapsed to a test AP of 0.55 to 0.61, a drop of about 0.40. Pasting more rare-class instances into training images brought Actinomyces' test AP up to 0.995, with no meaningful validation-to-test gap remaining.

Candida and BV also improved over both v1 variants, plausibly because `multi_scale=0.5` directly addresses Candida's documented high size variance.

## Known weaknesses

Trichomonas vaginalis is this model's worst class, and it happens to be the clinically most critical one. v1 YOLO26s actually scored higher on TV alone (0.510 versus v2n's 0.443). This specific gap is what motivated both the two-model ensemble, which was later rejected, and the v3 recipe experiment, neither of which has produced a working fix yet.

Candida remains weak in absolute terms at 0.381 AP@50, even though it's this model's second-best improvement over v1 — no run to date has pushed Candida much above 0.38.

No per-class deployment thresholds have been tuned for this model specifically. The threshold-tuning work in notebook 06 was built for the two-model ensemble and does not apply to this single checkpoint.

The model has only been validated on a single train/validation/test split. Every number above comes from one 47-image validation and 48-image test partition, so there is no cross-validated confidence interval on any of this model's own results — the k-fold harness built in notebook 07 was pointed at the v3 recipe, not at re-validating v2's own recipe.

## What was already tried and didn't work

The two-model class-routed ensemble routed v1s for TV and v2n for BV, Candida, and Actinomyces, with per-class recall-biased thresholds. It was rejected on product grounds: the requirement is a single model, not two models running per image.

The v3 recipe combined `cls_pw=0.35` class-loss weighting, a `multi_scale` cap of 0.25, and a `freeze=10` backbone warm-start. Validated through 3-fold stratified cross-validation on a YOLO26n proxy, it came out worse than v2n on every metric. Candida collapsed in all three folds (0.10, 0.19, and 0.03 AP@50, against v2n's 0.38) — a consistent pattern across folds, and therefore a real effect, most likely caused by the `multi_scale` cap removing the scale diversity that had been helping Candida. TV did not improve, despite being the intended target of the fix. One fold also showed the freeze/unfreeze warm-start actively hurting training: its best checkpoint occurred before the backbone had even unfrozen. Full detail is in `models/organism_det/results/metrics/kfold_v3_recipe_validation.csv`.

---

# What to try next

The following are ranked by expected value for effort. Items 1 to 3 build directly on the v3 finding above rather than discarding it wholesale — the three changes were tested together, not in isolation, so it is not yet known whether `cls_pw` or the freeze warm-start would help on their own.

1. **Re-test `multi_scale` at a looser cap, around 0.35 to 0.4, in isolation.** The v3 run bundled three changes together, and the Candida collapse most implicates the multi-scale cap specifically. Testing it alone against v2n's exact recipe — keeping `copy_paste=0.6` and `mixup=0.15`, dropping `cls_pw` and `freeze` for this run — would show whether a gentler cap can still help TV without sacrificing Candida.

2. **Test `cls_pw` in isolation**, on top of v2n's proven recipe, without the multi-scale change or the freeze warm-start. It has been mechanically verified to work correctly — a live smoke test computed weights of TV 0.79, BV 0.56, Candida 0.92, and Actinomyces 1.73, mean-normalised with the rarest class weighted highest — but its actual effect on model quality has never been isolated from the other two changes.

3. **Re-examine the freeze warm-start.** Fold 2 of the v3 run had its best checkpoint appear before the backbone unfroze at epoch 10, and results got worse afterward. This is weak evidence that the warm-start destabilises training at this data scale rather than helping, and is worth a direct comparison — the same recipe with the freeze on versus off — before using it again.

4. **Finish the stalled v2 YOLO26s run.** Notebook 05 was started but never completed, and no metrics file exists for it. The v2 recipe lifted YOLO26n by 12 points of mAP@50 over v1n; whether the same recipe lifts YOLO26s over v1s by a similar margin is untested, which matters because `yolo26_model_review.md` originally expected YOLO26s to be the production pick over the nano variant.

5. **Validate v2n's own recipe with the k-fold harness**, not only the v3 recipe. All of v2n's headline numbers, including the near-perfect Actinomyces score, come from a single 47/48 image validation/test split. The k-fold infrastructure already exists in `07_kfold_recipe_validation.ipynb`; pointing it at v2's actual recipe — without `cls_pw`, the capped `multi_scale`, or `freeze` — would put a real confidence interval on the numbers this report is based on.

6. **Tune per-class confidence thresholds for this specific checkpoint**, using the same recall-biased clinical rationale notebook 06 applied — a missed organism is worse than a false flag — but applied to this one model rather than a routed ensemble. This is needed before any deployment, regardless of what else changes.

7. **Add SAHI-tiled training crops for TV**, a recommendation from the original `training_report.md` that has still not been implemented. Extracting 640×640 crops centred on TV annotations would create more TV-positive training signal from the same images. This is orthogonal to everything above, and the most direct way to address TV without touching any multi-scale or loss-weighting tradeoffs.

8. **Collect more real Actinomyces and Candida data, if feasible.** `copy_paste` has already done most of the available work for Actinomyces from its 42 real instances. Candida has never exceeded roughly 0.38 AP@50 under any recipe tried so far, and its high size variance appears to be a genuine data property that no augmentation technique has fully solved. More real instances would be the highest-confidence fix; everything else in this list is a mitigation around the data limitation, not a substitute for it.

9. **Try an architecture change: `yolo26n-p2.yaml` or `yolo26s-p2.yaml`**, which add a stride-4 detection head specifically for very small objects, shipped as an architecture-only configuration in Ultralytics. This is worth trying only after the lower-effort options above are exhausted, since it requires training from scratch rather than loading the COCO checkpoint directly — but it is the most direct architectural lever available for TV specifically, as opposed to the recipe-level levers everything else here relies on.
