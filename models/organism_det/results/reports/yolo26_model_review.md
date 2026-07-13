# Why YOLO26 for Organism Detection

**Author:** Pritam Thapa | AI Researcher | AISCS

## The Problem

The task is to detect four organism types in 1280×720 LBC cytology patches:

- **Trichomonas vaginalis** — median size ~32×37 px, tiny and by far the hardest class to detect.
- **Bacterial vaginosis flora shift** — median size ~198×199 px.
- **Candida spp.** — median size ~180×180 px, with high size variance.
- **Actinomyces spp.** — median size ~71×71 px, rare, with only 42 training instances.

The training set has 378 images, and the intended deployment target is a real-time patch pipeline.

---

## Why YOLO26

### 1. STAL — built for small objects

YOLO26 introduces Small object label maintenance (STAL), which adjusts label assignment during training to preserve positive coverage when ground-truth boxes are too small for standard anchor matching. Trichomonas vaginalis at 32×37 px is exactly the failure mode this addresses. Neither YOLO11 nor YOLO12 has an equivalent mechanism.

### 2. NMS-free inference

YOLO26 uses a one-to-one detection head by default, which removes non-maximum suppression from the inference pipeline entirely. For a clinical patch pipeline, this brings a few practical benefits: there is no NMS threshold to tune per organism type, deployment is simpler since a single confidence threshold is all that's needed, and it matters in particular when multiple TV organisms appear close together on a patch, where NMS thresholds can otherwise suppress genuine detections.

### 3. MuSGD optimiser

The MuSGD optimiser, a hybrid of SGD and Muon, gives better convergence on small datasets. With only 378 training images, the quality of the optimiser matters more than it would on a larger dataset, where sheer data diversity can compensate for a weaker optimisation strategy.

### 4. Progressive loss

Training loss progressively shifts emphasis toward the inference-time architecture as training proceeds. This closes the gap between training and inference behaviour that otherwise makes validation mAP look better than real-world performance.

---

## YOLO26 versus the alternatives

| | YOLO11 | YOLO12 | YOLO26 |
|---|---|---|---|
| Small object mechanism | P2 head | Area attention | STAL |
| NMS at inference | Yes | Yes | No |
| Optimiser | SGD/Adam | SGD/Adam | MuSGD |
| COCO mAP (nano) | 37.3 | 40.6 | 40.9 |
| COCO mAP (small) | 48.0 | 48.4 | 48.6 |
| COCO mAP (medium) | 51.5 | 52.5 | 53.1 |

YOLO26 is ahead at every scale on COCO. The nano-scale improvement (+3.6 over YOLO11n) is the largest relative gain of the three, which matters here since the nano variant is the speed fallback for this project.

---

## Models under benchmark

| Model | Params | COCO mAP | T4 latency | Role |
|---|---|---|---|---|
| YOLO26n | 2.4M | 40.9 | 1.7ms | Speed baseline |
| YOLO26s | 9.5M | 48.6 | 2.5ms | Primary / expected winner |
| YOLO26m | 20.4M | 53.1 | 4.7ms | Accuracy ceiling |

All three are trained at 1280px input resolution from COCO-pretrained weights, on an RTX 4090 within an 8GB VRAM budget.

---

## Decision criteria for production

The default choice is YOLO26s, with two exceptions: deploy YOLO26m instead if YOLO26s's TV recall falls more than three points below YOLO26m's, and deploy YOLO26n instead if the latency budget is under 2ms, accepting the lower TV recall that comes with it.

TV recall per class is the primary clinical metric for this decision, not overall mAP.
