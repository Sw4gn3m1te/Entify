# Model comparison: person

How well each model finds person objects on images none of them was trained or checked on, compared with the original COCO annotations.

| Setting | Value |
|---|---|
| Date | 2026-11-18 |
| Entify version | 0.4.7 |
| Category | person |
| Ground truth | original COCO val2017 annotations, 711 person boxes |
| Images | 200 from COCO val2017 without crowd regions (seed 0): 50 without person, 50 with 1-2, 50 with 3-6, 50 with 7 or more person objects |
| Left out | 85 from `data/coco_subset`; 85 from `benchmarks/data/coco_subset` |
| Match | same category and IoU >= 0.5, each true box at most once |
| Overlap (IoU) for duplicates | 0.5 |
| Tiling | off |
| Sharpening | off |

## Summary

| Model | AP50 | Best F1 | at threshold | Precision | Recall | Counting MAE | Seconds per image |
|---|---|---|---|---|---|---|---|
| `dfine_small_coco` | 0.840 | 0.814 | 0.60 | 0.933 | 0.722 | 0.92 | 0.92 |

## By confidence threshold

| Threshold | Model | Predicted (true: 711) | Correct | Precision | Recall | F1 | Counting MAE | Counting bias |
|---|---|---|---|---|---|---|---|---|
| 0.30 | `dfine_small_coco` | 1002 | 613 | 0.612 | 0.862 | 0.716 | 1.70 | +1.46 |
| 0.40 | `dfine_small_coco` | 773 | 586 | 0.758 | 0.824 | 0.790 | 0.82 | +0.31 |
| 0.50 | `dfine_small_coco` | 639 | 548 | 0.858 | 0.771 | 0.812 | 0.77 | -0.36 |
| 0.60 | `dfine_small_coco` | 550 | 513 | 0.933 | 0.722 | 0.814 | 0.92 | -0.81 |
| 0.70 | `dfine_small_coco` | 456 | 438 | 0.961 | 0.616 | 0.751 | 1.27 | -1.27 |
| 0.80 | `dfine_small_coco` | 358 | 349 | 0.975 | 0.491 | 0.653 | 1.76 | -1.76 |

## How to read this

- Precision: share of the predicted person boxes that match a true one. Recall: share of the true boxes that were found. F1 combines both. Higher is better.
- AP50: average precision over all thresholds from 0.05 up, at IoU 0.5 (COCO's 101-point method), one number for the whole precision-recall curve. Higher is better.
- Counting MAE: mean absolute difference between predicted and true count per image, lower is better. Counting bias: positive means too many, negative too few.

## Reproduce

```bash
python benchmarks/compare_models.py -m dfine_small_coco
```

Images and annotations: `benchmarks/data/person_holdout` (not in the repository).
