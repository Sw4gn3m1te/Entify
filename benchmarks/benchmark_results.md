# Entify benchmark

Counting accuracy on a COCO val2017 subset, compared with the COCO ground truth.

| Setting | Value |
|---|---|
| Date | 2026-11-17 |
| Entify version | 0.4.0 |
| Model | `dfine_small_coco` |
| COCO subset seed | 0 |
| Images | 85 (COCO val2017, chosen by `scripts/fetch_coco_subset.py`) |
| Ground-truth objects | 933 |
| Overlap (IoU) | 0.5 |
| Tiling | off |
| Sharpening | off |
| Runtime | 40 s |

## Counting error by confidence threshold

| threshold | predicted (true: 933) | mean absolute error per image | mean error per image | pearson r |
|---|---|---|---|---|
| 0.30 | 1478 | 8.58 | +6.41 | 0.829 |
| 0.40 | 1021 | 5.11 | +1.04 | 0.830 |
| 0.50 | 792 | 4.34 | -1.66 | 0.817 |
| 0.60 | 632 | 4.62 | -3.54 | 0.807 |
| 0.70 | 505 | 5.48 | -5.04 | 0.748 |
| 0.80 | 367 | 6.85 | -6.66 | 0.628 |

Lowest counting error at threshold **0.50**.

## Largest errors at threshold 0.50

| category | true | predicted | mean absolute error | mean error |
|---|---|---|---|---|
| person | 309 | 276 | 0.74 | -0.39 |
| chair | 61 | 42 | 0.29 | -0.22 |
| book | 17 | 13 | 0.24 | -0.05 |
| handbag | 23 | 11 | 0.21 | -0.14 |
| cup | 34 | 34 | 0.19 | +0.00 |

## Reproduce

```bash
python benchmarks/run_benchmark.py --seed 0
```
