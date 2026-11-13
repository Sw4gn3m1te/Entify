import pandas as pd


def count_by_category(results, class_names):
    rows = [
        (image_name, class_names[label])
        for image_name, detections in results.items()
        for label in detections.labels
    ]
    detections = pd.DataFrame(rows, columns=["image", "category"])
    counts = detections.value_counts().rename("count").reset_index()
    return counts.sort_values(["image", "count"], ascending=[True, False], ignore_index=True)


def totals(counts):
    return counts.groupby("category")["count"].sum().sort_values(ascending=False)
