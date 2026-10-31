import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from lab2_starter import load_predictions, split_summary, write_json


def count_errors(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    predicted_positive = scores >= threshold
    actual_positive = labels == 1

    tp = int(np.count_nonzero(predicted_positive & actual_positive))
    fp = int(np.count_nonzero(predicted_positive & ~actual_positive))
    fn = int(np.count_nonzero(~predicted_positive & actual_positive))
    tn = int(np.count_nonzero(~predicted_positive & ~actual_positive))

    cost = 10 * fn + fp

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost}


def threshold_table(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order].astype(np.float64)
    sorted_labels = labels[order]

    group_ends = np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:])
    group_ends = np.append(group_ends, len(labels) - 1)

    thresholds = np.append(np.inf, sorted_scores[group_ends])

    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)
    cumulative_fp = np.cumsum(1 - sorted_labels, dtype=np.int64)
    tp = np.append(0, cumulative_tp[group_ends])
    fp = np.append(0, cumulative_fp[group_ends])

    positive_count = int(np.sum(labels))
    negative_count = len(labels) - positive_count

    fn = positive_count - tp
    tn = negative_count - fp

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "C": 10 * fn + fp,
        }
    )


def best_threshold(candidates: pd.DataFrame) -> float:
    minimum_cost = candidates["C"].min()
    best_candidates = candidates[candidates["C"] == minimum_cost]

    return float(best_candidates["threshold"].max())


def check_search(out: Path) -> None:
    cases = {
        "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
        "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
        "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 11),
    }

    for name, (values, labels) in cases.items():
        scores = np.array(values)
        labels = np.array(labels)

        fast = threshold_table(labels, scores)

        candidates = [float("inf")] + sorted(set(values), reverse=True)
        direct_rows = []

        for threshold in candidates:
            errors = count_errors(labels, scores, threshold)
            direct_rows.append({"threshold": threshold, **errors})

        direct = pd.DataFrame(direct_rows)
        pd.testing.assert_frame_equal(fast, direct)

        expected = min(direct_rows, key=lambda row: row["C"])
        threshold = best_threshold(fast)

        if threshold != expected["threshold"]:
            raise AssertionError(f"Датасет {name}: неправильний вибір серед рівних мінімумів")

        fast.to_csv(out / f"check_{name}.csv", index=False)

        print(f"Перевірка {name}: OK, t={threshold}, C={expected['C']}")


def plot_curves(labels: np.ndarray, scores: np.ndarray, out: Path) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)

    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title="Validation: precision-recall")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "pr_curve.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Validation: ROC")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=150)
    plt.close(fig)

    return auc


def error_examples(data: dict[str, np.ndarray], threshold: float) -> pd.DataFrame:
    labels = data["test_y"]
    scores = data["test_scores"]
    row_ids = data["test_row_ids"]

    predicted = scores >= threshold
    fp_indices = np.flatnonzero(predicted & (labels == 0))
    fn_indices = np.flatnonzero(~predicted & (labels == 1))

    frames = []

    for name, indices in [("FP", fp_indices), ("FN", fn_indices)]:
        row_order = np.argsort(row_ids[indices])
        indices = indices[row_order][:5]

        selected_scores = scores[indices].astype(np.float64)
        distances = np.abs(selected_scores - threshold)

        examples = pd.DataFrame(
            {
                "row_id": row_ids[indices],
                "error": name,
                "label": labels[indices],
                "score": selected_scores,
                "Amount": data["test_amount"][indices],
                "distance_to_threshold": distances,
            }
        )

        frames.append(examples)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saved",
        type=Path,
        default=Path("lab2/results"),
        help="Директорія прогнозів starter",
    )

    args = parser.parse_args()

    if not (args.saved / "predictions.npz").is_file():
        parser.error(f"Немає {args.saved / 'predictions.npz'}. Спочатку запустіть starter.")

    out = args.saved / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    check_search(out)

    data = load_predictions(args.saved)

    summary = split_summary(data)

    print("\nПоділ даних:")
    print(summary.to_string(index=False))

    labels = data["validation_y"]
    scores = data["validation_scores"]

    candidates = threshold_table(labels, scores)
    candidates.to_csv(out / "threshold_candidates.csv", index=False)

    threshold = best_threshold(candidates)

    rules = [
        ("default", 0.5),
        ("minimum_cost", threshold),
        ("all_negative", np.inf),
    ]
    comparison_rows = []

    for rule_name, rule_threshold in rules:
        errors = count_errors(labels, scores, rule_threshold)
        comparison_rows.append(
            {"rule": rule_name, "threshold": rule_threshold, **errors}
        )

    comparison = pd.DataFrame(comparison_rows)

    auc = plot_curves(labels, scores, out)

    print(f"\nValidation ROC-AUC: {auc:.6f}")
    print(comparison.to_string(index=False))

    test = count_errors(data["test_y"], data["test_scores"], threshold)

    precision_denominator = test["TP"] + test["FP"]
    precision = test["TP"] / precision_denominator if precision_denominator else None

    recall_denominator = test["TP"] + test["FN"]
    recall = test["TP"] / recall_denominator if recall_denominator else None

    test_metrics = {
        "threshold": "inf" if np.isinf(threshold) else threshold,
        **test,
        "precision": precision,
        "recall": recall,
    }
    write_json(out / "test_metrics.json", test_metrics)

    examples = error_examples(data, threshold)
    examples.to_csv(out / "test_errors.csv", index=False)

    print(f"\nTest: {test}, precision={precision}, recall={recall}")

    if precision is None or recall is None:
        print("None означає невизначену метрику: її знаменник дорівнює нулю.")

    print("\nПерші FP та FN за номером рядка CSV:")
    print(examples.to_string(index=False))

    print(f"\nРезультати аналізу: {out.resolve()}")


if __name__ == "__main__":
    main()
