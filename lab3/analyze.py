import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch import nn
from torchvision.models import resnet18

from common import (
    CLASSES,
    FIGURES_DIR,
    MODEL_PATH,
    RUNS,
    RUNS_DIR,
    eval_loader,
    image_path,
    labels_of,
    macro_f1,
    predict_logits,
    read_split,
    rows_of,
    run_name,
    select_run,
    split_sha256,
)


def load_logs() -> dict[tuple[str, int], dict[str, Any]]:
    logs = {}

    for regime, seed in RUNS:
        path = RUNS_DIR / run_name(regime, seed) / "run.json"
        logs[(regime, seed)] = json.loads(path.read_text())

    return logs


def check_split_hash(logs) -> None:
    current = split_sha256()

    for log in logs.values():
        if log["split_sha256"] != current:
            raise SystemExit("Хеш split.csv у логах не збігається з поточним. Перенавчіть усі сім запусків.")

    print(f"Хеш split.csv однаковий в усіх 7 логах: {current}")


def print_runs(logs) -> None:
    rows = []

    for (regime, seed), log in logs.items():
        rows.append({
            "regime": regime,
            "seed": seed,
            "train_loss": round(log["final_train_loss"], 4),
            "val_macro_f1": round(log["final_val_macro_f1"], 4),
            "train_seconds": round(log["train_seconds"], 1),
        })

    print("\nСім запусків (епоха 8):")
    print(pd.DataFrame(rows).to_string(index=False))


def plot_curves(logs) -> None:
    colors = {"scratch": "tab:gray", "frozen": "tab:blue", "unfreeze": "tab:orange"}
    styles = {1: "-", 2: "--", 3: ":"}

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4.5))

    for (regime, seed), log in logs.items():
        epochs = [h["epoch"] for h in log["history"]]
        losses = [h["train_loss"] for h in log["history"]]
        f1s = [h["val_macro_f1"] for h in log["history"]]

        style = {
            "color": colors[regime],
            "linestyle": styles[seed],
            "marker": "o",
            "ms": 3,
            "label": run_name(regime, seed),
        }

        ax_loss.plot(epochs, losses, **style)
        ax_f1.plot(epochs, f1s, **style)

    ax_loss.set(xlabel="Епоха", ylabel="Loss", title="Loss на train")
    ax_f1.set(xlabel="Епоха", ylabel="macro-F1", title="macro-F1 на validation")
    ax_f1.legend(fontsize=8)

    for ax in (ax_loss, ax_f1):
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "curves.png", dpi=120)
    plt.close(fig)


def print_effects(final_f1) -> None:
    f_scratch = final_f1[("scratch", 1)]

    mean = {}
    spread = {}

    for regime in ["frozen", "unfreeze"]:
        values = [final_f1[(regime, seed)] for seed in [1, 2, 3]]
        mean[regime] = np.mean(values)
        spread[regime] = max(values) - min(values)

    noise = max(spread.values())

    delta_pre = mean["frozen"] - f_scratch
    delta_unf = mean["unfreeze"] - mean["frozen"]

    print(f"\nF_scratch,1 = {f_scratch:.4f}")

    for regime in ["frozen", "unfreeze"]:
        print(f"{regime}: середнє {mean[regime]:.4f}, розмах {spread[regime]:.4f}")

    print(f"R = {noise:.4f}")

    for name, delta in [("delta_pre", delta_pre), ("delta_unf", delta_unf)]:
        if abs(delta) > noise:
            verdict = "вартий перевірки"
        else:
            verdict = "не відрізняється від розкиду"

        print(f"{name} = {delta:+.4f}: {verdict}")


def load_selected_model(final_f1) -> dict[str, Any]:
    selected = select_run(final_f1)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

    if (checkpoint["regime"], checkpoint["seed"]) != selected:
        raise SystemExit(f"lab3_model.pt не відповідає правилу вибору: очікувався {run_name(*selected)}")

    print(f"\nВибрано {run_name(*selected)}, lab3_model.pt відповідає правилу вибору")

    return checkpoint


def test_predictions(checkpoint, data_dir, rows):
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])

    loader = eval_loader(data_dir, rows, "test")
    logits = predict_logits(model, loader, "cpu")
    y = labels_of(loader)

    return logits, y


def print_test_metrics(y, logits, val_f1) -> None:
    pred = logits.argmax(1)

    test_f1 = macro_f1(y, logits)
    p = float((pred == y).mean())
    n = len(y)
    se = math.sqrt(p * (1 - p) / n)

    print(f"\nTest: macro-F1 {test_f1:.4f} (validation {val_f1:.4f})")
    print(f"accuracy {p:.4f}, sqrt(p(1-p)/n) = {se:.4f}, n = {n}")


def plot_confusion(cm) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")

    for true_class in range(len(CLASSES)):
        for pred_class in range(len(CLASSES)):
            value = cm[true_class, pred_class]
            color = "white" if value > cm.max() / 2 else "black"
            ax.text(pred_class, true_class, value, ha="center", va="center", color=color)

    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="Передбачений клас", ylabel="Справжній клас", title="Confusion matrix на test")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def five_errors(test_rows, y, pred, cm) -> list[int]:
    pairs = []

    for true_class in range(len(CLASSES)):
        for pred_class in range(len(CLASSES)):
            if true_class != pred_class and cm[true_class, pred_class] > 0:
                pairs.append((true_class, pred_class))

    pairs.sort(key=lambda pair: (-cm[pair], pair))

    files = {}

    for true_class, pred_class in pairs:
        indices = np.flatnonzero((y == true_class) & (pred == pred_class))
        files[(true_class, pred_class)] = sorted(indices, key=lambda i: test_rows[i]["file"])

    chosen = []
    max_rank = max(len(indices) for indices in files.values())

    for rank in range(max_rank):
        for pair in pairs:
            if len(chosen) < 5 and rank < len(files[pair]):
                chosen.append(files[pair][rank])

    return chosen


def describe_errors(indices, test_rows, y, logits, cm, data_dir) -> list[dict[str, Any]]:
    pred = logits.argmax(1)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    errors = []

    for i in indices:
        errors.append({
            "file": test_rows[i]["file"],
            "true": CLASSES[y[i]],
            "pred": CLASSES[pred[i]],
            "prob": round(float(probs[i, pred[i]]), 4),
            "count": int(cm[y[i], pred[i]]),
            "path": image_path(data_dir, test_rows[i]),
        })

    return errors


def plot_errors(errors) -> None:
    fig, axes = plt.subplots(1, len(errors), figsize=(3.2 * len(errors), 3.8))

    for ax, error in zip(axes, errors):
        image = Image.open(error["path"]).convert("RGB")
        title = f"{error['file']}\n{error['true']} -> {error['pred']} (p={error['prob']:.2f})"

        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "errors.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="директорія з seg_train/ і seg_test/")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(exist_ok=True)

    logs = load_logs()
    check_split_hash(logs)

    print_runs(logs)
    plot_curves(logs)

    final_f1 = {}
    for key, log in logs.items():
        final_f1[key] = log["final_val_macro_f1"]

    print_effects(final_f1)

    checkpoint = load_selected_model(final_f1)
    selected = (checkpoint["regime"], checkpoint["seed"])

    rows = read_split()
    test_rows = rows_of(rows, "test")

    logits, y = test_predictions(checkpoint, args.data, rows)
    pred = logits.argmax(1)

    print_test_metrics(y, logits, final_f1[selected])

    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))

    print("\nConfusion matrix (рядки: справжній клас, стовпці: передбачений):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    plot_confusion(cm)

    indices = five_errors(test_rows, y, pred, cm)
    errors = describe_errors(indices, test_rows, y, logits, cm, args.data)

    print("\nП'ять характерних помилок:")
    print(pd.DataFrame(errors).drop(columns="path").to_string(index=False))
    plot_errors(errors)


if __name__ == "__main__":
    main()
