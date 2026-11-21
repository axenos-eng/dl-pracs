import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import checks
from common import (
    CLASSES,
    EPOCHS,
    EVAL_BATCH,
    EVAL_TF,
    IMG_SIZE,
    MODEL_PATH,
    REGIMES,
    RUNS,
    RUNS_DIR,
    SPLIT_CSV,
    TRAIN_BATCH,
    TRAIN_TF,
    build_model,
    build_split,
    device_name,
    eval_loader,
    git_info,
    labels_of,
    macro_f1,
    pick_device,
    predict_logits,
    read_split,
    run_name,
    select_run,
    set_train_mode,
    split_sha256,
    train_epoch,
    train_loader,
    versions,
    write_split,
)


def run_settings(regime, seed, model, optimizer) -> dict[str, Any]:
    config = REGIMES[regime]

    optimizer_args = dict(optimizer.defaults)
    optimizer_args.pop("lr")

    return {
        "seed": seed,
        "regime": regime,
        "train_transforms": [str(t) for t in TRAIN_TF.transforms],
        "eval_transforms": [str(t) for t in EVAL_TF.transforms],
        "weights": "IMAGENET1K_V1" if config.pretrained else None,
        "fc": str(model.fc),
        "lr_groups": config.groups,
        "eval_modules": config.eval_modules,
        "batchnorm": {"momentum": model.bn1.momentum, "eps": model.bn1.eps},
        "optimizer": "AdamW",
        "optimizer_args": optimizer_args,
        "loss": "CrossEntropyLoss",
        "epochs": EPOCHS,
        "train_batch": TRAIN_BATCH,
        "eval_batch": EVAL_BATCH,
        "shuffle": True,
        "drop_last": False,
        "num_workers": 0,
    }


def train_run(regime, seed, data_dir, rows, device) -> dict[str, Any]:
    name = run_name(regime, seed)
    out_dir = RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, optimizer = build_model(regime, seed, device)

    loader = train_loader(data_dir, rows)
    val_loader = eval_loader(data_dir, rows, "validation")
    y_val = labels_of(val_loader)

    history = []
    train_seconds = 0.0

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()

        set_train_mode(model, regime)
        loss = train_epoch(model, optimizer, loader, device)

        train_seconds += time.perf_counter() - start

        val_logits = predict_logits(model, val_loader, device)
        f1 = macro_f1(y_val, val_logits)

        history.append({"epoch": epoch, "train_loss": loss, "val_macro_f1": f1})
        print(f"  {name} епоха {epoch}: train loss {loss:.4f}, val macro-F1 {f1:.4f}", flush=True)

        if epoch == EPOCHS:
            np.save(out_dir / "val_logits.npy", val_logits)

    log = {
        "run": name,
        "settings": run_settings(regime, seed, model, optimizer),
        "git": git_info(),
        "split_sha256": split_sha256(),
        "versions": versions(),
        "device": device_name(device),
        "history": history,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_macro_f1": history[-1]["val_macro_f1"],
        "train_seconds": train_seconds,
    }

    (out_dir / "run.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str))

    weights = {key: tensor.cpu() for key, tensor in model.state_dict().items()}
    torch.save(weights, out_dir / "model.pt")

    return log


def save_selected_model(final_f1) -> None:
    regime, seed = select_run(final_f1)

    weights = torch.load(RUNS_DIR / run_name(regime, seed) / "model.pt", weights_only=True)

    torch.save(
        {
            "state_dict": weights,
            "regime": regime,
            "seed": seed,
            "classes": CLASSES,
            "img_size": IMG_SIZE,
        },
        MODEL_PATH,
    )

    print(f"\nВибрано {run_name(regime, seed)} (val macro-F1 {final_f1[(regime, seed)]:.4f}) -> {MODEL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="директорія з seg_train/ і seg_test/")
    args = parser.parse_args()

    device = pick_device()

    write_split(build_split(args.data))
    rows = read_split()
    print(f"split.csv: {SPLIT_CSV}, sha256 {split_sha256()}")

    if not checks.run_all(args.data, rows, device):
        raise SystemExit("Перевірки не збіглися з очікуваними, тому основний експеримент не запускаємо.")

    final_f1 = {}

    for regime, seed in RUNS:
        print(f"\n=== {run_name(regime, seed)} на {device} ===", flush=True)

        log = train_run(regime, seed, args.data, rows, device)
        final_f1[(regime, seed)] = log["final_val_macro_f1"]

        print(f"  час навчання {log['train_seconds']:.1f} с", flush=True)

    save_selected_model(final_f1)


if __name__ == "__main__":
    main()
