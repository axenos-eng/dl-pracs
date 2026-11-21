import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from common import (
    CLASSES,
    FIGURES_DIR,
    REGIMES,
    TOP_MODULES,
    Row,
    build_model,
    count_params,
    denormalize,
    set_train_mode,
    train_loader,
)

EXPECTED_SPLIT = {
    "buildings": {"train": (200, 1996761), "validation": (218, 4837323), "test": (219, 4837559)},
    "forest": {"train": (200, 2032625), "validation": (237, 5276962), "test": (237, 5291326)},
    "glacier": {"train": (200, 2183923), "validation": (276, 6135652), "test": (277, 6157250)},
    "mountain": {"train": (200, 1950345), "validation": (262, 5789535), "test": (263, 5849765)},
    "sea": {"train": (200, 1987613), "validation": (255, 5655068), "test": (255, 5668011)},
    "street": {"train": (200, 1841817), "validation": (250, 5541845), "test": (251, 5547533)},
}

TOTAL_PARAMS = 11_179_590
EXPECTED_TRAINABLE = {"scratch": 11_179_590, "frozen": 3_078, "unfreeze": 8_396_806}

BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def check_split(rows: list[Row]) -> bool:
    table = []
    all_match = True

    for label in CLASSES:
        for part in ["train", "validation", "test"]:
            files = [row["file"] for row in rows if row["label"] == label and row["part"] == part]

            count = len(files)
            total = sum(int(name.removesuffix(".jpg")) for name in files)
            ok = (count, total) == EXPECTED_SPLIT[label][part]

            all_match = all_match and ok
            table.append({"label": label, "part": part, "count": count, "sum": total, "ok": ok})

    all_files = [row["file"] for row in rows]
    no_overlap = len(all_files) == len(set(all_files))

    print("\nСпліт (кількість і сума числових імен файлів):")
    print(pd.DataFrame(table).to_string(index=False))
    print(f"Усі клітинки збігаються: {all_match}, файлів у двох частинах немає: {no_overlap}")

    return all_match and no_overlap


def save_batch_figure(data_dir, rows) -> None:
    torch.manual_seed(0)
    x, y = next(iter(train_loader(data_dir, rows)))

    fig, axes = plt.subplots(4, 4, figsize=(8, 8.8))

    for ax, image, label in zip(axes.flat, x[:16], y[:16]):
        ax.imshow(denormalize(image))
        ax.set_title(CLASSES[label], fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "train_batch.png", dpi=120)
    plt.close(fig)


def initial_loss(data_dir, rows, device) -> None:
    print(f"\nПочатковий loss L0 (seed 1, перший батч), ln 6 = {math.log(6):.4f}")

    for regime in ["scratch", "frozen"]:
        model, _ = build_model(regime, seed=1, device=device)
        x, y = next(iter(train_loader(data_dir, rows)))

        set_train_mode(model, regime)

        with torch.no_grad():
            logits = model(x.to(device))
            loss = nn.CrossEntropyLoss()(logits, y.to(device)).item()

        print(f"  {regime}: L0 = {loss:.4f}")


def check_params(device) -> bool:
    print("\nПараметри (сума numel):")
    ok = True

    for regime, config in REGIMES.items():
        model, optimizer = build_model(regime, seed=1, device=device)

        total = count_params(model.parameters())
        trainable = count_params(p for p in model.parameters() if p.requires_grad)

        groups = []
        frozen_in_optimizer = False

        for (modules, _), group in zip(config.groups, optimizer.param_groups):
            groups.append((modules or "усі", count_params(group["params"]), group["lr"]))

            for param in group["params"]:
                if not param.requires_grad:
                    frozen_in_optimizer = True

        in_optimizer = sum(count for _, count, _ in groups)

        print(f"  {regime}: усього {total}, requires_grad=True {trainable}")
        print(f"    групи AdamW (модулі, параметрів, lr): {groups}")
        print(f"    requires_grad=False в оптимізаторі: {frozen_in_optimizer}")

        regime_ok = (
            total == TOTAL_PARAMS
            and trainable == EXPECTED_TRAINABLE[regime]
            and in_optimizer == trainable
            and not frozen_in_optimizer
        )
        ok = ok and regime_ok

    return ok


def snapshot(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def five_steps(model, optimizer, data_dir, rows, device) -> None:
    loss_fn = nn.CrossEntropyLoss()

    for step, (x, y) in enumerate(train_loader(data_dir, rows)):
        if step == 5:
            break

        optimizer.zero_grad()
        loss = loss_fn(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()


def any_changed(before, after, keys) -> bool:
    for key in keys:
        if not torch.equal(before[key], after[key]):
            return True

    return False


def compare_modules(before, after, params_should_change, bn_should_change) -> pd.DataFrame:
    rows = []

    for module in TOP_MODULES:
        param_keys = []
        bn_keys = []

        for key in before:
            if key.split(".")[0] != module:
                continue

            if key.endswith(BN_BUFFERS):
                bn_keys.append(key)
            else:
                param_keys.append(key)

        params_changed = any_changed(before, after, param_keys)
        params_ok = params_changed == (module in params_should_change)

        if bn_keys:
            bn_changed = any_changed(before, after, bn_keys)
            bn_ok = bn_changed == (module in bn_should_change)
        else:
            bn_changed = "немає BN"
            bn_ok = True

        rows.append({
            "module": module,
            "params_changed": params_changed,
            "bn_changed": bn_changed,
            "ok": params_ok and bn_ok,
        })

    return pd.DataFrame(rows)


def check_changes(data_dir, rows, device) -> bool:
    ok = True

    for regime in ["frozen", "unfreeze"]:
        model, optimizer = build_model(regime, seed=1, device=device)
        before = snapshot(model)

        set_train_mode(model, regime)
        five_steps(model, optimizer, data_dir, rows, device)
        after = snapshot(model)

        trained = set(TOP_MODULES) - set(REGIMES[regime].eval_modules)
        table = compare_modules(before, after, trained, trained)

        print(f"\n{regime}, 5 кроків із set_train_mode:")
        print(table.to_string(index=False))

        ok = ok and bool(table["ok"].all())

    return ok


def bn_experiment(data_dir, rows, device) -> None:
    model, optimizer = build_model("frozen", seed=1, device=device)
    before = snapshot(model)

    model.train()
    five_steps(model, optimizer, data_dir, rows, device)
    after = snapshot(model)

    table = compare_modules(before, after, {"fc"}, set(TOP_MODULES))

    print("\nfrozen, 5 кроків із model.train() (навмисна помилка):")
    print(table.to_string(index=False))

    for name in ["running_mean", "running_var"]:
        key = f"layer1.0.bn1.{name}"
        change = (after[key] - before[key]).abs().max().item()
        print(f"  max |зміна| {key}: {change:.6f}")


def run_all(data_dir: Path, rows, device: str) -> bool:
    FIGURES_DIR.mkdir(exist_ok=True)

    split_ok = check_split(rows)
    save_batch_figure(data_dir, rows)
    initial_loss(data_dir, rows, device)
    params_ok = check_params(device)
    changes_ok = check_changes(data_dir, rows, device)
    bn_experiment(data_dir, rows, device)

    return split_ok and params_ok and changes_ok
