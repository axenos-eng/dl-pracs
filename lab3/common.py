import csv
import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

LAB_DIR = Path(__file__).resolve().parent
SPLIT_CSV = LAB_DIR / "split.csv"
RUNS_DIR = LAB_DIR / "runs"
FIGURES_DIR = LAB_DIR / "figures"
MODEL_PATH = LAB_DIR / "lab3_model.pt"

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

EPOCHS = 8
TRAIN_BATCH = 32
EVAL_BATCH = 64
WEIGHT_DECAY = 1e-4

TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]

Row = dict[str, str]
Batch = tuple[torch.Tensor, int]


@dataclass
class Regime:
    pretrained: bool
    groups: list[tuple[list[str] | None, float]]
    eval_modules: list[str]


REGIMES = {
    "scratch": Regime(
        pretrained=False,
        groups=[(None, 3e-3)],
        eval_modules=[],
    ),
    "frozen": Regime(
        pretrained=True,
        groups=[(["fc"], 3e-3)],
        eval_modules=["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    ),
    "unfreeze": Regime(
        pretrained=True,
        groups=[(["fc"], 3e-3), (["layer4"], 3e-4)],
        eval_modules=["conv1", "bn1", "layer1", "layer2", "layer3"],
    ),
}

RUNS = [
    ("scratch", 1),
    ("frozen", 1),
    ("frozen", 2),
    ("frozen", 3),
    ("unfreeze", 1),
    ("unfreeze", 2),
    ("unfreeze", 3),
]

TRAIN_TF = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

EVAL_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def class_files(folder: Path) -> list[str]:
    names = [path.name for path in folder.iterdir() if path.suffix == ".jpg"]

    return sorted(names)


def build_split(data_dir: Path) -> list[Row]:
    rows = []

    rng = np.random.default_rng(0)

    for label in CLASSES:
        files = class_files(data_dir / "seg_train" / "seg_train" / label)

        idx = np.arange(len(files))
        rng.shuffle(idx)

        for i in idx[:200]:
            rows.append({"file": files[i], "label": label, "part": "train"})

    rng = np.random.default_rng(0)

    for label in CLASSES:
        files = class_files(data_dir / "seg_test" / "seg_test" / label)

        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2

        for i in idx[:cut]:
            rows.append({"file": files[i], "label": label, "part": "validation"})

        for i in idx[cut:]:
            rows.append({"file": files[i], "label": label, "part": "test"})

    return rows


def write_split(rows: list[Row]) -> None:
    with open(SPLIT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "part"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_split() -> list[Row]:
    with open(SPLIT_CSV, newline="") as f:
        return list(csv.DictReader(f))


def split_sha256() -> str:
    return hashlib.sha256(SPLIT_CSV.read_bytes()).hexdigest()


def rows_of(rows: list[Row], part: str) -> list[Row]:
    return [row for row in rows if row["part"] == part]


def image_path(data_dir: Path, row: Row) -> Path:
    folder = "seg_train" if row["part"] == "train" else "seg_test"

    return data_dir / folder / folder / row["label"] / row["file"]


class SplitDataset(Dataset[Batch]):
    def __init__(self, data_dir: Path, rows: list[Row], transform):
        self.paths = [image_path(data_dir, row) for row in rows]
        self.labels = [CLASSES.index(row["label"]) for row in rows]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")

        return self.transform(image), self.labels[i]


def train_loader(data_dir: Path, rows: list[Row]) -> DataLoader[Batch]:
    dataset = SplitDataset(data_dir, rows_of(rows, "train"), TRAIN_TF)

    return DataLoader(
        dataset,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )


def eval_loader(data_dir: Path, rows: list[Row], part: str) -> DataLoader[Batch]:
    dataset = SplitDataset(data_dir, rows_of(rows, part), EVAL_TF)

    return DataLoader(
        dataset,
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=0,
    )


def module_params(model: nn.Module, modules: list[str] | None) -> list[nn.Parameter]:
    if modules is None:
        return list(model.parameters())

    params = []
    for name in modules:
        params += list(getattr(model, name).parameters())

    return params


def count_params(params) -> int:
    return sum(param.numel() for param in params)


def build_model(regime: str, seed: int, device: str):
    config = REGIMES[regime]

    torch.manual_seed(seed)
    np.random.seed(seed)

    weights = ResNet18_Weights.IMAGENET1K_V1 if config.pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(512, len(CLASSES))

    for param in model.parameters():
        param.requires_grad = False

    for modules, _ in config.groups:
        for param in module_params(model, modules):
            param.requires_grad = True

    model.to(device)

    groups = []
    for modules, lr in config.groups:
        groups.append({"params": module_params(model, modules), "lr": lr})

    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)

    return model, optimizer


def set_train_mode(model: nn.Module, regime: str) -> None:
    model.train()

    for name in REGIMES[regime].eval_modules:
        getattr(model, name).eval()


def train_epoch(model, optimizer, loader, device) -> float:
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_count += len(y)

    return total_loss / total_count


@torch.no_grad()
def predict_logits(model, loader, device) -> np.ndarray:
    model.eval()
    logits = []

    for x, _ in loader:
        logits.append(model(x.to(device)).cpu())

    return torch.cat(logits).numpy()


def labels_of(loader) -> np.ndarray:
    return np.array(loader.dataset.labels)


def macro_f1(labels: np.ndarray, logits: np.ndarray) -> float:
    predictions = logits.argmax(1)
    score = f1_score(labels, predictions, labels=range(len(CLASSES)), average="macro")

    return float(score)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name()

    return f"{device} ({platform.machine()}, {platform.platform()})"


def run_git(*args) -> str:
    result = subprocess.run(["git", *args], cwd=LAB_DIR, capture_output=True, text=True)

    return result.stdout.strip()


def git_info() -> dict[str, Any]:
    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain", "--", "*.py", ":(top)pyproject.toml", ":(top)uv.lock")

    return {"commit": commit, "dirty": bool(status)}


def versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }


def run_name(regime: str, seed: int) -> str:
    return f"{regime}_seed{seed}"


def select_run(final_f1: dict[tuple[str, int], float]) -> tuple[str, int]:
    means = {}

    for regime in REGIMES:
        values = []

        for (run_regime, _), f1 in final_f1.items():
            if run_regime == regime:
                values.append(f1)

        means[regime] = sum(values) / len(values)

    best_regime = max(means, key=lambda regime: means[regime])

    seeds = [seed for regime, seed in final_f1 if regime == best_regime]
    best_seed = max(seeds, key=lambda seed: final_f1[(best_regime, seed)])

    return best_regime, best_seed


def denormalize(x: torch.Tensor) -> np.ndarray:
    std = torch.tensor(STD).view(3, 1, 1)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    image = x * std + mean

    return image.clamp(0, 1).permute(1, 2, 0).numpy()
