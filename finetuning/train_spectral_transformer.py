"""Train a GPU-oriented 1-D Vision Transformer for stellar-spectrum regression.

This is a spectral adaptation of the Vision Transformer: fixed wavelength
patches are embedded with Conv1d, then processed by a Transformer encoder.
CPU training is intentionally disabled by default because self-attention over
250 spectral patches makes the 30k/5k/15k experiment impractically slow.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Metrics:
    loss: float
    mae_teff: float
    mae_feh: float
    mae_logg: float


class SpectrumDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class SpectralTransformerRegressor(nn.Module):
    """A 1-D ViT-style regressor for 4000-pixel stellar spectra."""

    def __init__(
        self,
        *,
        input_dim: int = 4000,
        patch_size: int = 16,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        if input_dim % patch_size:
            raise ValueError("input_dim must be divisible by patch_size.")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads.")

        num_patches = input_dim // patch_size
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.input_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(spectra.unsqueeze(1)).transpose(1, 2)
        cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = self.input_norm(torch.cat((cls_token, tokens), dim=1) + self.pos_embed)
        return self.head(self.encoder(tokens)[:, 0])


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _augment(features: np.ndarray, targets: np.ndarray, *, factor: int, noise_level: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if factor <= 1:
        return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)
    rng = np.random.default_rng(seed)
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    sample_std = features.std(axis=1, keepdims=True)
    copies = [features]
    for _ in range(factor - 1):
        level = rng.uniform(0.01, noise_level)
        copies.append(features + level * sample_std * rng.standard_normal(features.shape).astype(np.float32))
    x_augmented = np.concatenate(copies, axis=0)
    y_augmented = np.tile(targets, (factor, 1))
    order = rng.permutation(len(x_augmented))
    return x_augmented[order], y_augmented[order]


def _loss_weights(targets: np.ndarray, device: torch.device) -> torch.Tensor:
    inverse_variance = 1.0 / np.maximum(np.var(targets, axis=0), 1e-8)
    weights = inverse_variance * (len(inverse_variance) / inverse_variance.sum())
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _weighted_huber(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.huber_loss(predictions.float(), targets.float(), delta=1.0, reduction="none")
    return (losses * weights.unsqueeze(0)).mean()


def _autocast(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    weights: torch.Tensor,
    centers: np.ndarray,
    scales: np.ndarray,
    amp: bool,
) -> Metrics:
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _autocast(device, amp):
                output = model(features)
                loss = _weighted_huber(output, labels, weights)
            losses.append(float(loss.cpu()))
            predictions.append(output.float().cpu().numpy())
            targets.append(labels.float().cpu().numpy())
    prediction_physical = np.concatenate(predictions) * scales + centers
    target_physical = np.concatenate(targets) * scales + centers
    mae = np.abs(prediction_physical - target_physical).mean(axis=0)
    return Metrics(float(np.mean(losses)), float(mae[0]), float(mae[1]), float(mae[2]))


def _make_loader(dataset: Dataset, batch_size: int, *, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _learning_rate(step: int, *, total_steps: int, warmup_steps: int, base: float, floor_ratio: float) -> float:
    if step < warmup_steps:
        return base * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return base * (floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def train_spectral_transformer(args: argparse.Namespace) -> Metrics:
    """Train the spectral transformer and save the best validation checkpoint."""
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA GPU detected. SpectralTransformerRegressor is GPU-only by default because CPU training is too slow. "
            "Use --allow-cpu only for a small debugging run."
        )
    if not torch.cuda.is_available():
        warnings.warn("CPU training is extremely slow; use only for debugging, not for reported experiments.", stacklevel=2)
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(data_path, allow_pickle=True) as data:
        x_train = np.asarray(data["X_train_features"], dtype=np.float32)
        y_train = np.asarray(data["y_train_targets"], dtype=np.float32)
        x_val = np.asarray(data["X_val_features"], dtype=np.float32)
        y_val = np.asarray(data["y_val_targets"], dtype=np.float32)
        x_test = np.asarray(data["X_test_features"], dtype=np.float32)
        y_test = np.asarray(data["y_test_targets"], dtype=np.float32)
        centers = np.asarray(data["label_robust_center"], dtype=np.float64).reshape(1, -1)
        scales = np.asarray(data["label_robust_scale"], dtype=np.float64).reshape(1, -1)

    x_train, y_train = _augment(x_train, y_train, factor=args.aug_factor, noise_level=args.noise_level, seed=args.seed)
    model = SpectralTransformerRegressor(
        input_dim=x_train.shape[1], patch_size=args.patch_size, d_model=args.d_model,
        num_layers=args.layers, num_heads=args.heads, mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    weights = _loss_weights(y_train, device)
    train_loader = _make_loader(SpectrumDataset(x_train, y_train), args.batch_size, shuffle=True, workers=args.workers)
    val_loader = _make_loader(SpectrumDataset(x_val, y_val), args.eval_batch_size, shuffle=False, workers=args.workers)
    test_loader = _make_loader(SpectrumDataset(x_test, y_test), args.eval_batch_size, shuffle=False, workers=args.workers)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(args.warmup_fraction * total_steps)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_loss = float("inf")
    step = 0

    print(f"Device: {device}; trainable parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True):
            for group in optimizer.param_groups:
                group["lr"] = _learning_rate(step, total_steps=total_steps, warmup_steps=warmup_steps, base=args.learning_rate, floor_ratio=args.min_lr_ratio)
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                loss = _weighted_huber(model(features), labels, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1

        validation = _evaluate(model, val_loader, device=device, weights=weights, centers=centers, scales=scales, amp=amp)
        print(f"Validation: {validation}")
        if validation.loss < best_loss:
            best_loss = validation.loss
            torch.save({"model": model.state_dict(), "args": vars(args), "validation": asdict(validation), "epoch": epoch}, output_dir / "best_model.pt")

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test = _evaluate(model, test_loader, device=device, weights=weights, centers=centers, scales=scales, amp=amp)
    (output_dir / "test_metrics.json").write_text(json.dumps(asdict(test), indent=2), encoding="utf-8")
    print(f"Test: {test}")
    return test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GPU-oriented 1-D ViT-style spectral transformer.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "sdss_dr12_processed_flux_benchmark.npz")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_and_evaluations" / "spectral_transformer")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aug-factor", type=int, default=3)
    parser.add_argument("--noise-level", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Debug only: CPU training is impractically slow.")
    return parser.parse_args()


if __name__ == "__main__":
    train_spectral_transformer(parse_args())
