"""Training loop for Vision LLM."""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from .dataset import EchoDataset, UppercaseDataset
from .model import VisionLLM


def combined_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """Weighted combination of MSE and (1 - SSIM) loss."""
    mse = nn.functional.mse_loss(pred, target)
    ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
    return alpha * mse + (1 - alpha) * (1 - ssim_val)


def train(
    task: str = "echo",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    dataset_size: int = 10000,
    device: str | None = None,
    save_dir: str = "checkpoints",
):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device} | Task: {task} | Epochs: {epochs}")

    # Dataset
    ds_cls = {"echo": EchoDataset, "uppercase": UppercaseDataset}[task]
    train_ds = ds_cls(size=dataset_size, seed=42)
    val_ds = ds_cls(size=dataset_size // 10, seed=99)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    # Model
    model = VisionLLM(latent_dim=256, think_depth=3).to(device)
    print(f"Parameters: {model.param_count():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for src, tgt in train_dl:
            src, tgt = src.to(device), tgt.to(device)
            pred = model(src)
            loss = combined_loss(pred, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * src.size(0)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_dl:
                src, tgt = src.to(device), tgt.to(device)
                pred = model(src)
                val_loss += combined_loss(pred, tgt).item() * src.size(0)
        val_loss /= len(val_ds)

        scheduler.step()
        elapsed = time.time() - t0
        print(f"[{epoch:3d}/{epochs}] train={train_loss:.4f} val={val_loss:.4f} ({elapsed:.1f}s)")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path / f"best_{task}.pt")

    torch.save(model.state_dict(), save_path / f"last_{task}.pt")
    print(f"Done. Best val loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Vision LLM")
    parser.add_argument("--task", choices=["echo", "uppercase"], default="echo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-size", type=int, default=10000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    train(
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset_size=args.dataset_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
