# continue_training.py
# Utilities to resume training cleanly from where you left off.

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_checkpoint_if_available(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    strict: bool = True,
) -> Tuple[int, float, List[float], List[float]]:
    """
    Returns:
      start_epoch: int (next epoch to run)
      best_val: float
      train_losses: list
      val_losses: list

    Behavior:
    - If ckpt_path exists and is a full checkpoint dict, restore model+optimizer+history.
    - If ckpt_path does not exist, start fresh.
    - If user accidentally points ckpt_path to a model-only state_dict, we still load it,
      but optimizer/history will be fresh.
    """
    if not os.path.exists(ckpt_path):
        return 1, float("inf"), [], []

    ckpt = torch.load(ckpt_path, map_location=device)

    # Case A: model-only state_dict (param_name -> tensor)
    # Heuristic: dict values are tensors, and no 'model_state' key exists
    if isinstance(ckpt, dict) and "model_state" not in ckpt and "optimizer_state" not in ckpt and "epoch" not in ckpt:
        model.load_state_dict(ckpt, strict=strict)
        return 1, float("inf"), [], []

    # Case B: full checkpoint
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"Checkpoint at {ckpt_path} is not recognized.")

    model.load_state_dict(ckpt["model_state"], strict=strict)

    if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"Warning: could not load optimizer state ({e}). Continuing with fresh optimizer.")

    last_epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    train_losses = [float(x) for x in ckpt.get("train_losses", [])]
    val_losses = [float(x) for x in ckpt.get("val_losses", [])]

    # continue from next epoch
    return last_epoch + 1, best_val, train_losses, val_losses


def save_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    train_losses: List[float],
    val_losses: List[float],
) -> None:
    """
    Saves a resume-capable checkpoint.
    epoch should be the last completed epoch.
    """
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_losses": list(train_losses),
        "val_losses": list(val_losses),
    }
    torch.save(payload, ckpt_path)


def plot_losses(
    plot_dir: str,
    filename: str,
    expt_description: str,
    train_losses: List[float],
    val_losses: List[float],
    hours: int,
    minutes: int,
) -> str:
    """
    Writes plots/<filename>.png and returns its path.
    Curve length matches total history, so it continues when you resume.
    """
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{filename}.png")

    n = len(train_losses)
    xs = np.arange(1, n + 1)

    plt.figure()
    plt.title(expt_description)
    plt.plot(xs, np.array(train_losses), "b")
    plt.plot(xs, np.array(val_losses), "r")
    plt.legend(["Train", "Validation"])
    plt.ylabel("Loss")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    hh = hours + (1 if minutes > 30 else 0)
    plt.xlabel(f"Epochs ({int(hh)} hours)")
    plt.savefig(plot_path)
    plt.close()

    return plot_path
