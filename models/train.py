import os
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from utils import (
    eval_transformations,
    mixup_cutmix_collate_fn,
    top_accuracy,
    train_transformations,
)
from vit import BasicViT


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 0

DATA_DIR = "/imagenet"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

NUM_CLASSES = 1000
BATCH_SIZE = 1024
VAL_BATCH_SIZE = BATCH_SIZE * 2
GRADIENT_ACCUMULATION_STEPS = 1

EMBED_DIM = 384
HIDDEN_DIM = 1536
NUM_LAYERS = 12
NUM_HEADS = 6

PRUNE_RATE = 0.0
DROP_AT_LAYERS = (0,)

DROPOUT = 0.0
KV_DROPOUT = 0.0
EMBEDDING_LAYER = 'two'
DROPOUT_P = 0.0
WEIGHT_DECAY = 0.1
LABEL_SMOOTHING = 0.1
LEARNING_RATE = 0.001

NUM_EPOCHS = 300
WARMUP_EPOCHS = 5

TRAIN_NUM_WORKERS = 8
VAL_NUM_WORKERS = 2

EXPERIMENT_NAME = "pat_vit"
RUN_NAME = "vit_base_prune0"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    val_loss: float,
    best_acc: float,
) -> None:
    """Save a training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_acc": best_acc,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    # -----------------------------------------------------------------------
    # Dataset and dataloaders
    # -----------------------------------------------------------------------

    train_transform = train_transformations()
    val_transform = eval_transformations()
    mixup_cutmix_collate = mixup_cutmix_collate_fn()

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=mixup_cutmix_collate,
        num_workers=TRAIN_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = BasicViT(
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=NUM_LAYERS,
        heads=NUM_HEADS,
        dropout=DROPOUT,
        kv_dropout=KV_DROPOUT,
        drop_at_layers=DROP_AT_LAYERS,
        embedding_layer=EMBEDDING_LAYER,
    )

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(model)
    print(f"Model parameters: {num_parameters:,}")
    print(f"Device: {device}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Run: {RUN_NAME}")

    # -----------------------------------------------------------------------
    # Optimizer and scheduler
    # -----------------------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS - WARMUP_EPOCHS,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
    ).to(device)

    # -----------------------------------------------------------------------
    # Training state
    # -----------------------------------------------------------------------

    best_acc = 0.0

    train_losses = []
    val_losses = []
    val_top1_accuracies = []
    val_top5_accuracies = []
    epoch_times = []

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nStarting epoch {epoch + 1}/{NUM_EPOCHS} "
            f"| {datetime.now()} "
            f"| prune_rate={PRUNE_RATE:.4f} "
            f"| lr={current_lr:.8f}"
        )

        start_time = time.time()

        # -------------------------------------------------------------------
        # Train
        # -------------------------------------------------------------------

        model.train()
        optimizer.zero_grad()

        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            predictions = model(
                x=images,
                prune_rate=PRUNE_RATE,
            )

            loss = criterion(predictions, labels)

            running_loss += loss.item()

            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (
                (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0
                or (batch_idx + 1) == len(train_loader)
            ):
                optimizer.step()
                optimizer.zero_grad()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        scheduler.step()

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------

        model.eval()

        total_val_loss = 0.0
        total_correct_top1 = 0
        total_correct_top5 = 0
        total_samples = 0

        with torch.no_grad():

            for val_images, val_labels in val_loader:

                val_images = val_images.to(
                    device,
                    non_blocking=True,
                )
                val_labels = val_labels.to(
                    device,
                    non_blocking=True,
                )

                val_outputs = model(
                    val_images,
                    prune_rate=PRUNE_RATE,
                )

                batch_val_loss = criterion(
                    val_outputs,
                    val_labels,
                )

                batch_size = val_images.size(0)

                total_val_loss += (
                    batch_val_loss.item() * batch_size
                )

                top1_acc, top5_acc = top_accuracy(
                    val_outputs,
                    val_labels,
                    topk=(1, 5),
                )

                total_correct_top1 += (
                    top1_acc * batch_size / 100.0
                )

                total_correct_top5 += (
                    top5_acc * batch_size / 100.0
                )

                total_samples += batch_size

        avg_val_loss = total_val_loss / total_samples
        avg_top1 = (
            total_correct_top1 / total_samples
        ) * 100.0
        avg_top5 = (
            total_correct_top5 / total_samples
        ) * 100.0

        val_losses.append(avg_val_loss)
        val_top1_accuracies.append(avg_top1)
        val_top5_accuracies.append(avg_top5)

        # -------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"top1={avg_top1:.2f}% | "
            f"top5={avg_top5:.2f}% | "
            f"time={epoch_time / 60:.1f} min"
        )

        # -------------------------------------------------------------------
        # Checkpointing
        # -------------------------------------------------------------------

        is_best = avg_top1 > best_acc

        if is_best:
            best_acc = avg_top1

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": avg_val_loss,
            "best_acc": best_acc,
        }

        latest_checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"{RUN_NAME}_latest.pth",
        )

        torch.save(
            checkpoint,
            latest_checkpoint_path,
        )

        if is_best:
            best_checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"{RUN_NAME}_best.pth",
            )

            torch.save(
                checkpoint,
                best_checkpoint_path,
            )

            print(
                f"New best model: "
                f"{best_acc:.2f}% top-1"
            )

        print(
            f"Finished epoch {epoch + 1}/{NUM_EPOCHS} "
            f"| {datetime.now()}"
        )

    # -----------------------------------------------------------------------
    # Save training curves
    # -----------------------------------------------------------------------

    epochs = range(1, NUM_EPOCHS + 1)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    axes[0].plot(
        epochs,
        train_losses,
        label="Train Loss",
    )
    axes[0].plot(
        epochs,
        val_losses,
        label="Validation Loss",
    )
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        epochs,
        val_top1_accuracies,
        label="Top-1 Accuracy",
    )
    axes[1].plot(
        epochs,
        val_top5_accuracies,
        label="Top-5 Accuracy",
    )
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.tight_layout()

    plot_path = os.path.join(
        RESULTS_DIR,
        f"{RUN_NAME}_training_curves.png",
    )

    fig.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    print("\nTraining complete.")
    print(f"Best validation top-1 accuracy: {best_acc:.2f}%")
    print(f"Training curves saved to: {plot_path}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"Finished at: {datetime.now()}")


if __name__ == "__main__":
    main()