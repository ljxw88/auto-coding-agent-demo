"""
Train CIFAR-10 classifier.

Usage:
    python train.py --epochs 500
    python train.py --epochs 10          # smoke test
    python train.py --epochs 500 --lr 0.01
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import config
from model import get_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_loaders(batch_size: int):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True, transform=test_transform
    )

    # Optionally use a subset for faster per-epoch timing
    if getattr(config, "TRAIN_SUBSET", None):
        indices = list(range(config.TRAIN_SUBSET))
        train_set = torch.utils.data.Subset(train_set, indices)

    # num_workers=0 avoids issues with MPS + multiprocessing
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    return train_loader, test_loader


def get_optimizer(model, lr: float):
    if config.OPTIMIZER == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=0.9, weight_decay=config.WEIGHT_DECAY
        )
    return torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )


def get_scheduler(optimizer, epochs: int):
    if config.LR_SCHEDULE == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif config.LR_SCHEDULE == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    return None


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 classifier")
    parser.add_argument("--epochs", type=int, default=config.full_train_epochs if hasattr(config, 'full_train_epochs') else 500)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    device = get_device()
    print(f"[train] Device: {device}")
    print(f"[train] Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    train_loader, test_loader = get_loaders(args.batch_size)
    model = get_model(config.NUM_CLASSES).to(device)
    optimizer = get_optimizer(model, args.lr)
    scheduler = get_scheduler(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "test_loss": test_loss,
            }, config.CHECKPOINT_FILE)

        if epoch % 10 == 0 or epoch <= 5 or epoch == args.epochs:
            avg_time = sum(epoch_times[-10:]) / min(len(epoch_times), 10)
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"Train acc: {train_acc:.4f} loss: {train_loss:.4f} | "
                f"Test acc: {test_acc:.4f} loss: {test_loss:.4f} | "
                f"Best: {best_acc:.4f} | "
                f"{elapsed:.1f}s ({avg_time:.1f}s/ep avg)"
            )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"\n[train] Done. Best test acc: {best_acc:.4f}")
    print(f"[train] Avg epoch time: {avg_epoch_time:.2f}s")


if __name__ == "__main__":
    main()
