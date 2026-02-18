"""
Evaluate CIFAR-10 classifier and write metrics to JSON.

Usage:
    python eval.py
    python eval.py --output results.json
    python eval.py --checkpoint checkpoints/best.pt --output results.json
"""

import argparse
import json
import os
import time

import torch
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


@torch.no_grad()
def run_eval(checkpoint_path: str, output_path: str):
    device = get_device()

    # Load test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_set = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=0, pin_memory=False
    )

    # Load model
    model = get_model(config.NUM_CLASSES).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        trained_epochs = ckpt.get("epoch", 0)
        print(f"[eval] Loaded checkpoint from epoch {trained_epochs}: {checkpoint_path}")
    else:
        print(f"[eval] No checkpoint found at {checkpoint_path} — evaluating untrained model")
        trained_epochs = 0

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    class_correct = [0] * config.NUM_CLASSES
    class_total = [0] * config.NUM_CLASSES

    t0 = time.time()
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct_mask = predicted.eq(targets)
        correct += correct_mask.sum().item()
        total += inputs.size(0)

        for i in range(inputs.size(0)):
            label = targets[i].item()
            class_correct[label] += correct_mask[i].item()
            class_total[label] += 1

    elapsed = time.time() - t0
    accuracy = correct / total
    avg_loss = total_loss / total

    CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
    per_class = {CLASSES[i]: round(class_correct[i] / class_total[i], 4) for i in range(10)}

    metrics = {
        "accuracy": round(accuracy, 6),
        "loss": round(avg_loss, 6),
        "correct": correct,
        "total": total,
        "eval_time_s": round(elapsed, 2),
        "trained_epochs": trained_epochs,
        "per_class_accuracy": per_class,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[eval] Accuracy: {accuracy:.4f} | Loss: {avg_loss:.4f}")
    print(f"[eval] Metrics written to: {output_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_FILE)
    parser.add_argument("--output", default=config.METRICS_FILE)
    args = parser.parse_args()
    run_eval(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
