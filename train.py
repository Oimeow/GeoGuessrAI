from __future__ import annotations

import torch
import torch.nn as nn


def train(
    model: torch.nn.Module,
    train_loader,
    dev: torch.device,
    epochs: int = 10,
    lr: float = 3e-4,
    decay: float = 1e-4,
) -> list[dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    history = []

    for ep in range(epochs):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(dev), labels.to(dev)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()

        metrics = {"loss": loss_sum / total, "accuracy": correct / total}
        history.append(metrics)
        print(f"epoch {ep + 1}/{epochs} | loss={metrics['loss']:.4f} | train_acc={metrics['accuracy']:.1%}")

    return history
