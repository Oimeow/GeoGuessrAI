import math
import matplotlib.pyplot as plt
import torch


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    country_codes: list[str],
    dev: torch.device | None = None,
    max_examples: int = 16,
) -> dict:
    if dev is None:
        dev = next(model.parameters()).device

    num_classes = len(country_codes)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    examples = []
    correct = total = 0
    model.eval()

    for images, labels in data_loader:
        logits = model(images.to(dev))
        predictions = logits.argmax(dim=1).cpu()
        labels = labels.cpu()
        correct += (predictions == labels).sum().item()
        total += labels.numel()

        for image, actual, predicted in zip(images, labels, predictions):
            confusion[actual.item(), predicted.item()] += 1
            if len(examples) < max_examples:
                examples.append((image.cpu(), actual.item(), predicted.item()))

    if total == 0:
        raise ValueError("The test loader contained no examples.")

    acc = correct / total
    print(f"test_acc: {acc:.1%} ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total, "confusion_matrix": confusion, "examples": examples}


def plot_confusion_matrix(confusion: torch.Tensor, country_codes: list[str]):
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis, label="number of test images")
    axis.set(xticks=range(len(country_codes)), yticks=range(len(country_codes)), xticklabels=country_codes, yticklabels=country_codes, xlabel="predicted country", ylabel="actual country", title="Country-classifier confusion matrix")

    threshold = confusion.max().item() / 2 if confusion.numel() else 0
    for row in range(confusion.size(0)):
        for col in range(confusion.size(1)):
            value = confusion[row, col].item()
            axis.text(col, row, str(value), ha="center", va="center", color="white" if value > threshold else "black")
    figure.tight_layout()
    return figure


def plot_prediction_grid(examples: list[tuple[torch.Tensor, int, int]], country_codes: list[str]):
    if not examples:
        raise ValueError("No examples were available to plot.")

    ncols = 4
    nrows = math.ceil(len(examples) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 4 * nrows))
    axes = axes.flatten()

    # Reverse the normalization performed by ResNet18_Weights.DEFAULT.transforms().
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for axis, (image, actual, predicted) in zip(axes, examples):
        display_image = (image * std + mean).clamp(0, 1).permute(1, 2, 0)
        axis.imshow(display_image)
        is_correct = actual == predicted
        color = "#00aa00" if is_correct else "#ff0000"
        axis.set_title(f"y: {country_codes[actual]}  |  net(X): {country_codes[predicted]}", color=color, fontweight="bold")
        axis.axis("off")

    for axis in axes[len(examples):]:
        axis.axis("off")

    figure.suptitle("Held-out predictions", fontsize=16)
    figure.tight_layout()
    return figure
