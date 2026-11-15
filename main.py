from pathlib import Path

import matplotlib.pyplot as plt
import torch

from eval import evaluate, plot_confusion_matrix, plot_prediction_grid
from load_data import get_data
from network import CountryClassifier
from train import train


COUNTRY_CODES = ["BR", "IN", "US", "MX"]
DATASET_DIR = Path("data/wsv500k_250")
BATCH_SIZE = 8
EPOCHS = 3
FREEZE_BACKBONE = False
SAVE_MODEL = False
MODEL_PATH = Path("models/country_classifier.pt")
SHOW_PLOTS = True


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(save_model: bool = SAVE_MODEL, show_plots: bool = SHOW_PLOTS):
    dev = choose_device()
    print(f"Using device: {dev}")

    train_loader, test_loader, label_to_index = get_data(
        country_codes=COUNTRY_CODES,
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
    )

    model = CountryClassifier(num_classes=len(COUNTRY_CODES), freeze_backbone=FREEZE_BACKBONE).to(dev)

    history = train(model, train_loader, dev=dev, epochs=EPOCHS)
    results = evaluate(model, test_loader, COUNTRY_CODES, dev=dev)

    if save_model:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "country_codes": COUNTRY_CODES,
            "freeze_backbone": FREEZE_BACKBONE,
            "history": history,
            "label_to_index": label_to_index,
        }, MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")

    if show_plots:
        plot_confusion_matrix(results["confusion_matrix"], COUNTRY_CODES)
        plot_prediction_grid(results["examples"], COUNTRY_CODES)
        plt.show()

    return model, results


if __name__ == "__main__":
    main()
