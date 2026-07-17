from pathlib import Path
import copy
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# # Values used throughout the script
SEED = 42
IMAGE_SIZE = 128
EPOCHS = 10


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "split"
MODEL_FOLDER = PROJECT_ROOT / "hw3_output" / "models"
PLOT_FOLDER = PROJECT_ROOT / "hw3_output" / "plots"
RESULT_FOLDER = PROJECT_ROOT / "hw3_output" / "results"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


# Random changes are used only for training images
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor()
])

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# Loading the prepared dataset
train_dataset = datasets.ImageFolder(
    DATA_FOLDER / "train",
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    DATA_FOLDER / "validation",
    transform=validation_transform
)


# CNN used for each experiment
class FishCNN(nn.Module):
    def __init__(self, number_of_classes, dropout_rate):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, number_of_classes)
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


# Training the model for one epoch
def train_one_epoch(model, loader, loss_function, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# Checking the model on validation images
def validate(model, loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# Changing one setting at a time
experiments = [
    {
        "name": "Baseline",
        "learning_rate": 0.001,
        "batch_size": 32,
        "dropout": 0.30
    },
    {
        "name": "Learning Rate 0.0005",
        "learning_rate": 0.0005,
        "batch_size": 32,
        "dropout": 0.30
    },
    {
        "name": "Batch Size 16",
        "learning_rate": 0.001,
        "batch_size": 16,
        "dropout": 0.30
    },
    {
        "name": "Dropout 0.50",
        "learning_rate": 0.001,
        "batch_size": 32,
        "dropout": 0.50
    }
]


def main():
    if not DATA_FOLDER.exists():
        raise FileNotFoundError(
            "The split dataset was not found. Run prepare_fish_dataset.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    number_of_classes = len(train_dataset.classes)

    tuning_results = []
    best_overall_accuracy = 0.0
    best_overall_model = None
    best_experiment = None

    print("Device:", device)
    print("Classes:", train_dataset.classes)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Experiments:", len(experiments))

    start_time = time.time()

    for experiment_number, experiment in enumerate(experiments, start=1):

        # Resetting the seed so the comparison is more consistent
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        print("\n" + "=" * 65)
        print(
            f"Experiment {experiment_number}/{len(experiments)}: "
            f"{experiment['name']}"
        )
        print(
            f"Learning rate: {experiment['learning_rate']} | "
            f"Batch size: {experiment['batch_size']} | "
            f"Dropout: {experiment['dropout']}"
        )
        print("=" * 65)

        train_loader = DataLoader(
            train_dataset,
            batch_size=experiment["batch_size"],
            shuffle=True
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=experiment["batch_size"],
            shuffle=False
        )

        # Creating a new model for this experiment
        model = FishCNN(
            number_of_classes,
            experiment["dropout"]
        ).to(device)

        loss_function = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=experiment["learning_rate"]
        )

        best_validation_accuracy = 0.0
        best_validation_loss = float("inf")
        best_epoch = 0
        best_model_state = None

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": []
        }

        for epoch in range(EPOCHS):
            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                loss_function,
                optimizer,
                device
            )

            validation_loss, validation_accuracy = validate(
                model,
                validation_loader,
                loss_function,
                device
            )

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["validation_loss"].append(validation_loss)
            history["validation_accuracy"].append(validation_accuracy)

            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | "
                f"Train Accuracy: {train_accuracy * 100:.2f}% | "
                f"Validation Accuracy: {validation_accuracy * 100:.2f}%"
            )

            # Keeping the epoch with the highest validation accuracy
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())

        experiment_result = {
            "name": experiment["name"],
            "learning_rate": experiment["learning_rate"],
            "batch_size": experiment["batch_size"],
            "dropout": experiment["dropout"],
            "epochs": EPOCHS,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_validation_accuracy": best_validation_accuracy,
            "history": history
        }

        tuning_results.append(experiment_result)

        print(
            f"Best validation accuracy: "
            f"{best_validation_accuracy * 100:.2f}% "
            f"at epoch {best_epoch}"
        )

        # Keeping the best model from all experiments
        if best_validation_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_validation_accuracy
            best_overall_model = best_model_state
            best_experiment = experiment_result

    total_time = time.time() - start_time

    model_path = MODEL_FOLDER / "best_tuned_cnn.pth"

    if best_overall_model is not None:
        torch.save(best_overall_model, model_path)

    results = {
        "classes": train_dataset.classes,
        "image_size": IMAGE_SIZE,
        "epochs_per_experiment": EPOCHS,
        "best_experiment": best_experiment,
        "experiments": tuning_results,
        "total_tuning_time_seconds": total_time
    }

    result_path = RESULT_FOLDER / "tuning_results.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Comparing the best validation accuracy from each experiment
    names = [result["name"] for result in tuning_results]
    accuracies = [
        result["best_validation_accuracy"] * 100
        for result in tuning_results
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies)

    plt.xlabel("Experiment")
    plt.ylabel("Best Validation Accuracy (%)")
    plt.title("Hyperparameter Tuning Comparison")
    plt.ylim(0, 100)
    plt.xticks(rotation=15)

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{accuracy:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    plot_path = PLOT_FOLDER / "hyperparameter_comparison.png"
    plt.savefig(plot_path)
    plt.close()

    print("\n" + "=" * 65)
    print("Hyperparameter tuning completed.")
    print("Best experiment:", best_experiment["name"])
    print(
        f"Best validation accuracy: "
        f"{best_overall_accuracy * 100:.2f}%"
    )
    print(f"Total time: {total_time / 60:.2f} minutes")

    print("\nSaved files:")
    print(model_path)
    print(result_path)
    print(plot_path)


if __name__ == "__main__":
    main()