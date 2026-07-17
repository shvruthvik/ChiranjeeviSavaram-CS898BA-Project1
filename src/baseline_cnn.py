from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Values used throughout the script
SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001


# Keep the results reproducible
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "split"
MODEL_FOLDER = PROJECT_ROOT / "hw3_output" / "models"
PLOT_FOLDER = PROJECT_ROOT / "hw3_output" / "plots"
RESULT_FOLDER = PROJECT_ROOT / "hw3_output" / "results"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


# Random changes are only used for training images
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# Loading the prepared train, validation, and test folders
train_dataset = datasets.ImageFolder(
    DATA_FOLDER / "train",
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    DATA_FOLDER / "validation",
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    DATA_FOLDER / "test",
    transform=test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Simple CNN created for fish classification
class FishCNN(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        # Learning image features
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

        # Using the learned features to predict a fish class
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, number_of_classes)
        )

    def forward(self, images):
        features = self.features(images)
        predictions = self.classifier(features)
        return predictions


# Training the model for one epoch
def train_one_epoch(model, loader, loss_function, optimizer, device):
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predicted_labels = outputs.argmax(dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_images += labels.size(0)

    average_loss = total_loss / total_images
    accuracy = correct_predictions / total_images

    return average_loss, accuracy


# Checking validation or test performance
def evaluate(model, loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predicted_labels = outputs.argmax(dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_images += labels.size(0)

    average_loss = total_loss / total_images
    accuracy = correct_predictions / total_images

    return average_loss, accuracy


def main():
    if not DATA_FOLDER.exists():
        raise FileNotFoundError(
            "The split dataset was not found. Run prepare_fish_dataset.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    number_of_classes = len(train_dataset.classes)
    model = FishCNN(number_of_classes).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("Device:", device)
    print("Classes:", train_dataset.classes)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Testing images:", len(test_dataset))
    print("\nStarting baseline CNN training...\n")

    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": []
    }

    best_validation_loss = float("inf")
    best_epoch = 0

    model_path = MODEL_FOLDER / "baseline_cnn.pth"

    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            loss_function,
            device
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy * 100:.2f}% | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy * 100:.2f}%"
        )

        # Save the model when validation loss improves
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)

    training_time = time.time() - start_time

    # Loading the best saved model before testing
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        loss_function,
        device
    )

    print("\nTraining completed.")
    print("Best epoch:", best_epoch)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Training time: {training_time / 60:.2f} minutes")

    # Saving the training information
    results = {
        "model": "Custom Baseline CNN",
        "classes": train_dataset.classes,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "training_time_seconds": training_time,
        "history": history
    }

    result_path = RESULT_FOLDER / "baseline_cnn_results.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Accuracy plot
    epochs = range(1, EPOCHS + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Training")
    plt.plot(epochs, history["validation_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline CNN Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER / "baseline_accuracy.png")
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training")
    plt.plot(epochs, history["validation_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline CNN Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER / "baseline_loss.png")
    plt.close()

    print("\nSaved files:")
    print(model_path)
    print(result_path)
    print(PLOT_FOLDER / "baseline_accuracy.png")
    print(PLOT_FOLDER / "baseline_loss.png")


if __name__ == "__main__":
    main()