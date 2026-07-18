from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Values used throughout the script
IMAGE_SIZE = 128
BATCH_SIZE = 16


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "split"
MODEL_PATH = PROJECT_ROOT / "hw3_output" / "models" / "best_tuned_cnn.pth"
PLOT_FOLDER = PROJECT_ROOT / "hw3_output" / "plots"
RESULT_FOLDER = PROJECT_ROOT / "hw3_output" / "results"

PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


# Test images are only resized and converted to tensors
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# Same CNN architecture used during hyperparameter tuning
class FishCNN(nn.Module):
    def __init__(self, number_of_classes, dropout_rate=0.30):
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


# Collecting the true and predicted labels from the test dataset
def evaluate_model(model, loader, device):
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            true_labels.extend(labels.numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    return true_labels, predicted_labels


# Creating and saving the confusion matrix figure
def save_confusion_matrix(matrix, class_names):
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix)

    plt.title("Tuned CNN Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        np.arange(len(class_names)),
        class_names,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        np.arange(len(class_names)),
        class_names
    )

    # Displaying the number of predictions inside each cell
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center"
            )

    plt.colorbar()
    plt.tight_layout()

    plot_path = PLOT_FOLDER / "tuned_cnn_confusion_matrix.png"

    plt.savefig(plot_path)
    plt.close()

    return plot_path


def main():
    test_folder = DATA_FOLDER / "test"

    # Check that the test data and tuned model are available
    if not test_folder.exists():
        raise FileNotFoundError(
            "The test dataset was not found in data/split/test."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The tuned model was not found. Run hyperparameter_tuning.py first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Loading the untouched test dataset
    test_dataset = datasets.ImageFolder(
        test_folder,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = test_dataset.classes
    number_of_classes = len(class_names)

    # Creating the CNN and loading the saved tuned weights
    model = FishCNN(
        number_of_classes=number_of_classes,
        dropout_rate=0.30
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    # Running the model on the test dataset
    true_labels, predicted_labels = evaluate_model(
        model,
        test_loader,
        device
    )

    # Calculating the main evaluation metrics
    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Storing the evaluation results in JSON format
    results = {
        "model": "Tuned CNN",
        "selected_hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 16,
            "dropout": 0.30
        },
        "test_images": len(test_dataset),
        "classes": class_names,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_score,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report
    }

    result_path = RESULT_FOLDER / "tuned_cnn_evaluation.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    plot_path = save_confusion_matrix(
        matrix,
        class_names
    )

    # Displaying the final evaluation results
    print("Device:", device)
    print("Test images:", len(test_dataset))
    print("Classes:", class_names)

    print("\nTuned CNN Test Results")
    print("=" * 45)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1_score * 100:.2f}%")

    print("\nClassification Report")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=class_names,
            zero_division=0
        )
    )

    print("Saved files:")
    print(result_path)
    print(plot_path)


if __name__ == "__main__":
    main()