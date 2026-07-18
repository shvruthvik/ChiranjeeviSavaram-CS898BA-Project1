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
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "split"
MODEL_FOLDER = PROJECT_ROOT / "hw3_output" / "models"
PLOT_FOLDER = PROJECT_ROOT / "hw3_output" / "plots"
RESULT_FOLDER = PROJECT_ROOT / "hw3_output" / "results"

BASELINE_MODEL_PATH = MODEL_FOLDER / "baseline_cnn.pth"
TUNED_MODEL_PATH = MODEL_FOLDER / "best_tuned_cnn.pth"
BASELINE_RESULTS_PATH = RESULT_FOLDER / "baseline_cnn_results.json"
TUNING_RESULTS_PATH = RESULT_FOLDER / "tuning_results.json"

PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


# Test images are resized, converted to tensors, and normalized.
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])


# Same CNN architecture used during baseline training and tuning.
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


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required results file was not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# Collect true and predicted labels from the test dataset.
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


def calculate_metrics(true_labels, predicted_labels, class_names):
    accuracy = accuracy_score(true_labels, predicted_labels)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    matrix = confusion_matrix(true_labels, predicted_labels)

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_score,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report
    }


def save_confusion_matrix(matrix, class_names, model_name, filename):
    matrix = np.asarray(matrix)

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix)

    plt.title(f"{model_name} Confusion Matrix")
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

    plot_path = PLOT_FOLDER / filename
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def save_metric_comparison(baseline_metrics, tuned_metrics):
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]

    baseline_values = [
        baseline_metrics["accuracy"] * 100,
        baseline_metrics["precision_macro"] * 100,
        baseline_metrics["recall_macro"] * 100,
        baseline_metrics["f1_macro"] * 100
    ]

    tuned_values = [
        tuned_metrics["accuracy"] * 100,
        tuned_metrics["precision_macro"] * 100,
        tuned_metrics["recall_macro"] * 100,
        tuned_metrics["f1_macro"] * 100
    ]

    positions = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(9, 6))
    plt.bar(positions - width / 2, baseline_values, width, label="Baseline CNN")
    plt.bar(positions + width / 2, tuned_values, width, label="Tuned CNN")

    plt.xlabel("Metric")
    plt.ylabel("Score (%)")
    plt.title("Baseline vs Tuned CNN Test Performance")
    plt.xticks(positions, metric_names)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    plot_path = PLOT_FOLDER / "baseline_vs_tuned_metrics.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def save_history_comparison(baseline_history, tuned_history):
    baseline_epochs = range(1, len(baseline_history["train_accuracy"]) + 1)
    tuned_epochs = range(1, len(tuned_history["train_accuracy"]) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(
        baseline_epochs,
        baseline_history["train_accuracy"],
        label="Baseline Training"
    )
    plt.plot(
        baseline_epochs,
        baseline_history["validation_accuracy"],
        label="Baseline Validation"
    )
    plt.plot(
        tuned_epochs,
        tuned_history["train_accuracy"],
        label="Tuned Training"
    )
    plt.plot(
        tuned_epochs,
        tuned_history["validation_accuracy"],
        label="Tuned Validation"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline and Tuned CNN Accuracy Curves")
    plt.legend()
    plt.tight_layout()

    accuracy_path = PLOT_FOLDER / "baseline_vs_tuned_accuracy_curves.png"
    plt.savefig(accuracy_path)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(
        baseline_epochs,
        baseline_history["train_loss"],
        label="Baseline Training"
    )
    plt.plot(
        baseline_epochs,
        baseline_history["validation_loss"],
        label="Baseline Validation"
    )
    plt.plot(
        tuned_epochs,
        tuned_history["train_loss"],
        label="Tuned Training"
    )
    plt.plot(
        tuned_epochs,
        tuned_history["validation_loss"],
        label="Tuned Validation"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline and Tuned CNN Loss Curves")
    plt.legend()
    plt.tight_layout()

    loss_path = PLOT_FOLDER / "baseline_vs_tuned_loss_curves.png"
    plt.savefig(loss_path)
    plt.close()

    return accuracy_path, loss_path


def print_model_results(model_name, metrics, class_names, true_labels, predicted_labels):
    print(f"\n{model_name} Test Results")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision_macro'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall_macro'] * 100:.2f}%")
    print(f"F1-score:  {metrics['f1_macro'] * 100:.2f}%")

    print("\nClassification Report")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=class_names,
            zero_division=0
        )
    )


def main():
    test_folder = DATA_FOLDER / "test"

    required_paths = [
        test_folder,
        BASELINE_MODEL_PATH,
        TUNED_MODEL_PATH,
        BASELINE_RESULTS_PATH,
        TUNING_RESULTS_PATH
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file or folder was not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    baseline_training_results = load_json(BASELINE_RESULTS_PATH)
    tuning_results = load_json(TUNING_RESULTS_PATH)
    best_experiment = tuning_results["best_experiment"]

    baseline_model = FishCNN(
        number_of_classes=number_of_classes,
        dropout_rate=0.30
    ).to(device)

    tuned_model = FishCNN(
        number_of_classes=number_of_classes,
        dropout_rate=best_experiment["dropout"]
    ).to(device)

    baseline_model.load_state_dict(
        torch.load(
            BASELINE_MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    tuned_model.load_state_dict(
        torch.load(
            TUNED_MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    baseline_true, baseline_predicted = evaluate_model(
        baseline_model,
        test_loader,
        device
    )

    tuned_true, tuned_predicted = evaluate_model(
        tuned_model,
        test_loader,
        device
    )

    baseline_metrics = calculate_metrics(
        baseline_true,
        baseline_predicted,
        class_names
    )

    tuned_metrics = calculate_metrics(
        tuned_true,
        tuned_predicted,
        class_names
    )

    results = {
        "test_images": len(test_dataset),
        "classes": class_names,
        "normalization": {
            "mean": NORMALIZE_MEAN,
            "std": NORMALIZE_STD
        },
        "baseline_model": {
            "selected_hyperparameters": {
                "learning_rate": baseline_training_results["learning_rate"],
                "batch_size": baseline_training_results["batch_size"],
                "dropout": 0.30
            },
            **baseline_metrics
        },
        "tuned_model": {
            "selected_hyperparameters": {
                "learning_rate": best_experiment["learning_rate"],
                "batch_size": best_experiment["batch_size"],
                "dropout": best_experiment["dropout"]
            },
            **tuned_metrics
        }
    }

    result_path = RESULT_FOLDER / "baseline_vs_tuned_evaluation.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    baseline_matrix_path = save_confusion_matrix(
        baseline_metrics["confusion_matrix"],
        class_names,
        "Baseline CNN",
        "baseline_cnn_confusion_matrix.png"
    )

    tuned_matrix_path = save_confusion_matrix(
        tuned_metrics["confusion_matrix"],
        class_names,
        "Tuned CNN",
        "tuned_cnn_confusion_matrix.png"
    )

    metric_plot_path = save_metric_comparison(
        baseline_metrics,
        tuned_metrics
    )

    accuracy_plot_path, loss_plot_path = save_history_comparison(
        baseline_training_results["history"],
        best_experiment["history"]
    )

    print("Device:", device)
    print("Test images:", len(test_dataset))
    print("Classes:", class_names)

    print_model_results(
        "Baseline CNN",
        baseline_metrics,
        class_names,
        baseline_true,
        baseline_predicted
    )

    print_model_results(
        "Tuned CNN",
        tuned_metrics,
        class_names,
        tuned_true,
        tuned_predicted
    )

    print("\nSelected tuned hyperparameters")
    print("=" * 50)
    print("Learning rate:", best_experiment["learning_rate"])
    print("Batch size:", best_experiment["batch_size"])
    print("Dropout:", best_experiment["dropout"])

    print("\nSaved files:")
    print(result_path)
    print(baseline_matrix_path)
    print(tuned_matrix_path)
    print(metric_plot_path)
    print(accuracy_plot_path)
    print(loss_plot_path)


if __name__ == "__main__":
    main()