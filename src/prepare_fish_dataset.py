from pathlib import Path
import json
import random
import shutil

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


# Using the same seed keeps the dataset split unchanged each time.
RANDOM_SEED = 42

# Images will be resized before they are saved in the split folders.
IMAGE_SIZE = (128, 128)

# These percentages give a 70/15/15 train, validation, and test split.
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_dataset_folder(project_root: Path) -> Path:
    """Find the folder that directly contains the fish class folders."""

    possible_locations = [
        project_root / "data" / "Fish",
        project_root / "data" / "Fish" / "Fish",
    ]

    for folder in possible_locations:
        if not folder.exists():
            continue

        class_folders = [item for item in folder.iterdir() if item.is_dir()]

        if len(class_folders) >= 2:
            return folder

    raise FileNotFoundError(
        "The fish dataset could not be found. Place the class folders "
        "inside data/Fish/ and run the script again."
    )


def collect_images(dataset_folder: Path):
    """Collect valid image paths and their matching class names."""

    image_paths = []
    labels = []
    skipped_files = []

    class_folders = sorted(
        [folder for folder in dataset_folder.iterdir() if folder.is_dir()]
    )

    if not class_folders:
        raise ValueError("No class folders were found in the dataset.")

    for class_folder in class_folders:
        for file_path in sorted(class_folder.iterdir()):
            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            try:
                # Opening and verifying each image helps detect damaged files.
                with Image.open(file_path) as image:
                    image.verify()

                image_paths.append(file_path)
                labels.append(class_folder.name)

            except (UnidentifiedImageError, OSError):
                skipped_files.append(str(file_path))

    return image_paths, labels, skipped_files


def create_split(image_paths, labels):
    """Create stratified training, validation, and testing splits."""

    train_paths, remaining_paths, train_labels, remaining_labels = (
        train_test_split(
            image_paths,
            labels,
            test_size=(VALIDATION_SIZE + TEST_SIZE),
            random_state=RANDOM_SEED,
            stratify=labels,
        )
    )

    # The remaining 30% is divided equally into validation and test data.
    validation_fraction = VALIDATION_SIZE / (
        VALIDATION_SIZE + TEST_SIZE
    )

    validation_paths, test_paths, validation_labels, test_labels = (
        train_test_split(
            remaining_paths,
            remaining_labels,
            test_size=(1 - validation_fraction),
            random_state=RANDOM_SEED,
            stratify=remaining_labels,
        )
    )

    return {
        "train": (train_paths, train_labels),
        "validation": (validation_paths, validation_labels),
        "test": (test_paths, test_labels),
    }


def save_split(split_data, split_folder: Path):
    """Resize and save the images in their new split folders."""

    if split_folder.exists():
        shutil.rmtree(split_folder)

    split_counts = {}

    for split_name, (paths, labels) in split_data.items():
        split_counts[split_name] = {}

        for image_path, class_name in zip(paths, labels):
            class_output_folder = split_folder / split_name / class_name
            class_output_folder.mkdir(parents=True, exist_ok=True)

            output_path = class_output_folder / image_path.name

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image = image.resize(IMAGE_SIZE)
                    image.save(output_path)

                split_counts[split_name][class_name] = (
                    split_counts[split_name].get(class_name, 0) + 1
                )

            except (UnidentifiedImageError, OSError) as error:
                print(f"Could not save {image_path.name}: {error}")

    return split_counts


def print_summary(split_counts):
    """Print the number of images saved for each class and split."""

    print("\nDataset split summary")
    print("-" * 45)

    for split_name, class_counts in split_counts.items():
        split_total = sum(class_counts.values())

        print(f"\n{split_name.capitalize()}: {split_total} images")

        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")


def main():
    random.seed(RANDOM_SEED)

    project_root = Path(__file__).resolve().parent.parent
    split_folder = project_root / "data" / "split"
    results_folder = project_root / "hw3_output" / "results"
    results_folder.mkdir(parents=True, exist_ok=True)

    dataset_folder = find_dataset_folder(project_root)

    print(f"Dataset folder: {dataset_folder}")

    image_paths, labels, skipped_files = collect_images(dataset_folder)

    print(f"Valid images found: {len(image_paths)}")
    print(f"Classes found: {len(set(labels))}")

    if skipped_files:
        print(f"Unreadable files skipped: {len(skipped_files)}")
    else:
        print("No unreadable images were found.")

    split_data = create_split(image_paths, labels)
    split_counts = save_split(split_data, split_folder)

    print_summary(split_counts)

    summary = {
        "random_seed": RANDOM_SEED,
        "image_size": list(IMAGE_SIZE),
        "total_valid_images": len(image_paths),
        "classes": sorted(set(labels)),
        "skipped_files": skipped_files,
        "split_counts": split_counts,
    }

    summary_path = results_folder / "dataset_split_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"\nSplit images saved to: {split_folder}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()