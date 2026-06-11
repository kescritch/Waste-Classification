import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
import os
import cv2
import pathlib


def validate_files(directory: str) -> None:
    for root, dirs, files in os.walk(directory):
        for file in files:
            img_path = os.path.join(root, file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print('Deleting invalid image ' + img_path)
                    os.remove(img_path)
            except Exception:
                print('Issue with image ' + img_path)
                os.remove(img_path)


def balance_data(dataset_subset, full_dataset) -> WeightedRandomSampler:
    labels = [full_dataset.targets[i] for i in dataset_subset.indices]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label].item() for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def process_data(directory: str, train_percent: int = 70, validation_percent: int = 20, batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader]:
    print("Validating files in directory...")
    validate_files(directory)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Two views of the same data, different transforms
    train_dataset = datasets.ImageFolder(directory, transform=train_transform)
    eval_dataset = datasets.ImageFolder(directory, transform=eval_transform)

    total_samples = len(train_dataset)
    train_size = int(total_samples * train_percent * 0.01)
    val_size = int(total_samples * validation_percent * 0.01)
    test_size = total_samples - train_size - val_size

    # Same seed -> same split indices on both datasets
    generator = torch.Generator().manual_seed(42)
    train_data, _, _ = random_split(train_dataset, [train_size, val_size, test_size], generator=generator)

    generator = torch.Generator().manual_seed(42)
    _, val_data, test_data = random_split(eval_dataset, [train_size, val_size, test_size], generator=generator)

    print("Balancing data...")
    sampler = balance_data(train_data, train_dataset)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def get_and_print_distribution(directory: str) -> dict:
    data_dir = pathlib.Path(directory)
    class_counts = {
        folder.name: len(list(folder.rglob("*.*")))
        for folder in sorted(data_dir.iterdir())
        if folder.is_dir()
    }
    total = sum(class_counts.values())
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    print(f"\n{'Class':<20} {'Count':>6} {'%':>7}")
    print("-" * 35)
    for cls, count in sorted(class_counts.items()):
        print(f"{cls:<20} {count:>6} {count/total*100:>6.1f}%")
    print(f"\nTotal images: {total}")
    print(f"Imbalance ratio: {max_count/min_count:.2f}x")

    return class_counts