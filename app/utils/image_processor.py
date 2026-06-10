import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms

import os
import cv2
import pathlib

def validate_files(directory : str) -> None:
    """
        Checks if all of the images in the file path are valid. If not it delets them.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            img_path = os.path.join(root,file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print('Deleting invalid image ' + img_path)
                    os.remove(img_path) 
            except Exception as e:
                print('Issue with image ' + img_path)
                os.remove(img_path)

def balance_data(dataset_subset, full_dataset) -> WeightedRandomSampler:
    """Returns a WeightedRandomSampler balanced across classes for a training subset."""
    
    # Get labels only for the subset indices
    labels = [full_dataset.targets[i] for i in dataset_subset.indices]
    
    # Count per class and assign inverse weights
    class_counts  = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label].item() for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
           
def process_data(directory: str, train_percent: int = 70, validation_percent: int = 20, batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Processes the image data from the specified directory, splits it into training, validation, and test sets, and returns DataLoaders for each subset."""
    print("Validating files in directory...")
    validate_files(directory)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(directory, transform=transform)

    # Split FIRST using dataset length
    total_samples = len(dataset)
    train_size = int(total_samples * train_percent * 0.01)
    val_size   = int(total_samples * validation_percent * 0.01)
    test_size  = total_samples - train_size - val_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    # Balance AFTER split, on train subset only
    print("Balancing data...")
    sampler = balance_data(train_data, dataset)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers = 4)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers = 4)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers = 4)

    return train_loader, val_loader, test_loader
          
def get_and_print_distribution(directory: str) -> dict:
    """Returns and prints class distribution and imbalance ratio from a directory."""
    
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
