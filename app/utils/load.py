import tensorflow as tf
import os
import cv2
import pathlib

from app.config import ROOT_DIR

def validate_files(directory : str):
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
                
def get_class_distribution(directory: str) -> dict:
    """Returns a dict of {class_name: count} for each subfolder."""
    data_dir = pathlib.Path(directory)
    return {
        folder.name: len(list(folder.rglob("*.*")))
        for folder in sorted(data_dir.iterdir())
        if folder.is_dir()
    }
    
def process_data(directory : str, train_percent : int=70, validation_percent : int=20, batch_size:int = 32) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: 
    """
    Processes image data from a directory and splits it into training, validation and test sets.
    """
    
    data = tf.keras.utils.image_dataset_from_directory(directory, batch_size=None, image_size=(254, 254))   # loading the data  
    
    data = data.map(lambda x,y: (x/255, y))                         # rescaling to values betweeen 1 and 0
        
    '''
    0 - not_allowed
    1 - paper
    2 - plastic
    3 - trash
    '''
    
    # Splitting the data
    total_samples = len(data)
    train_size = int(total_samples * train_percent * 0.01)
    val_size = int(total_samples * validation_percent * 0.01)
    test_size = total_samples - train_size - val_size

    train = data.take(train_size).batch(batch_size).prefetch(1)
    val = data.skip(train_size).take(val_size).batch(batch_size).prefetch(1)
    test = data.skip(train_size + val_size).take(test_size).batch(batch_size).prefetch(1)
    
    return train, val, test

def print_distribution(class_counts: dict):
    """Prints class distribution and imbalance ratio."""
    total = sum(class_counts.values())
    print(f"\n{'Class':<20} {'Count':>6} {'%':>7}")
    print("-" * 35)
    for cls, count in sorted(class_counts.items()):
        print(f"{cls:<20} {count:>6} {count/total*100:>6.1f}%")
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    print(f"\nTotal images: {total}")
    print(f"Imbalance ratio: {max_count/min_count:.2f}x")
    
def main():
    root_dir = ROOT_DIR
        
    validate_files(root_dir)   
    class_counts = get_class_distribution(root_dir)
    print_distribution(class_counts)

    train, val, test = process_data(root_dir)
    
    
    
if __name__ == "__main__":
    main()