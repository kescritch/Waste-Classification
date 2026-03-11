import tensorflow as tf
import os
import cv2

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

def process_data(directory : str, train_percent : int=70, validation_percent : int=20, test_percent : int=10) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: 
    """
    Processes image data from a directory and splits it into training, validation and test sets.
    """
    
    data = tf.keras.utils.image_dataset_from_directory(directory)   # loading the data  
    
    data = data.map(lambda x,y: (x/255, y))                         # rescaling to values betweeen 1 and 0
    data_iterator = data.as_numpy_iterator()                        #iterates through the data
    batch = next(data_iterator)
    
    '''
    0 - not_allowed
    1 - paper
    2 - plastic
    3 - trash
    '''
    
    # Splitting the data
    train_size = int(len(data) * train_percent * .01)
    val_size = int(len(data) * validation_percent * .01) + 1
    test_size = int(len(data) * test_percent* .01) + 1

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    
    return train, val, test

def main():
    root_dir = 'training_data'
        
    validate_files(root_dir)   
    process_data(root_dir)
    
if __name__ == "__main__":
    main()