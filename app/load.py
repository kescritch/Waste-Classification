import tensorflow as tf
import os
import cv2
from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
    
extensions = {'jpeg', 'jpg', 'png', 'bmp'}
root_dir = "training_data"

#checking if all of the files are valid
for root, dirs, files in os.walk(root_dir):
    for file in files:
        img_path = os.path.join(root,file)
        try:
            img = cv2.imread(img_path)
            category = os.path.basename(root)
        except Exception as e:
            print('Issue with image ' + img_path)

# loading the data  
data = tf.keras.utils.image_dataset_from_directory(root_dir)
# rescaling to values betweeen 1 and 0
data = data.map(lambda x,y: (x/255, y))
data_iterator = data.as_numpy_iterator()
batch = next(data_iterator)

# 0 - not_allowed
# 1 - paper
# 2 - plastic
# 3 - trash

# Splitting the data
train_size = int(len(data) * .7)
val_size = int(len(data) * .2) + 1
test_size = int(len(data) * .1) + 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

