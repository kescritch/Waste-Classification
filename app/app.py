import tensorflow as tf
import os
import cv2

extensions = {'jpeg', 'jpg', 'png', 'bmp'}
root_dir = "training_data"

if os.path.exists(root_dir):
    print(True)
else:
    print(False)

#checking if all of the fiels are valid
for root, dirs, files in os.walk(root_dir):
    for file in files:
        img_path = os.path.join(root,file)
        try:
            img = cv2.imread(img_path)
            category = os.path.basename(root)
        except Exception as e:
            print('Issue with image ' + img_path)
          
data = tf.keras.utils.image_dataset_from_directory(root_dir)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

print(batch[0].shape)

