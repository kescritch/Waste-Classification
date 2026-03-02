# import tensorflow as tf
import os
# import cv2

extensions = {'jpeg', 'jpg', 'png', 'bmp'}
root_dir = "../../training_data"

if os.path.exists(root_dir):
    print(True)
else:
    print(False)
for mainfolder, classification, folder in os.walk(root_dir):
    print("Processing folder:", folder)