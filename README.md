# Waste-Classification

**This project is actively in development**

# Overview:
Custom CNN architecture trained on 11,000 images that categorizes waste into 4 waste categories.
Currently experimenting with transfer learning with a goal accuracy of 95%.

#Instructions: 
- Create virtual environment.
- Download requirements using pip install -r requirements.txt.

#Structure: 
- main.py: Main file for initialization.
- cnn.py: Creates the CNN model and returns training history. Prints string summary of model when ran.
- load.py: Seperates Images into training, validation, and testing batches. Prints distribution of data when run. 
  
# Stack:
- Python, TensorFlow, Keras, OpenCV, NumPy

## Datasets
- https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2?select=standardized_256
- https://www.kaggle.com/datasets/jvnr1495/batteries
- https://www.kaggle.com/datasets/adithyachalla/waste-classification?select=1-Cardboard
