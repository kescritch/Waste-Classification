# Waste-Classification

**This project is actively in development**

# Overview:
Custom CNN architecture trained on 11,000 images that categorizes waste into 4 waste categories.
Currently experimenting with transfer learning with a goal accuracy of 95%.

# Instructions: 
- Create virtual environment.
- Download requirements using pip install -r requirements.txt.

# Structure: 
- main.py: Main file for initialization.
- cnn_V1.py: Creates the ground up CNN model and returns training history. Prints summary of model when run.
- cnn_v2.py: Creates a CNN that uses transfer learning and returns training history. Prints a summary of model when run.
- load.py: Seperates Images into training, validation, and testing batches. Prints distribution of data when run. 
  
  # Results:
| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| not_allowed | 0.90      | 0.76   | 0.82     | 625     |
| paper       | 0.77      | 0.80   | 0.78     | 1500    |
| plastic     | 0.79      | 0.59   | 0.68     | 1241    |
| trash       | 0.80      | 0.88   | 0.84     | 3024    |
| **Overall** | **0.81**  |        |          | **6390**|

- charts from testing are included in plots

# Stack:
- Python, TensorFlow, Keras, OpenCV, Matplotlib

## Datasets
- https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2?select=standardized_256
- https://www.kaggle.com/datasets/jvnr1495/batteries
- https://www.kaggle.com/datasets/adithyachalla/waste-classification?select=1-Cardboard
