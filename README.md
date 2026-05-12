# Waste-Classification

**This project is actively in development**

# Overview:
Implements a custom CNN architecture trained on 11,000 images that categorizes waste into 4 waste categories.
Implements a CNN using the transfer learning model MobileNetV2.

# Results:
Custom CNN model
| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| not_allowed | 0.90      | 0.76   | 0.82     | 625     |
| paper       | 0.77      | 0.80   | 0.78     | 1500    |
| plastic     | 0.79      | 0.59   | 0.68     | 1241    |
| trash       | 0.80      | 0.88   | 0.84     | 3024    |
| **Overall** | **0.81**  |        |          | **6390**|

# Instructions: 
- Create virtual environment.
- Download requirements using pip install -r requirements.txt.
- Uncomment the respective lines in run.py
- Run program using run.py

# Structure: 
- Model
  - data - holds the testing and training images.
  - models - contains both the custom and transfer learning model builders.
  - output - stores the logs and built models.
  - utils - stores the helper functions.
  - config.py - contains the constants.
  - main.py - entry point, builds tests and records data.
  
# Stack:
- Python, TensorFlow, Keras, OpenCV, Matplotlib

## Datasets
- https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2?select=standardized_256
- https://www.kaggle.com/datasets/jvnr1495/batteries

