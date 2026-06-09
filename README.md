# Waste-Classification
**This project is actively in development**

# Overview
Implements a custom CNN architecture trained on 11,000 images that categorizes waste into 4 waste categories.
Implements a CNN using transfer learning with MobileNetV2.

# Results

### Custom CNN (v1)
| Metric       | Value  |
|--------------|--------|
| Max Accuracy | 0.88   |
| Max Val Acc  | 0.83   |
| Min Loss     | 0.32   |
| Min Val Loss | 0.49   |

![v1 Loss and Accuracy](app/output/plots/loss_plot_and_accuracy_plot_2026-06-09_12-27-30.png)

---

### Transfer Learning MobileNetV2 (v2)
| Metric       | Value  |
|--------------|--------|
| Max Accuracy | 0.93   |
| Max Val Acc  | 0.93   |
| Min Loss     | 0.20   |
| Min Val Loss | 0.20   |

![v2 Loss and Accuracy](app\output\plots\loss_plot_and_accuracy_plot_2026-06-09_11-25-30.png)

---

# Instructions
- Create a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # Linux/macOS
  venv\Scripts\activate     # Windows
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Uncomment the respective lines in `run.py`. When training a new model, adjust the number of epochs in `config.py`.
- Run the program:
  ```bash
  python run.py
  ```

# Structure
```
Waste-Classification/
├── run.py                  # Main entry point
└── app/
    ├── config.py           # Constants (epochs, batch size, class names, paths)
    ├── main.py             # Main model logic
    ├── data/               # Training and testing images
    ├── models/             # CNN models (v1 custom, v2 MobileNetV2)
    ├── output/
    │   ├── logs/           # Training logs
    │   ├── models/         # Saved .pth model files
    │   └── plots/          # Loss and accuracy plots
    ├── camera/             # Webcam inference runner
    └── utils/              # Helper functions (model loading, plotting, etc.)
```

# Stack
- Python, PyTorch, OpenCV, Matplotlib, torchvision, torchmetrics

## Datasets
- https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2?select=standardized_256
- https://www.kaggle.com/datasets/jvnr1495/batteries