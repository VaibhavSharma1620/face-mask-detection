# Face Mask Detection Using Deep Learning 🧑‍⚕️🔍

## Project Overview

This project implements a robust face mask detection system using deep learning, specifically a fine-tuned ResNet18 neural network. The application can detect and classify faces in images based on mask-wearing status with high accuracy.

more examples in examples folder
![Project Demo](/examples/result8.jpg)
![Project Demo](/examples/result6.jpg)

## Key Features

- 🔬 **Advanced Face Detection**: Utilizes RetinaFace for precise face detection
- 🤖 **Deep Learning Classification**: Classifies mask status with high accuracy
- 🎨 **Visual Feedback**: Annotates images with bounding boxes and status labels
- 📊 **Three Mask Status Categories**:
  - With Mask (Green)
  - Without Mask (Red)
  - Mask Worn Incorrectly (Yellow)

## Performance Metrics

- **Training Accuracy**: 99.65%
- **Validation Accuracy**: 96.96%

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Minimum 8GB RAM

## Installation

1. Clone the repository:
```bash
git clone https://github.com/VaibhavSharma1620/face-mask-detection.git
cd face-mask-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset and Pre-trained Model

🗂️ **Dataset and Model Location**: 
The dataset and pre-trained model are hosted on Google Drive. 
- [Dataset Link](https://drive.google.com/path/to/dataset)
- [Pre-trained Model Link](https://drive.google.com/path/to/model)

## Usage

### Inference
```python
python inference.py --image path/to/your/image.jpg
```

### Training
```python
python train.py
```

## Project Structure

```
face-mask-detection/
│
├── maskdetection.ipynb  # combined training and inference script
├── train.py            # Model training script
├── inference.py        # Inference and detection script
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Technical Details

- **Model**: ResNet18 with custom final layer
- **Loss Function**: Focal Loss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Early Stopping**: Implemented with patience of 10 epochs

## Potential Applications

- Healthcare monitoring
- Public safety
- Workplace safety compliance
- Pandemic response management


## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/VaibhavSharma1620/face-mask-detection/issues).

## Acknowledgements

- Inspired by global pandemic response efforts
- Uses RetinaFace for face detection
