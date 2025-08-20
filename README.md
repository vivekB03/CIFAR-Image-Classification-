# CIFAR-Image-Classification-

# CIFAR-10 Image Classification Project

A deep learning project that classifies images from the CIFAR-10 dataset using a convolutional neural network (CNN) built with TensorFlow and Keras.

## Project Overview

This project implements an image classification model that can distinguish between 10 different categories of objects:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The model achieves competitive accuracy on the CIFAR-10 dataset, which consists of 60,000 32x32 color images.

## Model Architecture

The CNN architecture consists of:

**Feature Extraction Layers:**
- Conv2D (32 filters, 3x3 kernel, ReLU activation)
- MaxPooling2D (2x2 pool size)
- Conv2D (64 filters, 3x3 kernel, ReLU activation)
- MaxPooling2D (2x2 pool size)
- Conv2D (64 filters, 3x3 kernel, ReLU activation)

**Classification Layers:**
- Flatten layer
- Dense layer (64 units, ReLU activation)
- Output layer (10 units, Softmax activation)

## Dataset

The CIFAR-10 dataset is automatically downloaded when running the script. It contains:
- 50,000 training images
- 10,000 test images
- 10 classes with 6,000 images each

## Requirements

- Python 3.x
- TensorFlow 2.x
- Matplotlib
- NumPy

Install dependencies with:
```bash
pip install tensorflow matplotlib numpy
```

## Usage

1. Clone the repository
2. Run the classification script:
```bash
python classification.py
```

The script will:
- Load and preprocess the CIFAR-10 dataset
- Build and compile the CNN model
- Train the model for 10 epochs
- Evaluate model performance on test data
- Display sample predictions with visualization

## Results

After training, the model will output:
- Training progress with accuracy and loss metrics
- Final test accuracy
- Visual comparison of predictions vs actual labels for sample images

## Files

- `classification.py` - Main implementation of the CNN model and training pipeline

## Future Improvements

Potential enhancements for this project:
- Data augmentation to improve generalization
- Hyperparameter tuning for better performance
- Transfer learning with pre-trained models
- Deployment as a web application
- Integration with a custom image upload feature

## License

This project is open source and available under the [MIT License](LICENSE).
