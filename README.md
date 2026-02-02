# Fashion MNIST Image Classification using CNN

This project implements an image classification model using a Convolutional Neural Network (CNN) trained on the Fashion MNIST dataset. The model is developed using TensorFlow and Keras.

## Overview

Fashion MNIST is a dataset of 28×28 grayscale images representing 10 different clothing categories. The goal of this project is to classify these images accurately using a deep learning-based CNN model.

### Model Architecture
- Convolutional layers for feature extraction
- MaxPooling layers for dimensionality reduction
- Fully connected (Dense) layers
- Softmax activation for multi-class classification

### Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Dataset
- Fashion MNIST (available via keras.datasets)

### Training Results
- Epochs: 26
- Training Accuracy: 98.23%
- Training Loss: 0.0474
- Validation Accuracy: 90.73%
- Validation Loss: 0.4970

### Classification Results
Overall Test Accuracy: 91%
Macro Average Recall: 0.91

## Observations

The model performs strongly on classes such as Trouser, Sandal, Bag, and Ankle boot.

Lower recall for Shirt, Coat, and T-shirt/top is due to visual similarity between these categories.

## Conclusion

The CNN model achieves high training accuracy and demonstrates good generalization on the test dataset. This project shows the effectiveness of convolutional neural networks for image classification tasks and serves as a solid introduction to deep learning with TensorFlow and Keras.
