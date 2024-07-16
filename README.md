# Sign Language Detection

This repository contains a Sign Language Detection project that uses machine learning to recognize and classify sign language gestures. The project consists of four files that should be run in order to collect and preprocess data, train a model, and make predictions.

## Files

### 1. `collect_images.py`

This script is used to collect images of sign language gestures. It captures images from a webcam and saves them to a directory.

### 2. `create_dataset.py`

This script preprocesses the collected images and creates a dataset that can be used to train a machine learning model.

### 3. `train_model.py`

This script trains a machine learning model using the preprocessed dataset.

### 4. `inference_classifier.py`

This script uses the trained model to make predictions on new images.

## Running the Project

To run the project, follow these steps:

1. Run `collect_images.py` to collect images of sign language gestures.
2. Run `create_dataset.py` to preprocess the collected images and create a dataset.
3. Run `train_model.py` to train a machine learning model using the dataset.
4. Run `inference_classifier.py` to make predictions on new images using the trained model.

## Requirements

* Python 3.x
* OpenCV library
* TensorFlow or Keras library
* Webcam or camera device

