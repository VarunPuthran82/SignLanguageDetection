import os
import pickle

# Import mediapipe library for hand landmark detection
import mediapipe as mp

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Import OpenCV for image processing
import cv2

# Import matplotlib for plotting images
import matplotlib.pyplot as plt

# Importing hand landmark detection module
mp_hands = mp.solutions.hands

# Importing drawing utilities module
mp_drawing = mp.solutions.drawing_utils

# Importing drawing styles module
mp_drawing_styles = mp.solutions.drawing_styles

# Initializing hand landmark detection model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory to store the dataset
DATA_DIR = './data'

# List to store the hand landmark data
data = []

# List to store the class labels
labels = []

# Iterate over each directory in the data directory
for dir_ in os.listdir(DATA_DIR):
    # Iterate over each image in the directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        # Read the image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        
        # Convert the image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image using the hand landmark detection model
        results = hands.process(img_rgb)
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Iterate over each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Iterate over each landmark and calculate the offset with respect to minimum x and y values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append the hand landmark data to the main data list
            data.append(data_aux)
            
            # Append the class label to the main labels list
            labels.append(dir_)

# # Open a file to store the data and labels
# f = open('data.pickle', 'wb')

# # Dump the data and labels into the file
# pickle.dump({'data': data, 'labels': labels}, f)

# # Close the file
# f.close()

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Clean up
hands.close()
