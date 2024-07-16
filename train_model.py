# This script loads the data and labels from a pickle file,
# splits the data into training and testing sets,
# trains a random forest classifier on the training data,
# predicts the labels for the testing data,
# calculates the accuracy of the model,
# and saves the model to a pickle file.

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# Load the data and labels from a pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract the data and labels from the dictionary
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data and labels into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train a random forest classifier on the training data
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict the labels for the testing data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)

# Print the accuracy of the model
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model to a pickle file
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
