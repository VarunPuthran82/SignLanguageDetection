import os
from pathlib import Path
import cv2


# Directory to store the dataset
DATA_DIR = './data'

# Create the directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes in the dataset
number_of_classes = 3

# Size of each class's dataset
dataset_size = 100

# Open the webcam
cap = cv2.VideoCapture(0)

# Loop over each class
for j in range(number_of_classes):
    # Create a directory for the class if it doesn't exist
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Print a message indicating which class is being collected
    print('Collecting data for class {}'.format(j))

    # Flag to indicate when collection is done
    done = False
    try: 
        # Loop until 'q' is pressed to start collection
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        # Loop to collect images for the class
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)

            # Save the image to the class's directory
            cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

            counter += 1
    except:
        print("Not possible")

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
