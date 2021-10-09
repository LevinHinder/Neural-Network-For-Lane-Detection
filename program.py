import os
import sys
import subprocess
import pkg_resources


# Print iterations progress
def printProgressBar(iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█", printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


class Layer_Dense:

    def forward(self, inputs, training):
        # Calculate output values
        self.output = np.dot(inputs, self.weights) + self.biases


class Layer_Input:

    def forward(self, inputs, training):
        # No calculation needed. Mark inputs as output
        self.output = inputs


class Activation_ReLU:

    def forward(self, inputs, training):
        # Calculate output values
        self.output = np.maximum(0, inputs)


class Activation_Sigmoid:

    def forward(self, inputs, training):
        # Calculate output values
        self.output = 1 / (1 + np.exp(-inputs))

    def predictions(self, outputs):
        # Round output values to 1 or 0
        return (outputs > 0.5) * 1


class Model:

    def image_preprocess(self, image):
        # Convert image to grayscale, cut upper part away and apply blur
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[int(image.shape[0] * self.poi) - 1:-1]
        image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)

        # Resize image to proper resolution, flatten image and normalise it
        image = cv2.resize(image, (self.res_in[0], self.res_in[1]))
        image = (image.reshape(-1).astype(np.float32) - 127.5) / 127.5

        return image

    def predict(self, path):
        # Check whether path is valid
        if not os.path.exists(path):
            print("Invalid path!\n")
            return

        # Create video object
        video_input = cv2.VideoCapture(path)
        # Get filename of video
        index = np.maximum(path.rfind("\\"), path.rfind("/"))
        filename = path[index + 1:path.rfind(".")]
        # Get width and height of video
        width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Get number of frames
        length = int(video_input.get(cv2.CAP_PROP_FRAME_COUNT))
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_output = cv2.VideoWriter(os.path.join(os.path.dirname(sys.argv[0]), filename + "_output.mp4"), fourcc, 30, (width, height))
        # Create arrays
        X = []
        frames = []

        # Calculate new max array size
        max_array_size = int(psutil.virtual_memory().available / height / width / 5)

        process = 0
        printProgressBar(process, length + 1, prefix="Progress:", suffix="Complete", length=50)

        while True:
            # Read the video frame by frame
            ret, frame = video_input.read()
            if not ret:
                break

            # Preprocess frame and save it
            X.append(self.image_preprocess(frame))
            # Save original frame
            frames.append(frame)

            if not len(X) % max_array_size or process + len(X) == length:
                # Convert the data to proper numpy arrays
                X = np.array(X)
                frames = np.array(frames)

                # Pass the data through the network
                output = self.forward(X, training=False)
                predictions = self.layers[-1].predictions(output)
                # Reshape predictions to 2d array
                predictions = predictions.reshape(predictions.shape[0], self.res_out[1], self.res_out[0])
                # Add array filled with zeros that was cut away from the network
                fill = np.zeros(
                    (predictions.shape[0], int(self.res_out[1] / (1 - self.poi) * self.poi), self.res_out[0]))
                predictions = np.append(fill, predictions, axis=1)

                # Iterate over all frames
                for i in range(len(predictions)):
                    # Resize network output to original video resolution
                    predict = cv2.resize(predictions[i], (width, height))
                    # Mark predicted pixels in video red
                    frames[i, predict >= 0.5] = [0, 0, 255]
                    # Write frame as video
                    video_output.write(frames[i])
                    # Increment process and print progressbar
                    process += 1
                    printProgressBar(process, length + 1, prefix="Progress:", suffix="Complete", length=50)

                # Clear arrays
                X = []
                frames = []

                # Calculate new max array size
                max_array_size = int(psutil.virtual_memory().available / height / width / 5)

        # Release everything when job is finished
        video_input.release()
        video_output.release()
        printProgressBar(length + 1, length + 1, prefix="Progress:", suffix="Complete", length=50)
        print("\n")

    def forward(self, X, training):

        # Call forward method on the input layer to create output property
        self.input_layer.forward(X, training)

        # Call forward method of every object
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # Return output of last layer
        return layer.output

    @staticmethod
    def load(path):
        # Open file in the binary-read mode
        with open(path, "rb") as f:
            # Load the model
            model = dill.load(f)

        return model


print("Starting up...")

# Required libraries
required = {"numpy", "opencv-python", "dill", "psutil", "pip", "setuptools", "wheel"}

print("Checking for libraries...")
# Install all libraries
subprocess.check_call([sys.executable, "-m", "pip", "install", *required])
print("\n")

print("Checking for updates...")
# Update all libraries
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *required])
print("\n")

import cv2
import numpy as np
import dill
import psutil

print("Loading model...")
# Load the model
model = Model.load(os.path.join(os.path.dirname(sys.argv[0]), "lane_detection.model"))
print("\n")

# Clear the console
clear_console()

while True:
    # Get path of video as input
    path = input("path: ")

    # Exit while loop
    if path == "exit":
        break

    # Use the model to predict data
    model.predict(path)
