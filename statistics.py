import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(i):
    # Create arrays for all different statistical data
    learning_rate = []

    train_index = []
    train_accuracy = []
    train_data_loss = []
    train_reg_loss = []

    valid_index = []
    valid_accuracy = []
    valid_data_loss = []

    # Open statistics.json file in read mode
    with open(os.path.join(os.getcwd(), "statistics.json"), "r") as f:
        # Converts json string to python dict
        try:
            stat_json = json.loads(f.read())
        except:
            return

            # Iterate over all data points in key "train"
        for data_points in stat_json["train"]:
            # Extract all information and save them in the arrays
            train_index.append(data_points["index"])
            learning_rate.append(data_points["learning_rate"])
            train_accuracy.append(data_points["accuracy"])
            train_data_loss.append(data_points["data_loss"])
            train_reg_loss.append(data_points["reg_loss"])

        # Iterate over all data points in key "valid"
        for data_points in stat_json["valid"]:
            # Extract all information and save them in the arrays
            valid_index.append(data_points["index"])
            valid_accuracy.append(data_points["accuracy"])
            valid_data_loss.append(data_points["data_loss"])

        # Update the graphs
        plt.cla()
        # Plot all the training data
        plt.plot(train_index, learning_rate, label="learning rate")
        plt.plot(train_index, train_accuracy, label="training accuracy")
        plt.plot(train_index, np.array(train_data_loss) + np.array(train_reg_loss), label="training loss")

        # Plot all the validation data
        plt.plot(valid_index, valid_accuracy, label="validation accuracy")
        plt.plot(valid_index, valid_data_loss, label="validation loss")

        # Add legend to graph
        plt.legend(loc="upper left")
        # Close statistics.json file
        f.close()


# Animation function
# Calls animate() function every 1000 milliseconds
ani = FuncAnimation(plt.gcf(), animate, 1000)

# Show the graph
plt.show()
