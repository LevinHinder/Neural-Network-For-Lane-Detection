import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(i):
    learning_rate = []

    train_index = []
    train_accuracy = []
    train_data_loss = []
    train_reg_loss = []

    valid_index = []
    valid_accuracy = []
    valid_data_loss = []
    valid_reg_loss = []

    # Open statistics.json file in read mode
    with open(os.path.join(os.getcwd(), "statistics.json"), "r") as f:
        # Converts json string to python dict
        stat_json = json.loads(f.read())

        for data_points in stat_json["train"]:
            train_index.append(data_points["index"])
            learning_rate.append(data_points["learning_rate"])
            train_accuracy.append(data_points["accuracy"])
            train_data_loss.append(data_points["data_loss"])
            train_reg_loss.append(data_points["reg_loss"])

        for data_points in stat_json["valid"]:
            valid_index.append(data_points["index"])
            valid_accuracy.append(data_points["accuracy"])
            valid_data_loss.append(data_points["data_loss"])
            valid_reg_loss.append(data_points["reg_loss"])


        plt.cla()
        plt.plot(train_index, learning_rate, label="learning rate")
        plt.plot(train_index, train_accuracy, label="training accuracy")
        plt.plot(train_index, np.array(train_data_loss) + np.array(train_reg_loss), label="training loss")
    
        '''plt.plot(valid_index, valid_accuracy_accuracy)
        plt.plot(valid_index, valid_data_loss + valid_reg_loss)'''
    
        plt.legend(loc="upper left")
        f.close()


ani = FuncAnimation(plt.gcf(), animate, 1000)

plt.show()
