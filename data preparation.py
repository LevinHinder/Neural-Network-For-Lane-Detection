import json
import os
import glob
import cv2
import numpy as np

path_labels = r"F:\Curvelanes\train\labels"
path_images = r"F:\Curvelanes\train\images"

for filename in glob.glob(os.path.join(path_labels, "*.json")):
    with open(os.path.join(os.getcwd(), filename), "r") as f:
        road = json.loads(f.read())

        filename = filename[len(path_labels) + 1: -11] + ".png"
        image = cv2.imread(path_images + "\\" + filename, cv2.IMREAD_GRAYSCALE)
        lane_binary = np.zeros((len(image), len(image[1])))

        for lanes in road["Lines"]:
            lane_x = []
            lane_y = []
            for coordinates in lanes:
                lane_x.append(float(coordinates["x"]))
                lane_y.append(float(coordinates["y"]))

            lane_function = []
            for i in range(len(lane_y) - 1):
                lane_function.append(np.poly1d(np.polyfit(lane_y[i:i + 2], lane_x[i: i + 2], 1)))

            i = 0
            for y in range(round(lane_y[-1]), round(lane_y[0])):
                if y > lane_y[len(lane_y) - 1 - i - 1]:
                    i += 1
                if lane_function[len(lane_function) - 1 - i](y) > len(lane_binary[0]):
                    break

                if abs(lane_function[len(lane_function) - 1 - i].coeffs[0]) > 5:
                    for a in range(-3, 4):
                        if lane_function[len(lane_function) - 1 - i].coeffs[0] >= 0:
                            for b in range(round(-2 / abs(1 / lane_function[len(lane_function) - 1 - i].coeffs[0])), 4):
                                try:
                                    lane_binary[y + a, round(lane_function[len(lane_function) - 1 - i](y)) + b] = 255
                                except Exception:
                                    pass
                        else:
                            for b in range(-3, round(2 / abs(1 / lane_function[len(lane_function) - 1 - i].coeffs[0]))):
                                try:
                                    lane_binary[y + a, round(lane_function[len(lane_function) - 1 - i](y)) + b] = 255
                                except Exception:
                                    pass

                else:
                    for a in range(-3, 4):
                        for b in range(-3, 4):
                            try:
                                lane_binary[y + a, round(lane_function[len(lane_function) - 1 - i](y)) + b] = 255
                            except Exception:
                                pass

        cv2.imwrite(r'F:\\Curvelanes\\train\\labels2\\' + filename, lane_binary)
