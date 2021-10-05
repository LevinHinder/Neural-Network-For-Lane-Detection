import numpy as np
import os
import cv2
import pickle
import copy
import glob
import json
import dill


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, weight_regulator=0., bias_regulator=0.):
        # Initialise weights and biases
        self.weights = 0.001 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularisation strength
        self.weight_regulator = weight_regulator
        self.bias_regulator = bias_regulator

    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs
        # Calculate output values
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularisation
        self.dweights += 2 * self.weight_regulator * self.weights
        self.dbiases += 2 * self.bias_regulator * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Layer_Dropout:

    def __init__(self, rate):
        # Invert the rate
        self.rate = 1 - rate

    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs

        # If not in the training mode deactivate dropout
        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


class Layer_Input:

    def forward(self, inputs, training):
        # No calculation needed. Mark inputs as output
        self.output = inputs


class Activation_ReLU:

    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs
        # Calculate output values
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Copy dvalues
        self.dinputs = dvalues.copy()
        # Gradient on values
        self.dinputs[self.inputs <= 0] = 0


class Activation_Sigmoid:

    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs
        # Calculate output values
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        # Round output values to 1 or 0
        return (outputs > 0.5) * 1


class Optimizer_Adam:

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        # Initialise optimizer settings
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        # Update learning rate before any parameter updates
        self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # Create cache array filled with 0 if they do not exist
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentum = self.beta_1 * layer.weight_momentum + (1 - self.beta_1) * layer.dweights
        layer.bias_momentum = self.beta_1 * layer.bias_momentum + (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        weight_momentum_corrected = layer.weight_momentum / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentum_corrected = layer.bias_momentum / (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Update parameters
        layer.weights -= self.current_learning_rate * weight_momentum_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= self.current_learning_rate * bias_momentum_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        # Update iterations after any parameter updates
        self.iterations += 1


class Loss:

    def regularisation_loss(self):
        regularisation_loss = 0

        # Iterate over all trainable layers to calculate regularisation loss
        for layer in self.trainable_layers:
            regularisation_loss += layer.weight_regulator * np.sum(layer.weights * layer.weights)
            regularisation_loss += layer.bias_regulator * np.sum(layer.biases * layer.biases)

        return regularisation_loss

    def remember_trainable_layers(self, trainable_layers):
        # Set/remember trainable layers
        self.trainable_layers = trainable_layers

    def calculate(self, output, y_true):
        # Calculate loss for each sample
        sample_losses = self.forward(output, y_true)
        # Calculate mean loss over all samples
        data_loss = np.mean(sample_losses)

        # Update accumulated
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += sample_losses.size

        return data_loss, self.regularisation_loss()

    def calculate_accumulated(self):
        # Calculate mean loss over whole dataset
        data_loss = self.accumulated_sum / self.accumulated_count

        return data_loss, self.regularisation_loss()

    def new_pass(self):
        # Reset variables for accumulated loss
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Loss_UnbalancedSegmentation(Loss):

    def forward(self, output, y_true):
        # Normalise output
        output = y_true - output
        # Calculate sample-wise loss
        sample_losses = 1 / 32 * (output + 2) ** 4 - (output + 2) + 1.5
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        # Gradient on values
        self.dinputs = (1 / 8 * (dvalues - y_true - 2) ** 3 + 0.5) / len(dvalues[0]) / len(dvalues)


class Accuracy:

    def calculate(self, predictions, y_true):
        # Calculate correct predicted ones and zeros
        identical = predictions == y_true
        identical_1 = identical * predictions
        identical_0 = identical * (abs(predictions - 1))

        # Calculate accuracy over all samples
        accuracy = (np.mean(identical_1) / np.mean(y_true) + np.mean(identical_0) / (1 - np.mean(y_true))) / 2

        # Update accumulated
        self.accumulated_sum += accuracy
        self.accumulated_count += 1

        return accuracy

    def calculate_accumulated(self):
        # Calculate mean accuracy over whole dataset
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy

    def new_pass(self):
        # Reset variables for accumulated accuracy
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Model:

    def __init__(self):
        # Initialise list for layers
        self.layers = []
        self.index = 1

    def add(self, layer):
        # Add objects to the model
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, stats=False):
        # Set loss, optimizer and accuracy
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = Accuracy()

        # If stats are desired create statistics.json
        if stats:
            with open(os.path.join(os.getcwd(), "statistics.json"), "w") as f:
                # Create standard dictionary
                dictionary = {
                    "train": [],
                    "valid": []
                }
                # Serializing json
                json_object = json.dumps(dictionary, indent=4)
                # Writing to statistics.json
                f.write(json_object)
                f.close()

    def dataset(self, path, res_in, res_out, poi, kernel_size, threshold=None):
        # Path of dataset
        self.path = path
        # Resolution of input and output image
        self.res_in = [res_in[0], int(res_in[1] * poi)]
        self.res_out = [res_out[0], int(res_out[1] * poi)]
        # Kernel size for blur
        self.kernel_size = kernel_size
        # Threshold for edge detection
        self.theshold = threshold
        # Pixel of interest cuts the upper part of the image
        self.poi = 1 - poi

    def image_preprocess(self, image):
        # Convert image to grayscale, cut upper part away and apply blur
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[int(image.shape[0] * self.poi) - 1:-1]
        image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)

        # If canny edge detection is desired apply it
        if self.theshold is not None:
            image = cv2.Canny(image, self.theshold[0], self.theshold[1])

        # Resize image to proper resolution, flatten image and normalise it
        image = cv2.resize(image, (self.res_in[0], self.res_in[1]))
        image = (image.reshape(-1).astype(np.float32) - 127.5) / 127.5

        return image

    def finalize(self):
        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Initialise list for trainable layers
        self.trainable_layers = []

        # Iterate over all objects and define previous and next layer
        for i in range(len(self.layers)):

            # The previous layer of the first layer is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # All layers except for the first and the last
            elif i < len(self.layers) - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # The next layer of the last layer is the loss
            # Save reference to the last object
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # Create list with trainable layers
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

    def save_stat(self, key, accuracy, data_loss, reg_loss):
        # Open statistics.json in read mode
        with open(os.path.join(os.getcwd(), "statistics.json"), "r") as f:
            # Read everything and convert json string to python dict
            stat_json = json.loads(f.read())
            f.close()

        with open(os.path.join(os.getcwd(), "statistics.json"), "w") as f:
            # Create data to be written
            dictionary = {
                "index": self.index,
                "learning_rate": self.optimizer.current_learning_rate,
                "accuracy": accuracy,
                "data_loss": data_loss,
                "reg_loss": reg_loss,
            }

            # Append new data and format it correctly
            stat_json[key].append(dictionary)
            stat_json = json.dumps(stat_json, indent=4)

            # Write to statistics.json
            f.write(stat_json)
            f.close()

    def train(self, epochs=1, batch_size=None, print_every=1):

        # ==========================================================================================================
        # F:\Curvelanes\valid\images\000a706ea929755e6bf6583ff2ce4a81.jpg
        check = cv2.imread(r"D:\Curvelanes\train\images\000a9e2091e2902d988ffdd87258fc29.jpg")
        check = self.image_preprocess(check)
        # ==========================================================================================================

        # Main training loop
        for epoch in range(1, epochs + 1):
            train_steps = 1

            # Print epoch number
            print(f"epoch: {epoch}")

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            X = []
            y = []
            path_images = fr"{self.path}\train\images"
            path_labels = fr"{self.path}\train\labels"
            list_of_files = glob.glob(fr"{path_labels}\*")
            last_file = list_of_files[-1]

            # Iterate over all images in dataset
            for filename in glob.glob(os.path.join(path_images, "*.jpg")):
                image = cv2.imread(filename)
                image = self.image_preprocess(image)
                X.append(image)

                filename = os.path.join(path_labels, filename[len(path_images) + 1:])
                label = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                label = label[int(label.shape[0] * self.poi) - 1:-1]
                label = label / 255
                label = np.round(label)
                label = cv2.resize(label, (self.res_out[0], self.res_out[1]))
                label = label.reshape(label.shape[0] * label.shape[1])
                y.append(label)

                if len(y) == batch_size or filename == last_file:
                    # Convert the data to proper numpy arrays
                    X = np.array(X)
                    y = np.array(y)

                    # Perform the forward pass
                    output = self.forward(X, training=True)

                    # Calculate loss
                    data_loss, reg_loss = self.loss.calculate(output, y)
                    loss = data_loss + reg_loss

                    # Get predictions and calculate an accuracy
                    predictions = self.output_layer_activation.predictions(output)
                    accuracy = self.accuracy.calculate(predictions, y)

                    # Perform backward pass
                    self.backward(output, y)

                    # Optimize (update parameters)
                    self.optimizer.pre_update_params()
                    for layer in self.trainable_layers:
                        self.optimizer.update_params(layer)
                    self.optimizer.post_update_params()

                    # ====================================================================================================
                    if not self.index % 1000:
                        model.save(fr"F:\Curvelanes\{epoch}.{train_steps}.model")
                    # ====================================================================================================

                    # Print a summary
                    if not train_steps % print_every or filename == last_file:
                        print(f"step: {train_steps}, acc: {accuracy:}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, "
                              f"reg_loss: {reg_loss:.3f}), lr: {self.optimizer.current_learning_rate}")

                        # Save statistical information
                        self.save_stat(key="train", accuracy=accuracy, data_loss=data_loss, reg_loss=reg_loss)

                        # ==========================================================================================================
                        predict = self.forward(check, training=False)
                        predict = self.output_layer_activation.predictions(predict)
                        predict *= 255
                        predict = predict.reshape(self.res_out[1], self.res_out[0])
                        cv2.imwrite(fr"C:\Users\Levin\Downloads\New folder (2)\{epoch}.{train_steps}.png", predict)
                        # ==========================================================================================================

                    self.index += 1
                    train_steps += 1
                    X = []
                    y = []

            # Get epoch loss and accuracy
            epoch_data_loss, epoch_reg_loss = self.loss.calculate_accumulated()
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            # Print a summary of epoch
            print(f"training, acc: {epoch_accuracy:.3f}, loss: {epoch_loss:.3f} (data_loss: {epoch_data_loss:.3f}, "
                  f"reg_loss: {epoch_reg_loss:.3f}), lr: {self.optimizer.current_learning_rate}")

            # Evaluate the model
            self.evaluate(batch_size=batch_size)

    # Evaluates the model using passed-in dataset
    def evaluate(self, batch_size=None):
        # Reset accumulated values in loss and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()

        X_val = []
        y_val = []
        path_images = fr"{self.path}\valid\images"
        path_labels = fr"{self.path}\valid\labels"
        list_of_files = glob.glob(fr"{path_labels}\*")
        last_file = list_of_files[-1]

        for filename in glob.glob(os.path.join(path_images, "*.jpg")):
            image = cv2.imread(filename)
            image = self.image_preprocess(image)
            X_val.append(image)

            filename = os.path.join(path_labels, filename[len(path_images) + 1:])
            label = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            label = label / 255
            label = np.round(label)
            label = cv2.resize(label, (self.res_out[0], self.res_out[1]))
            label = label.reshape(label.shape[0] * label.shape[1])
            y_val.append(label)

            if len(y_val) == batch_size or filename == last_file:
                # Convert the data to proper numpy arrays
                X_val = np.array(X_val)
                y_val = np.array(y_val)

                # Perform the forward pass
                output = self.forward(X_val, training=False)

                # Calculate the loss
                self.loss.calculate(output, y_val)

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                self.accuracy.calculate(predictions, y_val)

                X_val = []
                y_val = []

        # Get validation loss and accuracy
        valid_data_loss, valid_reg_loss = self.loss.calculate_accumulated()
        valid_loss = valid_data_loss
        valid_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        print(f"validation, acc: {valid_accuracy:.3f}, loss: {valid_loss:.3f}\n")

        # Save statistical information
        self.save_stat(key="valid", accuracy=valid_accuracy, data_loss=valid_data_loss, reg_loss=0)

    def predict(self, path):
        # ==========================================================================================================
        import time
        start_time = time.time()
        # ==========================================================================================================
        # Create video object
        video_input = cv2.VideoCapture(path)
        # Get width and height of video
        width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_output = cv2.VideoWriter(os.path.join(os.getcwd(), "output.mp4"), fourcc, 30, (width, height))
        # Create arrays
        X = []
        frames = []

        while True:
            # Read the video frame by frame
            ret, frame = video_input.read()
            if not ret:
                break

            # Preprocess frame and save it
            X.append(self.image_preprocess(frame))
            # Save original frame
            frames.append(frame)

        # Convert the data to proper numpy arrays
        X = np.array(X)
        frames = np.array(frames)

        # Pass the data through the network
        output = self.forward(X, training=False)
        predictions = self.output_layer_activation.predictions(output)
        # Reshape predictions to 2d array
        predictions = predictions.reshape(predictions.shape[0], self.res_out[1], self.res_out[0])
        # Add array filled with zeros that was cut away from the network
        fill = np.zeros((predictions.shape[0], int(self.res_out[1] / (1 - self.poi) * self.poi), self.res_out[0]))
        predictions = np.append(fill, predictions, axis=1)

        # Iterate over all frames
        for i in range(len(predictions)):
            # Resize network output to original video resolution
            predict = cv2.resize(predictions[i], (width, height))
            # Mark predicted pixels in video red
            frames[i, predict >= 0.5] = [0, 0, 255]
            # Write frame as video
            video_output.write(frames[i])

        # Release everything when job is finished
        video_input.release()
        video_output.release()
        # ==========================================================================================================
        end_time = time.time()
        time_lapsed = end_time - start_time
        print(time_lapsed)
        # ==========================================================================================================

    def forward(self, X, training):

        # Call forward method on the input layer to create output property
        self.input_layer.forward(X, training)

        # Call forward method of every object
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # Return output of last layer
        return layer.output

    def backward(self, output, y):

        # Call backward method on the loss to create dinputs property
        self.loss.backward(output, y)

        # Call backward method of every object in reversed order
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def save(self, path):
        # Make a deep copy of the current model
        model = copy.deepcopy(self)

        # For each layer remove properties used for training
        for layer in model.layers[:]:
            for property in ["inputs", "output", "dinputs", "dweights", "dbiases", "bias_regulator",
                             "weight_regulator", "weight_momentum", "weight_cache", "bias_momentum", "bias_cache"]:
                layer.__dict__.pop(property, None)
            if isinstance(layer, Layer_Dropout):
                model.layers.remove(layer)

        # Finalize model again without dropout layers
        model.finalize()

        # Remove unnecessary objects
        model.input_layer.__dict__.pop("output", None)
        model.__dict__.pop("output_layer_activation", None)
        model.__dict__.pop("index", None)
        model.__dict__.pop("path", None)
        model.__dict__.pop("trainable_layers", None)
        model.__dict__.pop("loss", None)
        model.__dict__.pop("accuracy", None)
        model.__dict__.pop("optimizer", None)
        model.layers[-1].__dict__.pop("next", None)

        # Open a file in the binary-write mode and save the model
        with open(path, "wb") as f:
            dill.dump(model, f)

    @staticmethod
    def load(path):
        # Open file in the binary-read mode
        with open(path, "rb") as f:
            # Load the model
            model = pickle.load(f)

        return model


# Instantiate the model
model = Model()

model.dataset(path=r"D:\Testdataset", res_in=[256, 144], res_out=[256, 144], poi=0.5, kernel_size=3)

# Add layers
model.add(Layer_Dense(model.res_in[0] * model.res_in[1], 8192, weight_regulator=0.1, bias_regulator=0.1))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.25))
model.add(Layer_Dense(8192, 8192, weight_regulator=0.1, bias_regulator=0.1))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.25))
model.add(Layer_Dense(8192, 8192, weight_regulator=0.1, bias_regulator=0.1))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.25))
model.add(Layer_Dense(8192, 8192, weight_regulator=0.1, bias_regulator=0.1))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.25))
model.add(Layer_Dense(8192, model.res_out[0] * model.res_out[1]))
model.add(Activation_Sigmoid())

# Set loss and optimizer objects
model.set(loss=Loss_UnbalancedSegmentation(),
          optimizer=Optimizer_Adam(learning_rate=0.001, decay=5e-4, epsilon=1e-7, beta_1=0.825, beta_2=0.9),
          stats=True)

# Finalize the model
model.finalize()

# Save the model
model.save(r"D:\Curvelanes\lane_detection.model")

# Train the model
model.train(epochs=100000, batch_size=4, print_every=3)

# Save the model
model.save(r"F:\Curvelanes\lane_detection.model")

model = Model.load(r"D:\Curvelanes\1.2.model")

# Use the model to predict data
model.predict(os.path.join(os.getcwd(), "test.mp4"))
