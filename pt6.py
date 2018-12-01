import time

import numpy as np
import pandas as pd
import tensorflow as tf
from neupy.algorithms.competitive.lvq import LVQ2
from sklearn.metrics import accuracy_score as acs
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm


class NeuralNetwork:
    def __init__(self):
        self.X_train = np.asarray([])  # Training data
        self.y_train = np.asarray([])  # Training data (labels)
        self.X_test = np.asarray([])  # Testing data
        self.y_test = np.asarray([])  # Testing data (labels)

        self.sensor_name = "sensor"  # accelerometer or gyroscope or sensor
        self.device = "phone"  # phone or watch
        self.path_merged = "../Dataset/merged/"
        self.train_filename = self.device + "_" + self.sensor_name + "_features_reduced_train_lr.xlsx"
        self.test_filename = self.device + "_" + self.sensor_name + "_features_reduced_test_lr.xlsx"

        # Backpropagation specific
        self.backpropagation_eta = 0.1  # Learning rate
        self.backpropagation_epochs = 100000  # Number of epochs for learning
        self.w01 = np.asarray([])  # Weights from input to hidden layer
        self.w12 = np.asarray([])  # Weights from hidden layer to output node
        self.hidden_nodes = 15  # Number of hidden nodes
        if self.device == "watch":
            self.hidden_nodes = 15  # Number of hidden nodes
        self.weights01 = []  # Weights that will be used for training
        self.weights12 = []  # Weights that will be used for training
        self.error_list = []  # Used for graphical purposes

        self.lvq2_eta = 0.001  # Learning rate
        self.lvq2_epochs = 50000  # Number of epochs for learning

    def construct_input_vector(self):
        train_features_data_frame = pd.ExcelFile(self.path_merged + self.train_filename).parse('Sheet1')
        test_features_data_frame = pd.ExcelFile(self.path_merged + self.test_filename).parse('Sheet1')
        train_data = train_features_data_frame.values
        test_data = test_features_data_frame.values
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        self.X_train = train_data[:, :-1]
        self.X_test = test_data[:, :-1]
        self.y_train = train_data[:, -1:]
        self.y_test = test_data[:, -1:]

    def construct_model_for_backpropagation(self):
        tf.reset_default_graph()  # Reset the graph
        X = tf.placeholder(shape=(self.X_train.shape[0], self.X_train.shape[1]), dtype=tf.float64, name='X')
        y = tf.placeholder(shape=(self.X_train.shape[0], 1), dtype=tf.float64, name='y')
        self.define_weights_for_backpropagation()  # Initialize the weights
        a_h = tf.sigmoid(tf.matmul(X, self.w01))
        a_o = tf.sigmoid(tf.matmul(a_h, self.w12))
        loss = tf.reduce_mean(tf.square(a_o - y))
        optimizer = tf.train.GradientDescentOptimizer(self.backpropagation_eta)
        train = optimizer.minimize(loss)
        sess = tf.Session()  # Initialize the session
        sess.run(tf.global_variables_initializer())  # Initialize global variables

        for epoch in range(self.backpropagation_epochs):  # Run for some epcohs
            sess.run(train, feed_dict={X: self.X_train, y: self.y_train})
            self.error_list.append(sess.run(loss, feed_dict={X: self.X_train, y: self.y_train}))
            self.weights01 = sess.run(self.w01)  # Update the weights
            self.weights12 = sess.run(self.w12)  # Update the weights

        print("Loss with ", self.hidden_nodes, " hidden nodes and ", self.backpropagation_epochs, " epochs = ",
              self.error_list[-1])
        sess.close()

    def define_weights_for_backpropagation(self):
        self.w01 = tf.Variable(np.random.uniform(low=-0.1, high=0.1, size=(self.X_train.shape[1], self.hidden_nodes)),
                               dtype=tf.float64)
        self.w12 = tf.Variable(np.random.uniform(low=-0.1, high=0.1, size=(self.hidden_nodes, 1)), dtype=tf.float64)

    def run_backpropagation(self):
        print("Running Backpropagation")
        start_time = time.time()  # Start the timer
        self.construct_model_for_backpropagation()  # Construct the model using training data
        X = tf.placeholder(shape=(self.X_test.shape[0], self.X_test.shape[1]), dtype=tf.float64, name='X')
        y = tf.placeholder(shape=(self.X_test.shape[0], 1), dtype=tf.float64, name='y')
        a_h = tf.sigmoid(tf.matmul(self.X_test, self.weights01))
        a_o = tf.sigmoid(tf.matmul(a_h, self.weights12))
        init = tf.global_variables_initializer()  # Initialize the global variables
        with tf.Session() as sess:
            sess.run(init)
            y_pred = sess.run(a_o, feed_dict={X: self.X_test, y: self.y_test})
        print("Backpropagation Execution time: ", time.time() - start_time, " seconds")  # Calculate the execution time
        self.print_metrics(y_pred.round())  # Print the metrics
        # self.plot_error_graph()

    def run_lvq2(self):
        print("Running LVQ2")
        start_time = time.time()  # Start the timer
        learning_vector_quantization = LVQ2(n_inputs=self.X_train.shape[1], n_classes=2, step=self.lvq2_eta,
                                            n_subclasses=2)  # Initialize LVQ2 with regular and hyper parameters
        # Train the quantizer with number of epcohs
        learning_vector_quantization.train(self.X_train, self.y_train, epochs=self.lvq2_epochs)
        # Predict the results on testing data and the quantizer
        y_pred = learning_vector_quantization.predict(self.X_test)
        print("LVQ2 Execution time: ", time.time() - start_time, " seconds")  # Calculate the execution time
        self.print_metrics(y_pred)  # Print the metrics

    def print_metrics(self, predicted_output):
        res = cm(self.y_test, predicted_output)
        tp = res[0][0]
        fn = res[1][0]
        fp = res[0][1]
        tn = res[1][1]
        print("Accuracy: ", acs(self.y_test, predicted_output))
        print("TP: ", tp, ", FN: ", fn, ", FP: ", fp, "TN: ", tn)
        print(cr(self.y_test, predicted_output))


neural_network = NeuralNetwork()  # Create a neural network object
# Construct the input vector. This includes creating the training and testing data.
neural_network.construct_input_vector()
# neural_network.run_backpropagation()  # Run backpropagation
neural_network.run_lvq2()  # Run LVQ2

