import time

import numpy as np
import pandas as pd
import tensorflow as tf
from neupy.algorithms.competitive.lvq import LVQ2
from sklearn.metrics import accuracy_score as acs
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm


class NeuralNetwork:
    """
    Define the input vector, labels, training data, testing data,
    hyperparameters for backpropagation and LVQ2.

    Contains methods for backrpopagation and LVQ2.

    Backpropagation uses sigmoid activation function and loss function. The
    loss or error function is Mean Sqaured Error (MSE). Backpropagation is
    implemented using tensorflow. Gradient Descent Optimizer is used.

    LVQ2 is implemented using NeuPy.

    Also contains methods for printing metrics like accuracy, confusion
    matrix values, precision, recall, f1-score and other metrics using
    classification_report from sklearn.

    The results from the two algorithms are recorded and analyzed.

    """

    def __init__(self):
        """
        Some initialization stuff common to all algorithms and some specific
        for each algorithm (Backpropagation and LVQ2).

        The files and directory names may be changed according to the problem
        (fall or not fall, left side fall or right side fall).

        """

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

        # LVQ2 specific
        self.lvq2_eta = 0.001  # Learning rate
        self.lvq2_epochs = 50000  # Number of epochs for learning

    def construct_input_vector(self):
        """
        Now there are 2 files - train and test file. Get these files using
        pandas data frame and convert to numpy array and shuffle it. The
        data is now split four ways - X_train, X_test, y_train and y_test.

        The X_train is obtained by removing the last column from the training
        data. The X_test is obtained by removing the last column from the
        testing data. The y_train is obtained from removing all but last
        column from the training data. And the y_test is obtained by removing
        all but the last column from the testing data.

        """

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
        """
        This method contains tensorflow code to implement backpropagation.
        We define X and y which are placeholders for training data and labels.
        Sigmoid activation function and MSE is used here. Then, the algorithm
        is run for the defined number of iterations. At the end, we will have
        two weight vectors which will be used for testing.

        """

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
        """
        The values for the weights from the input layer to hidden layer and
        from hidden layer to the output layer are initialized here with random
        values ranging from -0.1 to 0.1. The shape is dynamically determined
        using the training data.

        """

        self.w01 = tf.Variable(np.random.uniform(low=-0.1, high=0.1, size=(self.X_train.shape[1], self.hidden_nodes)),
                               dtype=tf.float64)
        self.w12 = tf.Variable(np.random.uniform(low=-0.1, high=0.1, size=(self.hidden_nodes, 1)), dtype=tf.float64)

    def run_backpropagation(self):
        """
        First, construct the model using the training data. After the model is
        constructed, we will have two sets of weights which can be used in this
        method for testing. As before, we define X and y placeholders but with
        shapes taken from the dimensions of testing data. Since, it is a binary
        classification, the output will have one column and some number of
        rows.

        """

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
        """
        The Linear Vector Quantizers (LVQ) here uses NeuPy for its
        implementation. We are interested in LVQ2 here and use the LVQ2 class
        provided by the library. It has several hyperparameters and some
        regular parameters. The regular parameters are n_inputs (number of
        features) and n_classes (number of classes). There are other regular
        parameters to display the results, display the results after
        certain epochs etc. But these are meh. The hyperparameters are the
        learning rate, number of iterations and number of sub-classes. The
        number of sub-classes is 2 which is equal to the number of classes.
        Any other value just doesn't produce good results. The learning rate
        and the number of iterations are important here.

        """

        print("Running LVQ2")
        start_time = time.time()  # Start the timer
        learning_vector_quantization = LVQ2(n_inputs=self.X_train.shape[1], n_classes=2, step=self.lvq2_eta,
                                            n_subclasses=2)  # Initialize LVQ2 with regular and hyper parameters
        # Train the quantizer with number of epochs
        learning_vector_quantization.train(self.X_train, self.y_train, epochs=self.lvq2_epochs)
        # Predict the results on testing data and the quantizer
        y_pred = learning_vector_quantization.predict(self.X_test)
        print("LVQ2 Execution time: ", time.time() - start_time, " seconds")  # Calculate the execution time
        self.print_metrics(y_pred)  # Print the metrics

    def print_metrics(self, predicted_output):
        """
        Print some MVP metrics. sklearn is used for calculation of all the
        metric values. Confusion matrix values (true positive, false negative,
        false positive and true negative), precision, recall, f1-score and
        accuracy is calculated. There are few other metrics which comes under
        classification report, but meh ot them.

        We need the actual labels and the predicted labels to calculate the
        metrics. We can get the actual labels from the class variable and
        the predicted output or predicted labels are passed as a parameter
        after running each algorithm.

        :param predicted_output: Predicted labels

        """

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
