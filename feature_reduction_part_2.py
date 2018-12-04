import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as cr
from sklearn.metrics import accuracy_score as acs
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class FeatureReductionPart2:
    """
    This class deals with reducing the features even more. Simple classifiers
    like SVM and Naive Bayes is used to calculate baseline performance.

    """

    def __init__(self):
        """
        Initialization stuff happens here. Filenames and file paths vary
        on the problem (fall or not fall, left side fall or right side fall).

        """

        self.train_features_data_frame = pd.DataFrame()  # Initialize panda data frames
        self.test_features_data_frame = pd.DataFrame()  # Initialize panda data frames
        self.sensor_name = "sensor"  # accelerometer or gyroscope or sensor
        self.device = "phone"  # phone or watch
        self.path_merged = "../Dataset/merged/"
        self.train_filename = self.device + "_" + self.sensor_name + "_features_train_lr.xlsx"
        self.test_filename = self.device + "_" + self.sensor_name + "_features_test_lr.xlsx"
        self.X_train = np.asarray([])  # Training data
        self.y_train = np.asarray([])  # Training data (labels)
        self.X_test = np.asarray([])  # Testing data
        self.y_test = np.asarray([])  # Testing data (labels)
        self.svm_kernel = 'linear'  # Kernel for SVM. 'rbf' is Gaussian. 'linear' is linear

    def run_svm(self):
        """
        The Support Vector Machines (SVM) here uses sklearn for its
        implementation. The kernel is a parameter to the SVC class. The kernel
        can be Gaussian or linear or radial basis etc. There are of course
        other ones like sigmoid etc. There are other hyperparameters like C and
        maximum iterations. The value of C is 1.0 by default and maximum
        iterations is -1 by default. -1 means there is no limit. These two values
        are pretty ideal for our setting.

        """

        sv_classifier = SVC(kernel=self.svm_kernel)  # Initialize the classifier with a kernel
        sv_classifier.fit(self.X_train, self.y_train.ravel())  # Fit the training data
        y_pred = sv_classifier.predict(self.X_test)  # Predict the results on testing data and the classifier
        self.print_metrics(y_pred)  # Print the metrics

    def run_naive_bayes(self):
        """
        The Gaussian Naive Bayes suits the data and the data is fit using the
        classifier. No hyperparameters involved here. Naive Bayes utility from
        sklearn is used here.

        """

        nb_classifier = GaussianNB()  # Initialize the classifier with a kernel
        nb_classifier.fit(self.X_train, self.y_train.ravel())  # Fit the training data
        y_pred = nb_classifier.predict(self.X_test)  # Predict the results on testing data and the classifier
        self.print_metrics(y_pred)  # Print the metrics

    def print_metrics(self, predicted_output):
        """
        Print some MVP metrics. sklearn is used for calculation of all the
        metric values. Confusion matrix values (true positive, false negative,
        false positive and true negative), precision, recall, f1-score and
        accuracy is calculated. There are few other metrics which comes under
        classification report, but meh to them.

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

    def get_the_features(self):
        """
        Now there are 2 files - train and test file. Get these files using
        pandas data frame and convert to numpy array and shuffle it. The
        data is now split four ways - X_train, X_test, y_train and y_test.

        The X_train is obtained by removing the last column from the training
        data. The X_test is obtained by removing the last column from the
        testing data. The y_train is obtained from removing all but last
        column from the training data. And the y_test is obtained by removing
        all but the last column from the testing data.

        Then, a leave one column approach is performed and records are carefully
        analyzed nd recorded. After leaving one column at a time and analyzing
        the results deeply, the next step is to leave multiple columns at once
        and check the algorithm performance for each run. These results are
        also analyzed with care.

        """

        # Sheet 1 only
        self.train_features_data_frame = pd.ExcelFile(self.path_merged + self.train_filename).parse('Sheet1')
        self.test_features_data_frame = pd.ExcelFile(self.path_merged + self.test_filename).parse('Sheet1')
        train_data = self.train_features_data_frame.values
        test_data = self.test_features_data_frame.values
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        self.X_train = train_data[:, :-1]
        self.X_test = test_data[:, :-1]
        self.y_train = train_data[:, -1:]
        self.y_test = test_data[:, -1:]

        # To delete one column. DO not remove or mess with column_index values here. They are optimal
        # column_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        # column_index = [0, 1, 2, 3, 5, 6, 8, 15, 16, 18, 20, 25, 29, 31, 33]  # Watch
        # column_index = [0, 1, 3, 4, 5, 9, 11, 12, 13, 15, 17, 18, 20, 23, 24, 27, 29, 31]  # Phone
        column_index = []  # Include all the columns
        self.X_train = np.delete(self.X_train, column_index, axis=1)
        self.X_test = np.delete(self.X_test, column_index, axis=1)


feature_reduction_part2 = FeatureReductionPart2()
feature_reduction_part2.get_the_features()
feature_reduction_part2.run_svm()
feature_reduction_part2.run_naive_bayes()
