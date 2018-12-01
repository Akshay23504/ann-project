import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as cr
from sklearn.metrics import accuracy_score as acs
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class FeatureReductionPart2:
    # TODO: Borrow comments from homeworks
    # TODO: Remove Entropy
    def __init__(self):
        self.train_features_data_frame = pd.DataFrame()
        self.test_features_data_frame = pd.DataFrame()
        self.sensor_name = "sensor"  # accelerometer or gyroscope or sensor
        self.device = "watch"  # phone or watch
        self.path_merged = "../Dataset/merged/"
        self.train_filename = self.device + "_" + self.sensor_name + "_features_train_lr.xlsx"
        self.test_filename = self.device + "_" + self.sensor_name + "_features_test_lr.xlsx"
        self.X_train = np.asarray([])  # Training data
        self.y_train = np.asarray([])  # Training data (labels)
        self.X_test = np.asarray([])  # Testing data
        self.y_test = np.asarray([])  # Testing data (labels)
        self.svm_kernel = 'linear'  # Kernel for SVM. 'rbf' is Gaussian. 'linear' is linear

    def run_svm(self):
        sv_classifier = SVC(kernel=self.svm_kernel)  # Initialize the classifier with a kernel
        sv_classifier.fit(self.X_train, self.y_train.ravel())  # Fit the training data
        y_pred = sv_classifier.predict(self.X_test)  # Predict the results on testing data and the classifier
        self.print_metrics(y_pred)  # Print the metrics

    def run_naive_bayes(self):
        nb_classifier = GaussianNB()  # Initialize the classifier with a kernel
        nb_classifier.fit(self.X_train, self.y_train.ravel())  # Fit the training data
        y_pred = nb_classifier.predict(self.X_test)  # Predict the results on testing data and the classifier
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

    def get_the_features(self):
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

        # To delete one column
        # column_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        column_index = [0, 1, 2, 3, 5, 6, 8, 15, 16, 18, 20, 25, 29, 31, 33]  # Watch
        # column_index = [0, 1, 3, 4, 5, 9, 11, 12, 13, 15, 17, 18, 20, 23, 24, 27, 29, 31]  # Phone
        self.X_train = np.delete(self.X_train, column_index, axis=1)
        self.X_test = np.delete(self.X_test, column_index, axis=1)


feature_reduction_part2 = FeatureReductionPart2()
feature_reduction_part2.get_the_features()
feature_reduction_part2.run_svm()
feature_reduction_part2.run_naive_bayes()

