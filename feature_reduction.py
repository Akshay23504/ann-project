import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_classif


class FeatureReduction:
    """
    This class is used after feature extraction and selection. The dataset
    begins with 36 features (columns). The methods (correlation and
    information gain) explained in this class reduce these 36 features to
    around 10 features. This reduction is a manual process and we have to
    analyze and identify the features to remove based on the results obtained
    from the reduction techniques described here.

    """

    def __init__(self):
        """
        As usual, lot of initialization stuff here.
        Filenames and directory names vary on the problem (fall or not fall,
        left side fall or right side fall)

        """

        # Refer to this link for why 0.6 was chosen as the threshold
        # https://www.researchgate.net/post/What_is_the_minimum_value_of_correlation_coefficient_to_prove_the_existence_of_the_accepted_relationship_between_scores_of_two_of_more_tests
        self.threshold = 0.6
        self.features_data_frame = pd.DataFrame()  # Initialize panda data frames
        self.features_data_frame_merged = pd.DataFrame()  # Initialize panda data frames
        self.user_id = "500"  # Don't know why I need this here
        self.sensor_name = "accelerometer"  # accelerometer or gyroscope
        self.device = "watch"  # phone or watch
        self.path_merged_without_labels = "../Dataset/merged_without_labels/"
        self.path_merged = "../Dataset/merged/"
        self.filename = self.device + "_" + self.sensor_name + "_features.xlsx"

    def correlation(self):
        """
        Perform correlation between every pair of features. For example,
        feature 1 is compared with all the other 35 features. Feature 2
        is compared with all the other 34 features and so on. The
        correlation gives a matrix of 36 x 36 dimensions. The diagonal values
        are 1.0.

        """

        # We can also do a zip and check...but ehhh
        column_correlation = set()  # Not a list, but a set
        # Just call corr(). Cannot get easier than this!
        correlation_matrix = self.features_data_frame.corr()
        plt.figure(figsize=(10, 10))
        plt.pcolor(correlation_matrix, edgecolors='k', cmap='hot')
        plt.xticks(np.arange(0.5, len(correlation_matrix.columns), 1), correlation_matrix.columns, rotation=90)
        plt.yticks(np.arange(0.5, len(correlation_matrix.index), 1), correlation_matrix.index)
        # Uncomment this for the heat map
        # plt.show()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                # Consider removing a feature if the correlation values against all other features are above the
                # threshold
                if correlation_matrix.iloc[i, j] >= self.threshold:
                    column_to_remove = correlation_matrix.columns[i]
                    column_correlation.add(column_to_remove)
        print(len(column_correlation))
        print(column_correlation)

    def mutual_information(self):
        """
        Perform information gain between the features. To perform the mutual
        information, we need a labeled dataset. The mutual information spits
        out values for each feature. The values are sorted and then recorded
        for analyzing.

        """

        scores_label = {}
        data = self.features_data_frame_merged.values
        np.random.shuffle(data)
        x = data[:, :-1]
        y = data[:, -1:]
        # Obtain the information gain scores
        scores = mutual_info_classif(x, y.ravel())
        # Get the column names
        feature_names = self.features_data_frame_merged.columns.values
        for i in range(len(scores)):
            scores_label[feature_names[i]] = scores[i]
        # Sort the scores in ascending order
        scores_label = sorted(scores_label.items(), key=lambda kv: kv[1])
        # Reverse the order, because we need descending order
        scores_label.reverse()
        print(scores_label)
        plt.figure(figsize=(12, 8))
        plt.bar(np.arange(len(scores)), scores, align='center', alpha=0.5)
        plt.xticks(np.arange(len(scores)), feature_names, rotation=90)
        plt.ylabel('Scores')
        plt.xlabel('Features')
        plt.title('Information Gain')
        plt.grid()
        # Uncomment this for the bar graph of the scores of features for mutual information
        # plt.show()

    def get_the_features(self):
        """
        Get the data from the excel files using panda data frames. Uncomment
        or comment the data frames here based on the problem. Also uncomment
        or comment the reduction technique based on the necessity.

        """

        # Sheet 1 only
        self.features_data_frame = pd.ExcelFile(self.path_merged_without_labels + self.filename).parse('Sheet1')
        self.features_data_frame_merged = pd.ExcelFile(self.path_merged + self.filename).parse('Sheet1')
        self.correlation()
        self.mutual_information()


feature_reduction = FeatureReduction()
feature_reduction.get_the_features()
