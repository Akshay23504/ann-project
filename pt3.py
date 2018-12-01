import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_classif


class FeatureReduction:
    def __init__(self):
        # https://www.researchgate.net/post/What_is_the_minimum_value_of_correlation_coefficient_to_prove_the_existence_of_the_accepted_relationship_between_scores_of_two_of_more_tests
        self.threshold = 0.6
        self.features_data_frame = pd.DataFrame()
        self.features_data_frame_merged = pd.DataFrame()
        self.user_id = "500"  # Don't know why I need this here
        self.sensor_name = "accelerometer"  # accelerometer or gyroscope
        self.device = "phone"  # phone or watch
        # This directory has values from left and right side falls and without labels
        self.path_merged_without_labels = "../Dataset/RawFallRight/"
        self.path_merged = "../Dataset/merged/"
        self.filename = self.device + "_" + self.sensor_name + "_features_lr.xlsx"

    def correlation(self):
        # We can also do a zip and check...but ehhh
        column_correlation = set()
        correlation_matrix = self.features_data_frame.corr()
        plt.figure(figsize=(10, 10))
        plt.pcolor(correlation_matrix, edgecolors='k', cmap='hot')
        plt.xticks(np.arange(0.5, len(correlation_matrix.columns), 1), correlation_matrix.columns, rotation=90)
        plt.yticks(np.arange(0.5, len(correlation_matrix.index), 1), correlation_matrix.index)
        # plt.show()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if correlation_matrix.iloc[i, j] >= self.threshold:
                    column_to_remove = correlation_matrix.columns[i]
                    column_correlation.add(column_to_remove)
                    # if column_to_remove in self.data.columns:
                    #     del self.data[column_to_remove]
        print(len(column_correlation))
        print(column_correlation)

    def mutual_information(self):
        scores_label = {}
        data = self.features_data_frame_merged.values
        np.random.shuffle(data)
        x = data[:, :-1]
        y = data[:, -1:]
        scores = mutual_info_classif(x, y.ravel())
        feature_names = self.features_data_frame_merged.columns.values
        for i in range(len(scores)):
            scores_label[feature_names[i]] = scores[i]
        scores_label = sorted(scores_label.items(), key=lambda kv: kv[1])
        scores_label.reverse()
        print(scores_label)
        plt.figure(figsize=(12, 8))
        plt.bar(np.arange(len(scores)), scores, align='center', alpha=0.5)
        plt.xticks(np.arange(len(scores)), feature_names, rotation=90)
        plt.ylabel('Scores')
        plt.xlabel('Features')
        plt.title('Information Gain')
        plt.grid()
        # plt.show()

    def get_the_features(self):
        # Sheet 1 only
        # self.features_data_frame = pd.ExcelFile(self.path_merged_without_labels + self.filename).parse('Sheet1')
        self.features_data_frame_merged = pd.ExcelFile(self.path_merged + self.filename).parse('Sheet1')
        # self.correlation()
        self.mutual_information()


feature_reduction = FeatureReduction()
feature_reduction.get_the_features()

