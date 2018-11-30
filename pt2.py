import numpy as np
import pandas as pd
import scipy.stats as scs
from statsmodels import robust

import pt1


class Feature:

    def __init__(self, pre_process):
        self.feature_list = []
        self.label_list = ["Mean_x", "Median_x", "Variance_x", "Standard Deviance_x", "Median Absolute Deviation_x",
                           "Interquartile Range_x", "Power_x", "Energy_x", "Peak-to-Peak Amplitude_x",
                           "Auto Correlation_x", "Kurtosis_x", "Skew_x",
                           "Mean_y", "Median_y", "Variance_y", "Standard Deviance_y", "Median Absolute Deviation_y",
                           "Interquartile Range_y", "Power_y", "Energy_y", "Peak-to-Peak Amplitude_y",
                           "Auto Correlation_y", "Kurtosis_y", "Skew_y",
                           "Mean_z", "Median_z", "Variance_z", "Standard Deviance_z", "Median Absolute Deviation_z",
                           "Interquartile Range_z", "Power_z", "Energy_z", "Peak-to-Peak Amplitude_z",
                           "Auto Correlation_z", "Kurtosis_z", "Skew_z"
                           ]
        self.pre_process = pre_process
        self.df = pd.DataFrame(columns=self.label_list)

    def feature_mean(self, values):
        self.feature_list.append(np.mean(values))

    def feature_median(self, values):
        self.feature_list.append(np.median(values))

    def feature_variance(self, values):
        self.feature_list.append(np.var(values))

    def feature_standard_deviance(self, values):
        self.feature_list.append(np.std(values))

    def feature_median_absolute_deviation(self, values):
        self.feature_list.append(robust.mad(values))

    def feature_interquartile_range(self, values):
        self.feature_list.append(scs.iqr(values))

    def feature_power(self, values):
        self.feature_list.append(sum(np.array(np.square(np.abs(values)))) / len(values))

    def feature_energy(self, values):
        self.feature_list.append(sum(np.array(np.square(np.abs(values)))))

    def feature_peak_to_peak_amplitude(self, values):
        self.feature_list.append(max(values) - min(values))

    def feature_auto_correlation(self, values):
        # result = np.correlate(values, values, mode='full')
        # self.feature_list.append(result[int(result.size / 2):])
        result = pd.Series(values)
        self.feature_list.append(pd.Series.autocorr(result))

    def feature_kurtosis(self, values):
        self.feature_list.append(scs.kurtosis(values))

    def feature_skew(self, values):
        self.feature_list.append(scs.skew(values))

    def get_me_all_features(self, values):
        self.feature_mean(values)
        self.feature_median(values)
        self.feature_variance(values)
        self.feature_standard_deviance(values)
        self.feature_median_absolute_deviation(values)
        self.feature_interquartile_range(values)
        self.feature_power(values)
        self.feature_energy(values)
        self.feature_peak_to_peak_amplitude(values)
        self.feature_auto_correlation(values)
        self.feature_kurtosis(values)
        self.feature_skew(values)

    def some_pre_processing(self):
        self.pre_process.store_checkpoints()
        self.pre_process.store_fall_instances()
        self.pre_process.get_fall_instances()

    def construct_single_row(self, single_row_list, values):
        self.feature_list = []
        self.get_me_all_features(values)
        single_row_list.append(self.feature_list)

    def write_features_to_file(self):
        self.some_pre_processing()
        fall_instance_dimension = self.pre_process.get_fall_instance_dimension()
        for k, v in fall_instance_dimension.items():
            data_frame = self.df
            single_row_values = []
            self.construct_single_row(single_row_values, v['x'])
            self.construct_single_row(single_row_values, v['y'])
            self.construct_single_row(single_row_values, v['z'])
            single_row_values = [val for row in single_row_values for val in row]
            data_frame.loc["Instance_" + str(k)] = single_row_values
            data_frame.to_csv(self.pre_process.path + self.pre_process.user_id + "_" + self.pre_process.device + "_" +
                              self.pre_process.sensor_name + "_features_1.csv")


f = Feature(pt1.PreProcess())
f.write_features_to_file()

