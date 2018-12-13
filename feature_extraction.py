import numpy as np
import pandas as pd
import scipy.stats as scs
from statsmodels import robust

import pre_process


class FeatureExtraction:
    """
    This class takes care of all the feature extraction functionality. 12
    carefully selected features and the their methods are defined in this
    class. After extracting the features for the data, they are written to
    file which acts like a dataset which a lot of features. The 12 features
    contribute for one dimension and since there are 3 dimensions (x, y and z),
    the dataset will consist of 36 features. In other words, the dataset will
    contain 36 columns.

    """

    def __init__(self, pp):
        """
        Initialize a data frame, the pre_process object, feature list and the
        labe list. Just keeping it simple and getting the job done.

        :param pp: pre_process object

        """

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
        self.pre_process = pp
        self.df = pd.DataFrame(columns=self.label_list)

    def feature_mean(self, values):
        """
        Calculate the mean of the values. The values here are the values from
        a dimension (x, y or z).

        :param values: Dimension (x, y or z) values

        """

        self.feature_list.append(np.mean(values))

    def feature_median(self, values):
        """
        Calculate the median of the values. The values here are the values from
        a dimension (x, y or z).

        :param values: Dimension (x, y or z) values

        """

        self.feature_list.append(np.median(values))

    def feature_variance(self, values):
        """
        Calculate the variance of the values. The values here are the values from
        a dimension (x, y or z).

        :param values: Dimension (x, y or z) values

        """

        self.feature_list.append(np.var(values))

    def feature_standard_deviance(self, values):
        """
        Calculate the standard deviation of the values. The values here are
        the values from a dimension (x, y or z).

        :param values: Dimension (x, y or z) values

        """

        self.feature_list.append(np.std(values))

    def feature_median_absolute_deviation(self, values):
        """
        Calculate the median absolute deviation  of the values. The values
        here are the values from a dimension (x, y or z).

        :param values: Dimension (x, y or z) values

        """

        self.feature_list.append(robust.mad(values))

    def feature_interquartile_range(self, values):
        """
        Calculate the interquartile range of the values. The values here are
        the values from a dimension (x, y or z).

        :param values: Dimension (x, y or z) values

        """

        self.feature_list.append(scs.iqr(values))

    def feature_power(self, values):
        """
        Calculate the power of the values. The values here are the values from
        a dimension (x, y or z).

        :param values: Dimension (x, y or z) values

        """

        self.feature_list.append(sum(np.array(np.square(np.abs(values)))) / len(values))

    def feature_energy(self, values):
        """
        Calculate the energy of the values. The values here are the values from
        a dimension (x, y or z).

        :param values: Dimension (x, y or z) values

        """

        self.feature_list.append(sum(np.array(np.square(np.abs(values)))))

    def feature_peak_to_peak_amplitude(self, values):
        """
        Calculate the peak-to-peak amplitude of the values. The values here
        are the values from a dimension (x, y or z).

        :param values: Dimension (x, y or z) values

        """

        self.feature_list.append(max(values) - min(values))

    def feature_auto_correlation(self, values):
        """
        Calculate the auto correlation of the values. The values here are the
        values from a dimension (x, y or z). For auto correlation, we need the
        values in a pandas data frame series. The number of lags is 1.

        :param values: Dimension (x, y or z) values

        """

        result = pd.Series(values)
        self.feature_list.append(pd.Series.autocorr(result))

    def feature_kurtosis(self, values):
        """
        Calculate the kurtosis of the values. The values here are the values from
        a dimension (x, y or z).

        :param values: Dimension (x, y or z) values

        """

        self.feature_list.append(scs.kurtosis(values))

    def feature_skew(self, values):
        """
        Calculate the skew of the values. The values here are the values from
        a dimension (x, y or z).

        :param values: Dimension (x, y or z) values

        """

        self.feature_list.append(scs.skew(values))

    def get_me_all_features(self, values):
        """
        This method calls each feature in an order for the input parameter
        values.

        :param values: Dimension (x, y or z) values

        """

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
        """
        In order to calculate the features, we need the data. The data is
        obtained from the pre_process file. THe methods in the pre_process
        file take care of removing noise, clering unnecessary data etc. and
        formulates the data as needed for the feature extraction.

        """

        self.pre_process.store_checkpoints()
        self.pre_process.store_fall_instances()
        self.pre_process.get_fall_instances()

    def construct_single_row(self, single_row_list, values):
        """
        A single row in the dataset contains 36 columns. Once the features are
        calculate for x dimension, they have to be collected for y and z
        dimensions. All the 36 values from the three dimensions are flattened
        to a single row and appended to a list. The flattening takes place
        in write_to_features_file method.

        :param single_row_list: Append all the values as a single row
        :param values: x, y or z dimension values

        """

        # Reset the list
        self.feature_list = []
        # Get all the features for x, y and z dimensions and construct a list of lists.
        # In terms of matrix, this will be 1 column, 3 rows matrix.
        self.get_me_all_features(values)
        single_row_list.append(self.feature_list)

    def write_features_to_file(self):
        """
        Construct a data frame of all the features for all the instances of the
        data and then write it to a CSV file.

        """

        # Call the pre_processing
        self.some_pre_processing()
        fall_instance_dimension = self.pre_process.get_fall_instance_dimension()
        # For each instance and dimension of an instance
        for k, v in fall_instance_dimension.items():
            # Reset the data frame
            data_frame = self.df
            single_row_values = []
            self.construct_single_row(single_row_values, v['x'])
            self.construct_single_row(single_row_values, v['y'])
            self.construct_single_row(single_row_values, v['z'])

            """
            single_row_values will now have 1 column, 3 rows and looks something like this:
            [[Feature values for x], [Feature values for y], [Feature values for z]]
            Flatten this list into a 1 dimension to look like this:
            [Feature values for x...Feature values for y...Feature values for z]
            """
            single_row_values = [val for row in single_row_values for val in row]
            # Beauty of panda data frames. It knows how to insert the data for each row and column properly
            data_frame.loc["Instance_" + str(k)] = single_row_values
            data_frame.to_csv(self.pre_process.path + self.pre_process.user_id + "_" + self.pre_process.device + "_" +
                              self.pre_process.sensor_name + "_features_1.csv")


f = FeatureExtraction(pre_process.PreProcess())
f.write_features_to_file()
