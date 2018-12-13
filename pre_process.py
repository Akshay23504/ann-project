import csv
import datetime
import pandas as pd


def convert_string_to_time(time_in_str):
    """
    As the name says, converts 'time_in_str' from the string format to python
    datetime format. Reason: Using the datetime object, we can add, subtract
    and play freely with time.

    :param time_in_str: String format of time to be converted to datetime
    :return: The datetime format of the parameter

    """

    return datetime.datetime.strptime(time_in_str, '%Y-%m-%d %H:%M:%S:%f')


def convert_epoch_to_datetime(time_in_epoch):
    """
    As the name says, converts 'time_in_epoch' from the epoch format to python
    datetime format. Reason: Using the datetime object, we can add, subtract
    and play freely with time.

    :param time_in_epoch: Floating epoch variable
    :return: The datetime format of the parameter

    """

    return datetime.datetime.fromtimestamp(time_in_epoch / 1000)  # 1000 is for milliseconds


class PreProcess:
    """
    This class contains all the ingredients for pre-processing. From the initial
    step soon after the raw data is obtained, this class takes care of all the
    things that are necessary for the raw data to nourish and look pretty.

    All the pre-processing steps defined here can be run from the
    feature_extraction file or can also be run stand-alone. To run in
    stand-alone mode, uncomment the last couple of lines in this file. But,
    there is no need to run this file alone.

    """

    def __init__(self):
        """
        Initialization stuff. Different sensors, devices and users need to be
        configured here. And also the file names, file checkpoints and window
        size. In addition, there are also some dictionaries and lists
        initialized which are required for pre-processing.

        Once, the parameters like the device, sensor and user are set, just
        run the file and it should achieve the outcome. There is no need to
        touch other parts of the code.

        """

        # Do not be intimated by the 500 series number. We do not have 500 users!
        self.user_id = "505"  # User id
        self.device = "phone"  # phone or watch
        self.sensor_name = "gyroscope"  # accelerometer or gyroscope
        self.path = "../Dataset/RawFallRight/" + self.user_id + "/"  # RawFallRight or RawFallLeft
        self.filename = self.user_id + "_" + self.device + "_" + self.sensor_name + ".xlsx"  # Filename
        self.filename_checkpoint = "Events" + self.user_id + ".txt"  # Checkpoint file
        self.fall_checkpoints = []
        self.entries = {}
        self.window_size = 1  # In seconds
        self.fall_instance_dimension = {}

    def store_checkpoints(self):
        """
        Just storing the checkpoints in a list here. There is also some extent
        of cleaning done.

        """

        # Open the checkpoint file
        with open(self.path + self.filename_checkpoint) as input_file:
            read_input = csv.reader(input_file, delimiter=',')
            for row in read_input:
                if row[0] == 'event2':  # Just making sure we consider only the events
                    # Convert the timestamp to datetime and store in a list
                    self.fall_checkpoints.append(convert_epoch_to_datetime(float(row[1])))

    def store_fall_instances(self):
        """
        Get the raw data values from the excel file. Note that the values from
        data frame is considered which is numpy array. Some additional
        pre-processing is done based on the device.

        Then comes the core part. The data from the checkpoint instance to a
        certain window size (1 second) is considered. Because the sensors from
        the two devices continuously record the values, we only 1 second of the
        data for every instance. The other data might include but not limited
        to - falling on the air pillow, getting up, talking to other people,
        taking rest as falling deliberately is a tedious job and not falling
        correctly!

        The code in this method is not really optimized for time complexity!
        Pfff
        Maybe there is a better way...hmmm...

        """

        # Open the raw data file and convert to numpy array
        input_file = pd.ExcelFile(self.path + self.filename).parse('Sheet1').get_values()  # Sheet 1 only
        for row in input_file:
            # Suppose there are worst ways to do this...
            if self.device == "phone":
                # We already have timestamp in phone data and the UNIX timestamp ain't working
                time = convert_string_to_time(row[5])
            else:
                # This is for watch data which has proper format of time
                time = convert_epoch_to_datetime(float(row[4]))
            for i in range(len(self.fall_checkpoints)):
                # Consider all the data in the window size from the checkpoint.
                # Store the entries in a dictionary so that the data for each device and each instance can be
                # accessed in O(1) time
                if self.fall_checkpoints[i] <= time <= \
                        self.fall_checkpoints[i] + datetime.timedelta(seconds=self.window_size):
                    if self.entries.get(i + 1) is None:
                        self.entries[i + 1] = []
                    self.entries[i + 1].append([i + 1, row[1], row[2], row[3], row[4]])  # row[4] is not required!

    def get_fall_instances(self):
        """
        All the data is in the entries dictionary. But, sometimes, we need the
        data to be in a different format. Here, the format is, we need separate
        x, y and z values. Reason: We can extract features from this, if we
        have x, y and z as separate columns.

        fall_instance_dimension is for each fall and there are
        x, y and z values

        Example: fall_instance_dimension[1]['x'] is fall instance 1, x value

        """

        for k, v in self.entries.items():
            # Create empty dictionaries for each instance
            # Create empty lists for each dimension
            self.fall_instance_dimension[k] = {}
            self.fall_instance_dimension[k]['x'] = []
            self.fall_instance_dimension[k]['y'] = []
            self.fall_instance_dimension[k]['z'] = []
            for vs in v:
                # Each value is list. So, iterate over the list and get the first three values of each list.
                # The first three values correspond to x, y and z values.
                self.fall_instance_dimension[k]['x'].append(vs[1])
                self.fall_instance_dimension[k]['y'].append(vs[2])
                self.fall_instance_dimension[k]['z'].append(vs[3])

    def get_fall_instance_dimension(self):
        """
        Just a getter method

        :return: fall_instance_dimension

        """

        return self.fall_instance_dimension


# Uncomment the below lines if this file needs to be run separately
# pre_process = PreProcess()
#
# pre_process.store_checkpoints()
# pre_process.store_fall_instances()
# pre_process.get_fall_instances()
#
