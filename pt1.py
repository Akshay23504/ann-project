import csv
import datetime
import pandas as pd


def convert_string_to_time(time_in_str):
    return datetime.datetime.strptime(time_in_str, '%Y-%m-%d %H:%M:%S:%f')


def convert_epoch_to_datetime(time_in_epoch):
    return datetime.datetime.fromtimestamp(time_in_epoch / 1000)  # 1000 is for milliseconds


class PreProcess:
    def __init__(self):
        self.user_id = "505"
        self.sensor_name = "gyroscope"  # accelerometer or gyroscope
        self.device = "phone"  # phone or watch
        self.path = "../Dataset/RawFallRight/" + self.user_id + "/"
        self.filename = self.user_id + "_" + self.device + "_" + self.sensor_name + ".xlsx"
        self.filename_checkpoint = "Events" + self.user_id + ".txt"
        self.fall_checkpoints = []
        self.entries = {}
        self.window_size = 3  # In seconds
        self.fall_instance_dimension = {}

    def store_checkpoints(self):
        """
        Just storing the checkpoints in a list

        """

        with open(self.path + self.filename_checkpoint) as input_file:
            read_input = csv.reader(input_file, delimiter=',')
            for row in read_input:
                if row[0] == 'event2':  # Just making sure we consider only the events
                    self.fall_checkpoints.append(convert_epoch_to_datetime(float(row[1])))

    def store_fall_instances(self):
        """
        The code is not really optimized for time complexity! Pffff
        Maybe there is a better way..hmmm...

        """

        input_file = pd.ExcelFile(self.path + self.filename).parse('Sheet1').get_values()  # Sheet 1 only
        for row in input_file:
            # Suppose there are worst ways to do this...
            if self.device == "phone":
                time = convert_string_to_time(row[5])  # We already have timestamp in phone data and the UNIX timestamp ain't working
            else:
                time = convert_epoch_to_datetime(float(row[4]))
            for i in range(len(self.fall_checkpoints)):
                if self.fall_checkpoints[i] <= time <= \
                        self.fall_checkpoints[i] + datetime.timedelta(seconds=self.window_size):
                    if self.entries.get(i + 1) is None:
                        self.entries[i + 1] = []
                    self.entries[i + 1].append([i + 1, row[1], row[2], row[3], row[4]])  # row[4] is not required!

    def get_fall_instances(self):
        """
        fall_instance_dimension is for each fall and there are
        x, y and z values

        Example: fall_instance_dimension[1]['x'] is fall instance 1, x value

        """

        for k, v in self.entries.items():
            self.fall_instance_dimension[k] = {}
            self.fall_instance_dimension[k]['x'] = []
            self.fall_instance_dimension[k]['y'] = []
            self.fall_instance_dimension[k]['z'] = []
            for vs in v:
                self.fall_instance_dimension[k]['x'].append(vs[1])
                self.fall_instance_dimension[k]['y'].append(vs[2])
                self.fall_instance_dimension[k]['z'].append(vs[3])

    def get_fall_instance_dimension(self):
        return self.fall_instance_dimension


# pre_process = PreProcess()
#
# pre_process.store_checkpoints()
# pre_process.store_fall_instances()
# pre_process.get_fall_instances()
#
print()
