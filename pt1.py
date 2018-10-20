import csv
import datetime
import pandas as pd


def convert_string_to_time(time_in_str):
    return datetime.datetime.strptime(time_in_str, '%Y-%m-%d %H:%M:%S.%f')


def convert_epoch_to_datetime(time_in_epoch):
    return datetime.datetime.fromtimestamp(time_in_epoch / 1000)  # 1000 is for milliseconds


class PreProcess:
    def __init__(self):
        self.user_id = "500"
        self.sensor_name = "gyroscope"
        self.path = "../Dataset/RawFallLeft/" + self.user_id + "/"
        self.filename = self.user_id + "_" + self.sensor_name + ".xlsx"
        self.filename_checkpoint = "Events" + self.user_id + ".txt"
        self.door_open_checkpoints = {}
        self.entries = {}
        self.window_size = 3  # In seconds
        self.door_instance_dimension = {}

    def store_checkpoints(self):
        """
        Just storing the checkpoints in a list

        """

        with open(self.path + self.filename_checkpoint) as input_file:
            read_input = csv.reader(input_file, delimiter=',')
            door_counter = 1
            for row in read_input:
                if row[0] == 'door open':
                    if self.door_open_checkpoints.get(door_counter) is None:
                        self.door_open_checkpoints[door_counter] = []
                    self.door_open_checkpoints[door_counter].append(convert_epoch_to_datetime(float(row[1])))
                if row[0] == 'event1':
                    door_counter += 1

    def store_door_opening_instances(self):
        """
        The code is not really optimized for time complexity!
        Maybe there is a better way..hmmm...

        The jth loop is the instance. The instance starts from 0. This instance is
        added as the first column to the entries

        """

        input_file = pd.ExcelFile(self.path + self.filename).parse('Sheet1').get_values()  # Sheet 1 only
        for row in input_file:
            time = convert_epoch_to_datetime(float(row[4]))
            for i in range(len(self.door_open_checkpoints)):
                for j in range(len(self.door_open_checkpoints[i + 1])):  # Ewwww....3 loops
                    if (self.door_open_checkpoints[i + 1][j] - datetime.timedelta(seconds=self.window_size)) \
                            <= time <= \
                            (self.door_open_checkpoints[i + 1][j] + datetime.timedelta(seconds=self.window_size)):
                        if self.entries.get(i + 1) is None:
                            self.entries[i + 1] = {}
                            self.entries[i + 1][j] = []
                        if self.entries.get(i + 1).get(j) is None:
                            self.entries.get(i + 1)[j] = []
                        self.entries[i + 1][j].append([j, row[1], row[2], row[3], row[4]])

    def get_door_opening_instances(self):
        """
        door_instance_dimension is for each door, for each instance, there are
        x, y and z values

        Example: door_instance_dimension[1][1]['x'] = -0.005
        Door 1, instance 1, x value

        """

        for k, v in self.entries.items():
            if self.door_instance_dimension.get(k) is None:
                self.door_instance_dimension[k] = {}
            for ks, vs in self.entries.get(k).items():
                if self.door_instance_dimension.get(k).get(ks) is None:
                    self.door_instance_dimension[k][ks] = {}
                    self.door_instance_dimension[k][ks]['x'] = []
                    self.door_instance_dimension[k][ks]['y'] = []
                    self.door_instance_dimension[k][ks]['z'] = []
                for vss in self.entries.get(k).get(ks):
                    self.door_instance_dimension[k][ks]['x'].append(vss[1])
                    self.door_instance_dimension[k][ks]['y'].append(vss[2])
                    self.door_instance_dimension[k][ks]['z'].append(vss[3])

    def write_entries_to_file(self):
        # Code to put the results (entries) into CSV
        for entry in self.entries:
            with open(self.path + self.user_id + "_" + self.sensor_name + "_door" + str(entry) + ".csv", "w+") as o_file:
                for e in self.entries[entry]:
                    csv.writer(o_file).writerows(self.entries[entry][e])


# pre_process = PreProcess()
#
# pre_process.store_checkpoints()
# pre_process.store_door_opening_instances()
# pre_process.write_entries_to_file()
# pre_process.get_door_opening_instances()

print()
