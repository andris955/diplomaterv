import os
import pandas as pd
import config
import sys


class Logger:
    def __init__(self, id: str, tasks: list):
        self.id = id
        self.tasks = tasks
        self.folder_path = os.path.join(config.log_path, self.id)
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        self.fields = None
        self.data = None
        self.pd_data = None

        self.__pd_data_init()

    def init_train_data(self):
        return_data = {}
        elapsed_time = 0
        total_episodes_learnt = 0
        total_timesteps = 0
        total_training_updates = 0
        files = [file for file in os.listdir(self.folder_path) if file[-4:] == ".csv" and not file[0:6] == 'Global']
        for file_name in files:
            data = pd.read_csv(os.path.join(self.folder_path, file_name), sep=";")
            return_data[file_name[:-4]] = data.tail(1) # only the last row
            if data.tail(1)['elapsed_time'].values[0] > elapsed_time:
                elapsed_time = data.tail(1)['elapsed_time'].values[0]
            if data.tail(1)['total_episodes_learnt'].values[0] > total_episodes_learnt:
                total_episodes_learnt = data.tail(1)['total_episodes_learnt'].values[0]
            if data.tail(1)['total_timesteps'].values[0] > total_timesteps:
                total_timesteps = data.tail(1)['total_timesteps'].values[0]
            if data.tail(1)['total_training_updates'].values[0] > total_training_updates:
                total_training_updates = data.tail(1)['total_training_updates'].values[0]

        return return_data, elapsed_time, total_episodes_learnt, total_timesteps, total_training_updates

    def __save_path(self, task: str):
        return os.path.join(self.folder_path, task + ".csv")

    def __pd_data_init(self):
        self.data = {}
        self.pd_data = {}
        for task in self.tasks:
            self.pd_data[task] = None
            self.data[task] = {}

    def __make_pd_data(self):
        for task in self.tasks:
            if self.pd_data[task] is None:
                self.pd_data[task] = pd.DataFrame(self.data[task])
            else:
                print("Error in Logger.py")

    def log(self, task: str, values):
        try:
            if self.fields is None:
                self.fields = values._fields
            elif self.fields != values._fields:
                print("The fields of the named tuples must be consistent")
        except:
            print("Values must be a namedtuple")
            sys.exit()

        if self.data[task] == {}:
            for field in self.fields:
                self.data[task][field] = []

        for i, field in enumerate(self.fields):
            self.data[task][field].append(values[i])

    def dump(self):
        self.__make_pd_data()
        for task in self.tasks:
            if os.path.exists(self.__save_path(task)):
                with open(self.__save_path(task), 'a') as f:
                    self.pd_data[task].to_csv(f, sep=";", index=False, header=False)
            else:
                with open(self.__save_path(task), 'a') as f:
                    self.pd_data[task].to_csv(f, sep=";", index=False)
        self.__pd_data_init()


