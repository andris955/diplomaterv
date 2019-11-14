import time
import os
import config
import numpy as np
from scipy.stats import hmean
from Logger import Logger
from utils import CustomMessengerClass


class PerformanceLogger:
    def __init__(self, tasks: list, model_id: str, envs_for_test: dict, ta: dict):
        tasks_performance = [task + "_performance" for task in tasks]
        log_values = "elapsed_time timestep " + " ".join(tasks_performance)
        self.log_value_list = log_values.split(" ")
        self.tasks = tasks
        self.ta = ta
        self.logvalue = CustomMessengerClass
        self.name = "Global_performance"
        self.logger = Logger(model_id, [self.name])
        self.start_time = time.time()
        self.data_available = False
        self.transfer = True if os.path.exists(os.path.join(config.model_path, model_id)) and model_id else False
        self.scores = np.zeros([len(self.tasks)])
        self.avg_performance = 0
        self.harmonic_performance = 0
        self.env_for_test = envs_for_test
        self.logobject = self.logvalue(self.log_value_list)

        if self.transfer:
            _, elapsed_time = self.logger.init_train_data()
            self.start_time -= elapsed_time

    def log(self, timestep: int):
        update_dict = {}
        for field in self.logobject._fields:
            if field == "elapsed_time":
                update_dict[field] = time.time()-self.start_time
            elif field == "timestep":
                update_dict[field] = timestep
            else:
                task_name = field.split('_')[0]
                update_dict[field] = self.scores[self.tasks.index(task_name)]/self.ta[task_name]
        self.logobject.__dict__.update(update_dict)
        self.logger.log(self.name, self.logobject)

    def dump(self):
        self.logger.dump()

    def performance_test(self, n_games: int, amta, ta: dict):
        performance = np.zeros(len(self.tasks))
        for i, task in enumerate(self.tasks):
            self.scores[i] = amta.play_n_game(amta.model, task, n_games, self.env_for_test[task])
            performance[i] = min(self.scores[i] / ta[task], 1)
        self.avg_performance = np.around(np.mean(performance), 2)
        self.harmonic_performance = np.around(hmean(performance), 2)
        return self.avg_performance, self.harmonic_performance
