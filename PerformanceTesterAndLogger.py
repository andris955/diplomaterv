import time
import os
import config
import numpy as np

from scipy.stats import hmean
from Logger import Logger

from utils import CustomMessengerClass


class PerformanceTesterAndLogger:
    def __init__(self, tasks: list, model_id: str, ta: dict, logging: bool):
        self.tasks = tasks
        self.ta = ta
        self.start_time = time.time()
        self.data_available = False
        self.transfer = True if os.path.exists(os.path.join(config.model_path, model_id)) and model_id else False
        self.scores = np.zeros(len(self.tasks))
        self.timesteps = np.zeros(len(self.tasks))
        self.performance = np.zeros(len(self.tasks))
        self.worst_performing_task_timestep = 1000
        self.logging = logging

        if self.logging:
            tasks_score = [task + "_score" for task in tasks]
            tasks_performance = [task + "_performance" for task in tasks]
            interleaved_task_logvalues = [val for pair in zip(tasks_score, tasks_performance) for val in pair]
            log_values = "elapsed_time timestep " + " ".join(interleaved_task_logvalues)
            self.log_value_list = log_values.split(" ")
            self.logvalue = CustomMessengerClass
            self.name = "GlobalPerformance"
            self.logger = Logger(model_id, [self.name])
            self.logobject = self.logvalue(*self.log_value_list)
            if self.transfer:
                _, elapsed_time, _, _, _ = self.logger.init_train_data()
                self.start_time -= int(elapsed_time)

    def log(self, timestep: int):
        if self.logging:
            update_dict = {}
            for field in self.logobject._fields:
                if field == "elapsed_time":
                    update_dict[field] = int(time.time()-self.start_time)
                elif field == "timestep":
                    update_dict[field] = timestep
                else:
                    task_name, field_type = field.split('_')
                    if field_type == 'performance':
                        update_dict[field] = round(float(self.performance[self.tasks.index(task_name)]), 4)
                    elif field_type == 'score':
                        update_dict[field] = round(float(self.scores[self.tasks.index(task_name)]))

            self.logobject.__dict__.update(update_dict)
            self.logger.log(self.name, self.logobject)

    def dump(self):
        if self.logging:
            self.logger.dump()
            print("Global information logged")

    def performance_test(self, amta, ta: dict):
        min_performance = np.zeros(len(self.tasks))
        print("-----------------------------------------------------------------")
        for i, task in enumerate(self.tasks):
            self.scores[i], self.timesteps[i] = amta.test_performance(task)
            self.performance[i] = self.scores[i] / ta[task]
            min_performance[i] = min(self.scores[i] / ta[task], 1)
        print("-----------------------------------------------------------------")
        avg_performance = float(np.mean(min_performance))
        harmonic_performance = float(hmean(min_performance))
        return avg_performance, harmonic_performance
