import time
import os
import config
import numpy as np
from scipy.stats import hmean
from Logger import Logger
from utils import CustomMessengerClass


class PerformanceLogger:
    def __init__(self, tasks: list, model_id: str, envs_for_test: dict, ta: dict):
        tasks_score = [task + "_score" for task in tasks]
        tasks_performance = [task + "_performance" for task in tasks]
        interleaved_task_logvalues = [val for pair in zip(tasks_score, tasks_performance) for val in pair]
        log_values = "elapsed_time timestep " + " ".join(interleaved_task_logvalues)
        self.log_value_list = log_values.split(" ")
        self.tasks = tasks
        self.ta = ta
        self.logvalue = CustomMessengerClass
        self.name = "Global_performance"
        self.logger = Logger(model_id, [self.name])
        self.start_time = time.time()
        self.data_available = False
        self.transfer = True if os.path.exists(os.path.join(config.model_path, model_id)) and model_id else False
        self.scores = np.zeros(len(self.tasks))
        self.timesteps = np.zeros(len(self.tasks))
        self.performance = np.zeros(len(self.tasks))
        self.envs_for_test = envs_for_test
        self.logobject = self.logvalue(*self.log_value_list)
        self.worst_performing_task_timestep = 10_000

        if self.transfer:
            return_data, elapsed_time, total_episodes_learnt, total_timesteps, total_training_updates = self.logger.init_train_data()
            self.start_time -= int(elapsed_time)

    def log(self, timestep: int):
        update_dict = {}
        for field in self.logobject._fields:
            if field == "elapsed_time":
                update_dict[field] = time.time()-self.start_time
            elif field == "timestep":
                update_dict[field] = timestep
            else:
                task_name, field_type = field.split('_')
                if field_type == 'performance':
                    update_dict[field] = self.scores[self.tasks.index(task_name)]/self.ta[task_name]
                elif field_type == 'score':
                    update_dict[field] = self.scores[self.tasks.index(task_name)]

        self.logobject.__dict__.update(update_dict)
        self.logger.log(self.name, self.logobject)

    def dump(self):
        self.logger.dump()

    def performance_test(self, n_games: int, amta, ta: dict):
        min_performance = np.zeros(len(self.tasks))
        for i, task in enumerate(self.tasks):
            self.scores[i], self.timesteps[i] = amta._play_n_game(amta.model, task, n_games, env=self.envs_for_test[task])
            self.performance[i] = self.scores[i] / ta[task]
            min_performance[i] = min(self.scores[i] / ta[task], 1)
        index = self.performance.argmin()
        self.worst_performing_task_timestep = self.timesteps[index]
        print("-----------------------------------------------------------------")
        print("Worst performing task: {} with {} timestep".format(self.tasks[index], self.worst_performing_task_timestep))
        print("-----------------------------------------------------------------")
        avg_performance = np.around(np.mean(min_performance), 4)
        harmonic_performance = np.around(hmean(min_performance), 4)
        return avg_performance, harmonic_performance
