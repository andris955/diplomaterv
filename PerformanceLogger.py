from Logger import Logger
from collections import namedtuple
import time
import os
import config
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
import gym
from scipy.stats import hmean


class PerformanceLogger:
    def __init__(self, list_of_tasks, model_id):
        log_values = "elapsed_time total_train_steps " + " ".join(list_of_tasks)
        self.tasks = list_of_tasks
        self.logvalue = namedtuple("LogValue", log_values)
        self.logger = Logger(model_id, list_of_tasks)
        self.start_time = time.time()
        self.data_available = False
        self.transfer = True if os.path.exists(os.path.join(config.model_path, model_id)) else False
        self.scores = np.zeros([len(self.tasks)])
        self.avg_performance = 0
        self.harmonic_performance = 0

        self.env_for_test = {}
        for task in self.tasks:
            self.env_for_test[task] = DummyVecEnv([lambda: gym.make(task)])

        data = None
        if self.transfer:
            data, elapsed_time = self.logger.init_train_data()
            self.start_time -= elapsed_time

        self.train_step = {}
        self.episode_learn_per_task = {}
        if data is None:
            for game in list_of_tasks:
                self.train_step[game] = 0
                self.episode_learn_per_task = 0
        else:
            for game in list_of_tasks:
                self.train_step[game] = data[game]['train_step'].values[0]
                self.episode_learn_per_task = data[game]['episode_learn'].values[0]

    def log(self, task, episode_learnt, train_steps, total_train_steps, score, policy_loss, value_loss):
        self.train_step[task] += train_steps
        self.episode_learn_per_task[task] += 1
        log_value = self.logvalue(elapsed_time=int(time.time() - self.start_time),
                                  total_train_steps=total_train_steps, episodes_learnt=self.episode_learn_per_task[task],
                                  train_steps=self.train_step[task],
                                  relative_performance=np.around(score / config.target_performances[task], 2),
                                  score=np.around(score, 2), policy_loss=np.around(policy_loss, 2),
                                  value_loss=np.around(value_loss, 2))
        self.logger.log(task, log_value)
        self.data_available = True
        if episode_learnt % config.file_logging_frequency_in_episodes == 0 and self.data_available is True:
            self.logger.dump()

    def performance_test(self, n_games, amta, ta):
        performance = np.zeros(len(self.tasks))
        for i, task in enumerate(self.tasks):
            self.scores[i] = amta.play_n_game(amta.model, task, n_games, self.env_for_test[task])
            performance[i] = min(self.scores[i] / ta[task], 1)
        self.avg_performance = np.around(np.mean(performance), 2)
        self.harmonic_performance = np.around(hmean(performance), 2)
        return self.avg_performance, self.harmonic_performance
