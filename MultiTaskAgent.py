from MultiTaskA2C import MultitaskA2C
from MultiTaskRunner import MultiTaskA2CRunner
import config
from Logger import Logger

import gym
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv

import os
from collections import namedtuple
import time
import numpy as np


class MultiTaskAgent:
    def __init__(self, model_id, policy, list_of_tasks, n_steps, max_train_steps, n_cpus, tensorboard_logging, logging):
        self.list_of_tasks = list_of_tasks
        self.max_train_steps = max_train_steps
        self.n_cpus = n_cpus
        self.sub_proc_environments = {}
        self.policy = policy
        self.logging = logging
        if tensorboard_logging:
            self.tb_log = config.tensorboard_log
        else:
            self.tb_log = None
        self.start_time = time.time()
        self.n_steps = n_steps

        self.model_id = model_id
        self.transfer = True if os.path.exists(os.path.join(config.model_path, model_id)) else False

        if self.logging:
            self.LogValue = namedtuple("LogValue", "elapsed_time total_train_step episode_learn train_step relative_performance scores policy_loss value_loss")
            self.logger = Logger(self.model_id, self.list_of_tasks)

        data = None
        if self.transfer:
            data, elapsed_time = self.logger.init_train_data()
            self.start_time -= elapsed_time

        self.tbw = None
        self.writer = None
        self.model = None

        self.episode_learn = 0
        self.data_available = [False]*len(self.list_of_tasks)

        self.__setup_environments()
        self.__setup_model()
        self.__setup_runners()

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

    def __setup_environments(self):
        for task in self.list_of_tasks:
            env = SubprocVecEnv([lambda: gym.make(task) for i in range(self.n_cpus)])
            assert isinstance(env.action_space, gym.spaces.Discrete), "Error: all the input games must have Discrete action space"
            self.sub_proc_environments[task] = env

    def __setup_model(self):
        if not self.transfer:
            self.model = MultitaskA2C(self.policy, self.sub_proc_environments, tensorboard_log=self.tb_log,
                                      full_tensorboard_log=(self.tb_log is not None), n_steps=self.n_steps)
        else:
            self.model, _ = MultitaskA2C.load(self.model_id, envs_to_set=self.sub_proc_environments, transfer=True)

        self.tbw = self.model._setup_multitask_learn(self.model_id, self.max_train_steps)
        if self.tbw is not None:
            self.writer = self.tbw.writer

    def __setup_runners(self):
        self.runners = {}
        for task in self.list_of_tasks:
            self.runners[task] = MultiTaskA2CRunner(task, self.sub_proc_environments[task],
                                                    self.model, n_steps=self.n_steps, gamma=0.99)

    def train_for_one_episode(self, task):
        runner = self.runners[task]
        ep_scores, policy_loss, value_loss, train_steps = self.model.multi_task_learn_for_one_episode(task, runner, self.writer)
        if self.logging:
            self.episode_learn += 1
            self.train_step[task] += train_steps
            self.episode_learn_per_task[task] += 1
            log_value = self.LogValue(elapsed_time=int(time.time()-self.start_time),
                                      total_train_step=self.model.train_step, episode_learn=self.episode_learn_per_task[task],
                                      train_step=self.train_step[task],
                                      relative_performance=np.around(np.mean(ep_scores) / config.target_performances[task], 2),
                                      scores=np.around(np.mean(ep_scores), 2), policy_loss=np.around(policy_loss, 2),
                                      value_loss=np.around(value_loss, 2))
            self.logger.log(task, log_value)
            self.data_available[self.list_of_tasks.index(task)] = True
            if self.episode_learn % config.file_logging_frequency_in_episodes == 0 and all(self.data_available) is True:
                self.logger.dump()

        return ep_scores, train_steps

    @staticmethod
    def play_for_one_episode(model, env, n_games: int, display=False):
        sum_reward = 0
        if isinstance(env, str):
            env = DummyVecEnv([lambda: gym.make(env)])
        elif not isinstance(env, DummyVecEnv):
                raise ValueError("Env must be a DummyVecEnv or a str!")
        obs = env.reset()
        done = False
        state = None
        mask = None
        for i in range(n_games):
            while not done:
                action, state = model.predict(task, obs, state, mask)
                obs, reward, done, info = env.step(action)
                sum_reward += reward
                if display is True:
                    env.render()
        sum_reward = sum_reward / n_games
        env.close()
        return sum_reward

    @staticmethod
    def play(model_id, n_games=1, display=True):
        model, tasks = MultitaskA2C.load(model_id)
        for task in tasks:
            print(task)
            sum_reward = MultiTaskAgent.play_for_one_episode(model, task, n_games, display)
            print("Achieved score: {}".format(sum_reward))

    def save_model(self, avg_performance, harmonic_performance):
        try:
            base_path = os.path.join(config.model_path, self.model_id)
            id = "{:08}-{:1.2f}-{:1.2f}".format(self.model.train_step, avg_performance, harmonic_performance)
            if not os.path.exists(base_path):
                os.mkdir(base_path)
            self.model.save(base_path, id)
        except:
            print("Error saving the Multi-task model")

    def exit_tbw(self):
        if self.tbw is not None:
            self.tbw.exit()

    def flush_tbw(self):
        if self.tbw is not None:
            self.tbw.flush()


