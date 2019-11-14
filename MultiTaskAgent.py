import config
from Logger import Logger

import gym
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from MultiTaskA2C import MultitaskA2C
from MultiTaskRunner import MultiTaskA2CRunner

import os
from collections import namedtuple
import time
import numpy as np


class MultiTaskAgent:
    def __init__(self, model_id: str, policy, tasks, n_steps, max_timesteps, n_cpus, tensorboard_logging, logging):
        self.tasks = tasks
        self.max_timesteps = max_timesteps
        self.n_cpus = n_cpus
        self.n_steps = n_steps
        self.model_id = model_id
        self.sub_proc_environments = {}
        self.policy = policy
        self.logging = logging
        if tensorboard_logging:
            self.tb_log = config.tensorboard_log
        else:
            self.tb_log = None

        self.start_time = time.time()
        self.episodes_learnt = 0
        self.total_timesteps = 0
        self.total_training_updates = 0

        self.transfer = True if os.path.exists(os.path.join(config.model_path, model_id)) and model_id else False

        if self.logging:
            self.logvalue = namedtuple("LogValue", "elapsed_time total_timesteps total_training_updates episode_learnt training_updates "
                                                   "relative_performance score policy_loss value_loss")
            self.logger = Logger(self.model_id, self.tasks)

        data = None
        if self.transfer:
            data, elapsed_time = self.logger.init_train_data()
            self.start_time -= elapsed_time
            self.episodes_learnt = 0 #TODO transfernél beolvasni
            self.total_timesteps = 0
            self.total_training_updates = 0

        self.tbw = None
        self.writer = None
        self.model = None

        self.data_available = [False]*len(self.tasks)

        self.__setup_environments()
        self.__setup_model()
        self.__setup_runners()

        self.training_updates = {}
        self.episodes_learnt_per_task = {}
        if data is None:
            for task in self.tasks:
                self.training_updates[task] = 0
                self.episodes_learnt_per_task[task] = 0
        else:
            for task in self.tasks:
                self.training_updates[task] = data[task]['train_step'].values[0]
                self.episodes_learnt_per_task[task] = data[task]['episode_learn'].values[0]

    def __setup_environments(self):
        for task in self.tasks:
            env = SubprocVecEnv([lambda: gym.make(task) for _ in range(self.n_cpus)])
            assert isinstance(env.action_space, gym.spaces.Discrete), "Error: all the input games must have Discrete action space"
            self.sub_proc_environments[task] = env

    def __setup_model(self):
        if not self.transfer:
            self.model = MultitaskA2C(self.policy, self.sub_proc_environments, tensorboard_log=self.tb_log,
                                      full_tensorboard_log=(self.tb_log is not None), n_steps=self.n_steps)
        else:
            self.model, _ = MultitaskA2C.load(self.model_id, envs_to_set=self.sub_proc_environments, transfer=True)

        self.tbw = self.model._setup_multitask_learn(self.model_id, self.max_timesteps)
        if self.tbw is not None:
            self.writer = self.tbw.writer

    def __setup_runners(self):
        self.runners = {}
        for task in self.tasks:
            self.runners[task] = MultiTaskA2CRunner(task, self.sub_proc_environments[task],
                                                    self.model, n_steps=self.n_steps, gamma=0.99)

    def train_for_one_episode(self, task: str, max_episode_timestep: int):
        runner = self.runners[task]
        episode_scores, policy_loss, value_loss, episodes_training_updates, full_episode = \
            self.model.multi_task_learn_for_one_episode(task, runner, max_episode_timestep, self.writer) # TODO befejezni
        self.total_timesteps = self.model.num_timesteps
        if self.logging:
            self.episodes_learnt += 1
            self.total_training_updates += episodes_training_updates
            self.training_updates[task] += episodes_training_updates
            self.episodes_learnt_per_task[task] += 1
            log_value = self.logvalue(elapsed_time=int(time.time() - self.start_time),
                                      total_timesteps=self.total_timesteps,
                                      total_training_updates=self.model.train_step,
                                      episode_learnt=self.episodes_learnt_per_task[task],
                                      training_updates=self.training_updates[task],
                                      relative_performance=np.around(np.mean(episode_scores) / config.target_performances[task], 2),
                                      score=np.around(np.mean(episode_scores), 2), policy_loss=np.around(policy_loss, 2),
                                      value_loss=np.around(value_loss, 2))
            self.logger.log(task, log_value)
            self.data_available[self.tasks.index(task)] = True
            if self.episodes_learnt % config.file_logging_frequency_in_episodes == 0 and all(self.data_available) is True:
                self.logger.dump()

        return episode_scores

    @staticmethod
    def play_n_game(model, task: str, n_games: int, display=False, env=None):
        sum_reward = 0
        if env is None:
            env = DummyVecEnv([lambda: gym.make(task)])
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
            sum_reward = MultiTaskAgent.play_n_game(model, task, n_games, display)
            print("Achieved score: {}".format(sum_reward[0]))
            print("Relative performance: {}%".format(np.around(sum_reward[0]/config.target_performances[task], 2)*100))

    def save_model(self, avg_performance, harmonic_performance):
        try:
            base_path = os.path.join(config.model_path, self.model_id)
            id = "{:08}-{:1.2f}-{:1.2f}".format(self.model.num_timesteps, avg_performance, harmonic_performance)
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


