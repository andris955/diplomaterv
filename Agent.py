from MultiTaskA2C import MultitaskA2C
from runner import myA2CRunner
from MultiTaskPolicy import MultiTaskA2CPolicy
import gym
from stable_baselines.common.vec_env import SubprocVecEnv
import global_config
import datetime
import os
from logger import Logger
from collections import namedtuple
import time
import numpy as np


class Agent:
    def __init__(self, algorithm, list_of_games, max_train_steps, n_cpus, transfer_id, tensorboard_logging):
        self.algorithm = algorithm
        self.list_of_games = list_of_games
        self.max_train_steps = max_train_steps
        self.n_cpus = n_cpus
        self.sub_proc_environments = {}
        self.policy = MultiTaskA2CPolicy
        self.tb_log = tensorboard_logging
        self.start_time = time.time()

        now = str(datetime.datetime.now())[2:16]
        now = now.replace(' ', '_')
        now = now.replace(':', '_')
        now = now.replace('-', '_')
        self.initialize_time = now

        self.transfer_id = None

        if transfer_id is not None:
            self.model_name = transfer_id
            self.transfer_id = transfer_id
        else:
            self.model_name = self.algorithm + "_" + self.initialize_time

        self.LogValue = namedtuple("LogValue", "elapsed_time total_train_step train_step relative_performance scores policy_loss value_loss")
        self.logger = Logger(self.model_name, self.list_of_games)

        data = None
        if transfer_id:
            data, elapsed_time = self.logger.init_train_data()
            self.start_time -= elapsed_time

        self.tbw = None
        self.writer = None
        self.model = None

        self.episode_learn = 0
        self.data_available = [False]*len(self.list_of_games)

        self.__setup_environments()
        self.__setup_model()
        self.__setup_runners()

        #TODO total train step bugos logolásnál
        self.train_step = {}
        if data is None:
            for game in list_of_games:
                self.train_step[game] = 0
        else:
            for game in list_of_games:
                self.train_step[game] = data[game]['train_step'].values[0]

    def __setup_environments(self):
        for game in self.list_of_games:
            env = SubprocVecEnv([lambda: gym.make(game) for i in range(self.n_cpus)])
            assert isinstance(env.action_space, gym.spaces.Discrete), "Error: all the input games must have Discrete action space"
            self.sub_proc_environments[game] = env

    def __setup_model(self):
        if self.transfer_id is None:
            self.model = MultitaskA2C(self.policy, self.sub_proc_environments, verbose=1, tensorboard_log=self.tb_log,
                                      full_tensorboard_log=(self.tb_log is not None), n_steps=global_config.n_steps)
        else:
            self.model, _ = MultitaskA2C.load(self.transfer_id, envs_to_set=self.sub_proc_environments, transfer=True)
        self.tbw = self.model._setup_multitask_learn(self.algorithm, self.max_train_steps, self.initialize_time)
        if self.tbw is not None:
            self.writer = self.tbw.writer

    def __setup_runners(self):
        self.runners = {}
        for environment in self.list_of_games:
            self.runners[environment] = myA2CRunner(environment, self.sub_proc_environments[environment],
                                                    self.model, n_steps=global_config.n_steps, gamma=0.99)

    def train_for_one_episode(self, game, logging=True):
        runner = self.runners[game]
        ep_scores, policy_loss, value_loss, train_steps = self.model.multi_task_learn_for_one_episode(game, runner, self.writer)
        if logging:
            self.episode_learn += 1
            self.train_step[game] += train_steps
            log_value = self.LogValue(elapsed_time=int(time.time()-self.start_time),
                                      total_train_step=self.model.train_step, train_step=self.train_step[game],
                                      relative_performance=np.mean(ep_scores) / global_config.target_performances[game], scores=np.mean(ep_scores),
                                      policy_loss=policy_loss, value_loss=value_loss)
            self.logger.log(game, log_value)
            self.data_available[self.list_of_games.index(game)] = True
            if self.episode_learn % global_config.logging_frequency == 0 and all(self.data_available) is True:
                self.logger.dump()

        return ep_scores, train_steps

    @staticmethod
    def play(model_id, max_number_of_games, display=False):
        model, games = MultitaskA2C.load(model_id)
        for game in games:
            number_of_games = 0
            sum_reward = 0
            env = gym.make(game)
            obs = env.reset()
            while number_of_games < max_number_of_games:
                action = model.predict(game, obs)
                obs, rewards, done, info = env.step(action)
                if done:
                    obs = env.reset()
                    number_of_games += 1
                    print(sum_reward)
                    sum_reward = 0
                sum_reward += rewards
                if display is True:
                    env.render()

    def save_model(self, performance):
        try:
            base_path = "./data/models/" + self.model_name
            name = "{:08}-{:1.2f}".format(self.model.train_step, performance)
            if not os.path.exists(base_path):
                os.mkdir(base_path)
            self.model.save(base_path, name)
        except:
            print("Error saving the model")

    def exit_tbw(self):
        if self.tbw is not None:
            self.tbw.exit()

    def flush_tbw(self):
        if self.tbw is not None:
            self.tbw.flush()


