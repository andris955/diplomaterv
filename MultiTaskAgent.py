import config
from Logger import Logger

import gym
from env_utils import make_atari_env
from MultiTaskA2C import MultitaskA2C
from MultiTaskRunner import MultiTaskA2CRunner

import os
from utils import CustomMessengerClass
import time
import numpy as np


class MultiTaskAgent:
    def __init__(self, model_id: str, policy: str, tasks: list, n_steps: int, n_cpus: int, tensorboard_logging, logging, env_kwargs):
        self.model_id = model_id
        self.policy = policy
        self.tasks = tasks
        self.n_steps = n_steps
        self.n_cpus = n_cpus
        self.sub_proc_environments = {}
        self.runners = {}
        self.env_kwargs = env_kwargs
        self.logging = logging
        if tensorboard_logging:
            self.tb_log = config.tensorboard_log
        else:
            self.tb_log = None

        self.start_time = time.time()
        self.total_episodes_learnt = 0
        self.total_timesteps = 0
        self.total_training_updates = 0

        self.transfer = True if os.path.exists(os.path.join(config.model_path, self.model_id)) and self.model_id else False

        if self.logging:
            self.logvalue = CustomMessengerClass
            self.logger = Logger(self.model_id, self.tasks)

        data = None
        if self.transfer:
            data, elapsed_time, total_episodes_learnt, total_timesteps, total_training_updates = self.logger.init_train_data()
            self.start_time -= elapsed_time
            self.total_episodes_learnt = total_episodes_learnt
            self.total_timesteps = total_timesteps
            self.total_training_updates = total_training_updates

        self.training_updates = {}
        self.episodes_learnt = {}
        if data is None:
            for task in self.tasks:
                self.training_updates[task] = 0
                self.episodes_learnt[task] = 0
        else:
            for task in self.tasks:
                self.training_updates[task] = data[task]['train_step'].values[0]
                self.episodes_learnt[task] = data[task]['episode_learn'].values[0]

        self.tbw = None
        self.writer = None
        self.model = None

        self.data_available = [False]*len(self.tasks)

        self.__setup_environments()
        self.__setup_model()
        self.__setup_runners()

    def __setup_environments(self):
        for task in self.tasks:
            env = make_atari_env(task, self.n_cpus, config.seed, wrapper_kwargs=self.env_kwargs)
            # env.reset()
            # for _ in range(1):
            #     import matplotlib.pyplot as plt
            #     actions = [env.action_space.sample()]*self.n_cpus
            #     obs,_,_,_ = env.step(actions)
            #     env.render()
            #     plt.imshow(obs[0,:,:,0])
            #     plt.show()
            assert isinstance(env.action_space, gym.spaces.Discrete), "Error: all the input games must have Discrete action space"
            self.sub_proc_environments[task] = env

    def __setup_model(self):
        if not self.transfer:
            self.model = MultitaskA2C(self.policy, self.sub_proc_environments, tensorboard_log=self.tb_log,
                                      full_tensorboard_log=(self.tb_log is not None), n_steps=self.n_steps)
        else:
            self.model, _ = MultitaskA2C.load(self.model_id, envs_to_set=self.sub_proc_environments, transfer=True,
                                              total_train_steps=self.total_training_updates, num_timesteps=self.total_timesteps)

        self.tbw = self.model._setup_multitask_learn(self.model_id)
        if self.tbw is not None:
            self.writer = self.tbw.writer

    def __setup_runners(self):
        for task in self.tasks:
            self.runners[task] = MultiTaskA2CRunner(task, self.sub_proc_environments[task],
                                                    self.model, n_steps=self.n_steps, gamma=0.99)

    def train_for_one_episode(self, task: str, max_episode_timesteps: int):
        runner = self.runners[task]
        episode_score, policy_loss, value_loss, episodes_training_updates = \
            self.model.multi_task_learn_for_one_episode(task, runner, max_episode_timesteps, self.writer)
        self.total_timesteps = self.model.num_timesteps
        if self.logging:
            self.total_episodes_learnt += 1
            self.episodes_learnt[task] += 1
            self.total_training_updates += episodes_training_updates
            self.training_updates[task] += episodes_training_updates
            log_value = self.logvalue(elapsed_time=int(time.time() - self.start_time),
                                      total_timesteps=self.total_timesteps,
                                      total_training_updates=self.total_training_updates,
                                      total_episodes_learnt=self.total_episodes_learnt,
                                      episodes_learnt=self.episodes_learnt[task],
                                      training_updates=self.training_updates[task],
                                      # relative_performance=np.around(episode_score / config.target_performances[task], 2),
                                      # score=np.around(episode_score, 2),
                                      policy_loss=np.around(policy_loss, 2),
                                      value_loss=np.around(value_loss, 2))
            self.logger.log(task, log_value)
            self.data_available[self.tasks.index(task)] = True
            if self.total_episodes_learnt % config.file_logging_frequency_in_episodes == 0 and all(self.data_available) is True:
                self.logger.dump()

        return episode_score

    @staticmethod
    def _play_n_game(model, task: str, n_games: int, display=False, env=None):
        sum_reward = 0
        if env is None:
            env = make_atari_env(task, 1, config.seed)
        obs = env.reset()
        done = False
        state = None
        mask = None
        timesteps = 0
        for i in range(n_games):
            while not done:
                action, state = model.predict(task, obs, state, mask)
                obs, reward, done, info = env.step(action)
                timesteps += 1
                sum_reward += reward
                if display is True:
                    env.render()
        sum_reward = sum_reward / n_games
        timesteps = timesteps / n_games
        env.close()
        return sum_reward, timesteps

    @staticmethod
    def play(model_id, n_games=1, display=True):
        model, tasks = MultitaskA2C.load(model_id)
        for task in tasks:
            print(task)
            sum_reward = MultiTaskAgent._play_n_game(model, task, n_games, display)
            print("Achieved score: {}".format(sum_reward[0]))
            print("Relative performance: {}%".format(np.around(sum_reward[0]/config.target_performances[task], 2)*100))

    def save_model(self, avg_performance: float, harmonic_performance: float, json_params: dict):
        json_params.update({
            "total_episodes_learnt": self.total_episodes_learnt,
            "total_timesteps": self.total_timesteps,
            "total_training_updates": self.total_training_updates,
        })
        base_path = os.path.join(config.model_path, self.model_id)
        id = "{:08}-{:1.2f}-{:1.2f}".format(self.model.num_timesteps, avg_performance, harmonic_performance)
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        try:
            self.model.save(base_path, id)
        except:
            print("Error saving the MultiTaskA2C model")

    def exit_tbw(self):
        if self.tbw is not None:
            self.tbw.exit()

    def flush_tbw(self):
        if self.tbw is not None:
            self.tbw.flush()


