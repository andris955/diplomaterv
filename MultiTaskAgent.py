import gym
import config
import os
import time
import copy
import numpy as np

from env_utils import make_atari_env
from scipy.stats import hmean


from Logger import Logger
from utils import CustomMessengerClass
from MultiTaskA2C import MultitaskA2C
from MultiTaskRunner import MultiTaskA2CRunner


class MultiTaskAgent:
    def __init__(self, model_id: str, policy: str, tasks: list, n_steps: int, n_cpus: int, n_episodes: int, tensorboard_logging, logging, env_kwargs):
        self.model_id = model_id
        self.policy = policy
        self.tasks = tasks
        self.n_steps = n_steps
        self.n_cpus = n_cpus
        self.n_episodes = n_episodes
        self.learning_envs = {}
        self.testing_envs = {}
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

        self.logger = None
        self.logvalue = None
        if self.logging:
            self.logvalue = CustomMessengerClass
            self.logger = Logger(self.model_id, self.tasks)

        data = None
        if self.transfer and self.logging:
            data, elapsed_time, total_episodes_learnt, total_timesteps, total_training_updates = self.logger.init_train_data()
            self.start_time -= elapsed_time
            self.total_episodes_learnt = int(total_episodes_learnt)
            self.total_timesteps = int(total_timesteps)
            self.total_training_updates = int(total_training_updates)

        self.training_updates = {}
        self.episodes_learnt = {}
        if data is None:
            for task in self.tasks:
                self.training_updates[task] = 0
                self.episodes_learnt[task] = 0
        else:
            for task in self.tasks:
                self.training_updates[task] = data[task]['training_updates'].values[0]
                self.episodes_learnt[task] = data[task]['episodes_learnt'].values[0]

        self.tbw = None
        self.writer = None
        self.model = None

        self.data_available = [False]*len(self.tasks)

        self.__setup_environments()
        self.__setup_model()
        self.__setup_runners()

    def __setup_environments(self):
        learning_env_kwargs = copy.deepcopy(self.env_kwargs)
        learning_env_kwargs['episode_life'] = False
        learning_env_kwargs['clip_rewards'] = False
        for task in self.tasks:
            env = make_atari_env(task, self.n_cpus, config.seed, wrapper_kwargs=self.env_kwargs)
            assert isinstance(env.action_space, gym.spaces.Discrete), "Error: all the input games must have Discrete action space"
            self.learning_envs[task] = env
            self.testing_envs[task] = make_atari_env(task, self.n_episodes, config.seed, wrapper_kwargs=learning_env_kwargs)

    def __setup_model(self):
        if not self.transfer:
            self.model = MultitaskA2C(self.policy, self.learning_envs, tensorboard_log=self.tb_log,
                                      full_tensorboard_log=(self.tb_log is not None), n_steps=self.n_steps)
        else:
            self.model, _ = MultitaskA2C.load(self.model_id, envs_to_set=self.learning_envs, transfer=True,
                                              total_training_updates=self.total_training_updates, total_timesteps=self.total_timesteps)

        self.tbw = self.model._setup_multitask_learn(self.model_id)
        if self.tbw is not None:
            self.writer = self.tbw.writer

    def __setup_runners(self):
        for task in self.tasks:
            self.runners[task] = MultiTaskA2CRunner(task, self.learning_envs[task],
                                                    self.model, n_steps=self.n_steps, gamma=0.99)

    def train_for_one_episode(self, task: str):
        runner = self.runners[task]
        episode_score, policy_loss, value_loss, episodes_training_updates = \
            self.model.multi_task_learn_for_one_episode(task, runner, self.writer)
        self.total_timesteps = self.model.num_timesteps
        self.episodes_learnt[task] += 1
        self.total_episodes_learnt += 1
        self.total_training_updates += int(episodes_training_updates)
        self.training_updates[task] += int(episodes_training_updates)
        print("{} episodes learnt: {}".format(task, self.episodes_learnt[task]))
        print("Total episodes learnt: {}".format(self.total_episodes_learnt))
        if self.logging and self.episodes_learnt[task] % config.logging_frequency_in_episodes == 0:
            print("Logging {}".format(task))
            policy_loss = round(float(policy_loss), 2)
            value_loss = round(float(value_loss), 2)
            log_value = self.logvalue(elapsed_time=int(time.time() - self.start_time),
                                      total_timesteps=self.total_timesteps,
                                      total_training_updates=self.total_training_updates,
                                      total_episodes_learnt=self.total_episodes_learnt,
                                      episodes_learnt=self.episodes_learnt[task],
                                      training_updates=self.training_updates[task],
                                      policy_loss=policy_loss,
                                      value_loss=value_loss)
            self.logger.log(task, log_value)
            self.data_available[self.tasks.index(task)] = True
        if self.logging and self.total_episodes_learnt % (config.logging_frequency_in_episodes*10) == 0 and all(self.data_available):
            self.logger.dump()
            print("Training information logged")

        return episode_score

    def test_performance(self, task: str):
        start = time.time()
        env = self.testing_envs[task]
        obs = env.reset()
        n_env = obs.shape[0]
        sum_reward = np.zeros(n_env)
        timesteps = np.zeros(n_env)
        state = None
        all_done = False
        done = None
        mask = np.asarray([False]*n_env)
        while not all_done:
            action, state = self.model.predict(task, obs, state, done)
            obs, reward, done, info = env.step(action)
            mask[np.where(done==True)] = True
            all_done = all(mask)
            timesteps += np.ones(n_env) * (1 - mask)
            sum_reward += reward * (1 - mask)
        if sum_reward == 0:  # harmonic mean needs greater than zero elements
            sum_reward = 0.1
        sum_reward = int(hmean(sum_reward))
        timesteps = int(np.mean(timesteps))
        duration = int(time.time()-start)
        print(task)
        print("Testing {} env ended in {} sec with {} fps".format(n_env, duration, int(n_env*timesteps/duration)))
        print("Performance: {}% timestep: {}\n".format(int(sum_reward/config.target_performances[task]*100), timesteps))
        return sum_reward, timesteps

    @staticmethod
    def _play_n_game(model, task: str, n_games: int, display=False):
        env = model.env_dict[task]
        timesteps = 0
        sum_reward = 0
        for i in range(n_games):
            obs = env.reset()
            done = None
            state = None
            while not done:
                action, state = model.predict(task, obs, state, done)
                obs, reward, done, info = env.step(action)
                timesteps += 1
                sum_reward += reward
                if display is True:
                    env.render()
                    time.sleep(0.005)
        sum_reward = int(sum_reward / n_games)
        if sum_reward == 0:  # harmonic mean needs greater than zero elements
            sum_reward = 0.1
        timesteps = int(timesteps / n_games)
        env.close()
        return sum_reward, timesteps

    @staticmethod
    def play(model_id, n_games=1, display=True):
        model, tasks = MultitaskA2C.load(model_id)
        for task in tasks:
            print(task)
            sum_reward, timesteps = MultiTaskAgent._play_n_game(model, task, n_games, display)
            print("Achieved score: {}".format(sum_reward))
            print("Timesteps: {}".format(timesteps))
            print("Relative performance: {}%".format(round(sum_reward/config.target_performances[task], 2)*100))

    def save_model(self, avg_performance: float, harmonic_performance: float, json_params: dict):
        json_params.update({
            "total_episodes_learnt": int(self.total_episodes_learnt),
            "total_timesteps": int(self.total_timesteps),
            "total_training_updates": int(self.total_training_updates),
        })
        base_path = os.path.join(config.model_path, self.model_id)
        id = "{:010}-{:1.2f}-{:1.2f}".format(self.model.num_timesteps, avg_performance, harmonic_performance)
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        try:
            self.model.save(base_path, id, json_params)
        except:
            print("Error saving the MultiTaskA2C model")

    def exit_tbw(self):
        if self.tbw is not None:
            self.tbw.exit()

    def flush_tbw(self):
        if self.tbw is not None:
            self.tbw.flush()


