import config
import numpy as np
from utils import one_hot, read_params
from stable_baselines.common import SetVerbosity
from env_utils import make_atari_env
from MultiTaskAgent import MultiTaskAgent
from MetaAgent import MetaAgent
import utils
import datetime
from PerformanceLogger import PerformanceLogger
from collections import deque


class MultiTaskLearning:
    def __init__(self, tasks: list, algorithm: str, policy: str, n_cpus: int, json_params: dict,
                 logging=True, model_id=None, tensorboard_logging=False, verbose=1):
        """

        :param algorithm: Chosen multi-task algorithm. Available: 'A5C','EA4C' ...
        """
        self.algorithm = algorithm
        if model_id:
            if tasks is not None:
                print("The given set of tasks is overwritten by the tasks used by the referred model (transfer_id).")
            params = read_params(model_id, "multitask")
            self.tasks = params['tasks']
            self.model_id = model_id
        else:
            self.tasks = tasks
            now = str(datetime.datetime.now())[2:16]
            now = now.replace(' ', '_')
            now = now.replace(':', '_')
            now = now.replace('-', '_')
            self.model_id = self.algorithm + "_" + now

        self.ta = config.target_performances  # Target score in task Ti. This could be based on expert human performance or even published scores from other technical works

        self.verbose = verbose
        self.logging = logging
        self.best_avg_performance = 0.1
        self.best_harmonic_performance = 0.05

        self.n_steps = config.n_steps
        self.n_cpus = n_cpus

        self.uniform_policy_steps = config.uniform_policy_steps  # Number of tasks to consider in the computation of r2.
        self.n = config.number_of_episodes_for_estimating  # Number of episodes which are used for estimating current
                                                           # average performance in any task Ti
        env_kwargs = {
            'episode_life': True,
            'clip_rewards': False,
            'frame_stack': True,
            'scale': False,
        }

        self.amta = MultiTaskAgent(self.model_id, policy, self.tasks, self.n_steps, self.n_cpus, tensorboard_logging, logging, env_kwargs=env_kwargs)

        env_for_test = {}
        for task in self.tasks:
            env_for_test[task] = make_atari_env(task, 1, config.seed, wrapper_kwargs=env_kwargs)

        self.performance_logger = PerformanceLogger(tasks=self.tasks, model_id=self.model_id, envs_for_test=env_for_test, ta=self.ta)

        meta_n_steps = 5 # TODO ennek utána nézni.
        meta_decider = MetaAgent(self.model_id, meta_n_steps, 3 * len(self.tasks), len(self.tasks))

        self.p = np.ones(len(self.tasks)) * (1 / len(self.tasks))  # Probability of training on an episode of task Ti next.
        self.s = []  # List of last n scores that the multi-tasking agent scored during training on task Ti.
        self.a = []  # Average scores for every task

        for _ in range(len(self.tasks)):
            self.s.append(deque([1.0], 1))
            self.a.append(0.0)

        self.json_params = json_params
        self.json_params.update({
            'algorithm': self.algorithm,
            'tasks': self.tasks,
            "seed": config.seed,
            "model_id": self.model_id,
            "uniform_policy_steps": self.uniform_policy_steps,
            "number_of_episodes_for_estimating": self.n,
            "best_avg_performance": self.best_avg_performance,
            "best_harmonic_performance": self.best_harmonic_performance,
        })
        self.json_params.update(env_kwargs)

        if self.algorithm == "A5C":
            self.__A5C_init()
        elif self.algorithm == "EA4C":
            self.__EA4C_init(meta_decider, config.meta_lambda)

    def __A5C_init(self):
        self.m = []
        self.tau = config.tau  # Temperature hyper-parameter of the softmax task-selection non-parametric policy
        for _ in range(len(self.tasks)):
            self.m.append(1.0)

    def __A5C_train(self):
        with SetVerbosity(self.verbose):
            while 1:
                for j in range(len(self.tasks)):
                    self.a[j] = sum(self.s[j])/len(self.s[j])
                    self.m[j] = max(self.ta[self.tasks[j]] - self.a[j], 0) / (self.ta[self.tasks[j]] * self.tau)  # minél kisebb annál jobban teljesít az ágens az adott gamen
                if self.amta.total_episodes_learnt > self.uniform_policy_steps:
                    self.p = utils.softmax(np.asarray(self.m))
                if self.amta.total_episodes_learnt % config.file_logging_frequency_in_episodes == 0 and self.amta.total_episodes_learnt > 0:
                    avg_performance, harmonic_performance = self.performance_logger.performance_test(n_games=self.n, amta=self.amta, ta=self.ta)
                    if self.logging:
                        self.performance_logger.log(self.amta.total_timesteps)
                        self.performance_logger.dump()
                    if avg_performance > self.best_avg_performance or harmonic_performance > self.best_avg_performance:
                        self.amta.save_model(avg_performance, harmonic_performance, self.json_params)
                    self.amta.flush_tbw()
                    if avg_performance > self.best_avg_performance:
                        self.best_avg_performance = avg_performance
                    if harmonic_performance > self.best_harmonic_performance:
                        self.best_harmonic_performance = harmonic_performance
                j = np.random.choice(np.arange(0, len(self.p)), p=self.p)
                max_episode_timesteps = int(1.2 * self.performance_logger.worst_performing_task_timestep)
                episode_score = self.amta.train_for_one_episode(self.tasks[j], max_episode_timesteps=max_episode_timesteps)
                self.s[j].append(episode_score)

    def __EA4C_init(self, meta_decider, lambda_):
        self.l = len(self.tasks) // 2
        self.training_episodes = np.zeros(len(self.tasks))  # Count of the number of training episodes of task Ti.
        assert isinstance(meta_decider, MetaAgent)
        self.ma = meta_decider  # Meta Learning MultiTaskAgent.
        self.lambda_ = lambda_  # Lambda weighting.

    def __EA4C_train(self):
        with SetVerbosity(self.verbose):
            episodes_learnt = 0
            action = j = 0
            value = np.array([0])
            while 1:
                max_episode_timesteps = int(1.2 * self.performance_logger.worst_performing_task_timestep)
                ep_score = self.amta.train_for_one_episode(self.tasks[j], max_episode_timesteps)
                self.training_episodes[j] = self.training_episodes[j] + 1
                episodes_learnt += 1
                self.s[j].append(ep_score)
                for i in range(len(self.a)):
                    self.a[i] = sum(self.s[i]) / len(self.s[i])
                if episodes_learnt % config.file_logging_frequency_in_episodes == 0 and episodes_learnt > 0:
                    avg_performance, harmonic_performance = self.performance_logger.performance_test(n_games=self.n, amta=self.amta, ta=self.ta)
                    if self.logging:
                        self.performance_logger.log(self.amta.total_timesteps)
                        self.performance_logger.dump()
                    if avg_performance > self.best_avg_performance or harmonic_performance > self.best_avg_performance:
                        self.amta.save_model(avg_performance, harmonic_performance, self.json_params)
                    self.ma.save_model(self.amta.model.train_step)
                    self.amta.flush_tbw()
                    if avg_performance > self.best_avg_performance:
                        self.best_avg_performance = avg_performance
                    if harmonic_performance > self.best_harmonic_performance:
                        self.best_harmonic_performance = harmonic_performance
                s_avg_norm = [self.a[i] / self.ta[self.tasks[i]] for i in range(len(self.a))]
                s_avg_norm.sort()
                s_min_l = s_avg_norm[0:self.l]
                r1 = 1 - self.a[j] / self.training_episodes[j]
                r2 = 1 - np.mean(np.clip(s_min_l, 0, 1))
                reward = self.lambda_ * r1 + (1 - self.lambda_) * r2
                self.ma.train(action, value, reward)
                self.p, value, neglogp = self.ma.sample(game_input=np.concatenate([self.training_episodes / np.sum(self.training_episodes),
                                                        self.p, one_hot(j, len(self.tasks))])) # Input 3*len(tasks)
                action = j = np.random.choice(np.arange(0, len(self.p)), p=self.p)

    def train(self):
        if self.algorithm == "A5C":
            self.__A5C_train()
        elif self.algorithm == "EA4C":
            self.__EA4C_train()

