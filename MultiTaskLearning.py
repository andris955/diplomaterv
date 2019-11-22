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
import copy


class MultiTaskLearning:
    def __init__(self, tasks: list, algorithm: str, policy: str, n_cpus: int, json_params: dict,
                 logging=True, model_id=None, tensorboard_logging=False, verbose=1):
        """

        :param algorithm: Chosen multi-task algorithm. Available: 'A5C','EA4C' ...
        """

        if model_id:
            print("The given inputs are overwritten by the referred model's params (model_id).")
            params = read_params(model_id, "multitask")
            self.tasks = params['tasks']
            self.model_id = model_id
            self.algorithm = params['algorithm']
            self.n_steps = params['n_steps']
            self.timestep_coeff = params['timestep_coeff']
            self.uniform_policy_steps = params["uniform_policy_steps"]  # Number of tasks to consider in the computation of r2.
            self.n = params["number_of_episodes_for_estimating"]  # Number of episodes which are used for estimating current performance in any task Ti
            self.lambda_ = params['lambda']
            env_kwargs = params['env_kwargs']
        else:
            self.tasks = tasks
            self.algorithm = algorithm
            now = str(datetime.datetime.now())[2:16]
            now = now.replace(' ', '_')
            now = now.replace(':', '_')
            now = now.replace('-', '_')
            self.n_steps = config.n_steps
            self.timestep_coeff = config.timestep_coeff
            self.model_id = self.algorithm + "_" + now
            self.uniform_policy_steps = config.uniform_policy_steps  # Number of tasks to consider in the computation of r2.
            self.n = config.number_of_episodes_for_estimating  # Number of episodes which are used for estimating current performance in any task Ti
            self.lambda_ = config.meta_lambda
            env_kwargs = {
                'episode_life': True,
                'clip_rewards': False,
                'frame_stack': policy != "lstm",
                'scale': False,
            }

        self.ta = config.target_performances  # Target score in task Ti. This could be based on expert human performance or even published scores from other technical works

        self.verbose = verbose
        self.logging = logging
        self.best_avg_performance = 0.01
        self.best_harmonic_performance = 0.01

        self.n_cpus = n_cpus

        self.json_params = json_params
        self.json_params.update({
            'algorithm': self.algorithm,
            'tasks': self.tasks,
            "seed": config.seed,
            "model_id": self.model_id,
            "timestep_coeff": self.timestep_coeff,
            "lambda": self.lambda_,
            "uniform_policy_steps": self.uniform_policy_steps,
            "number_of_episodes_for_estimating": self.n,
            "best_avg_performance": self.best_avg_performance,
            "best_harmonic_performance": self.best_harmonic_performance,
        })
        self.json_params.update({"env_kwargs": copy.deepcopy(env_kwargs)})

        self.amta = MultiTaskAgent(self.model_id, policy, self.tasks, self.n_steps, self.n_cpus, tensorboard_logging, logging, env_kwargs=env_kwargs)

        env_kwargs['episode_life'] = False
        envs_for_test = {}
        for task in self.tasks:
            envs_for_test[task] = make_atari_env(task, 1, config.seed, wrapper_kwargs=env_kwargs)

        self.performance_logger = PerformanceLogger(tasks=self.tasks, model_id=self.model_id, envs_for_test=envs_for_test, ta=self.ta)

        self.p = np.ones(len(self.tasks)) * (1 / len(self.tasks))  # Probability of training on an episode of task Ti next.
        self.a = []  # Average scores for every task

        for _ in range(len(self.tasks)):
            self.a.append(0.0)

        if self.algorithm == "A5C":
            self.__A5C_init()
        elif self.algorithm == "EA4C":
            self.__EA4C_init()

    def __A5C_init(self):
        self.m = []
        self.tau = config.tau  # Temperature hyper-parameter of the softmax task-selection non-parametric policy
        for _ in range(len(self.tasks)):
            self.m.append(1.0)

    def __A5C_train(self):
        with SetVerbosity(self.verbose):
            while 1:
                if self.amta.total_episodes_learnt % config.file_logging_frequency_in_episodes == 0:
                    avg_performance, harmonic_performance = self.performance_logger.performance_test(n_games=self.n, amta=self.amta, ta=self.ta)
                    if self.logging:
                        self.performance_logger.log(self.amta.total_timesteps)
                        self.performance_logger.dump()
                    if avg_performance > self.best_avg_performance or harmonic_performance > self.best_avg_performance:
                        self.amta.save_model(avg_performance, harmonic_performance, self.json_params)
                        print("Model saved")
                    self.amta.flush_tbw()
                    if avg_performance > self.best_avg_performance:
                        self.best_avg_performance = avg_performance
                    if harmonic_performance > self.best_harmonic_performance:
                        self.best_harmonic_performance = harmonic_performance
                    for j in range(len(self.tasks)):
                        self.a[j] = self.performance_logger.scores[j]
                        self.m[j] = max(self.ta[self.tasks[j]] - self.a[j], 0) / (self.ta[self.tasks[j]] * self.tau)  # the less the better
                    if self.amta.total_episodes_learnt > self.uniform_policy_steps:
                        self.p = utils.softmax(np.asarray(self.m))
                j = np.random.choice(np.arange(0, len(self.p)), p=self.p)
                max_episode_timesteps = int(self.timestep_coeff * self.performance_logger.worst_performing_task_timestep)
                self.amta.train_for_one_episode(self.tasks[j], max_episode_timesteps=max_episode_timesteps)

    def __EA4C_init(self):
        self.l = len(self.tasks) // 2
        meta_n_steps = 5  # TODO ennek utána nézni.
        self.ma = MetaAgent(self.model_id, meta_n_steps, 3 * len(self.tasks), len(self.tasks))

    def __EA4C_train(self):
        with SetVerbosity(self.verbose):
            action = j = 0
            value = np.array([0])
            while 1:
                if self.amta.total_episodes_learnt % config.file_logging_frequency_in_episodes == 0:
                    avg_performance, harmonic_performance = self.performance_logger.performance_test(n_games=self.n, amta=self.amta, ta=self.ta)
                    if self.logging:
                        self.performance_logger.log(self.amta.total_timesteps)
                        self.performance_logger.dump()
                    if avg_performance > self.best_avg_performance or harmonic_performance > self.best_avg_performance:
                        self.amta.save_model(avg_performance, harmonic_performance, self.json_params)
                    self.ma.save_model(self.amta.model.total_train_steps)
                    self.amta.flush_tbw()
                    if avg_performance > self.best_avg_performance:
                        self.best_avg_performance = avg_performance
                    if harmonic_performance > self.best_harmonic_performance:
                        self.best_harmonic_performance = harmonic_performance
                    for j in range(len(self.tasks)):
                        self.a[j] = self.performance_logger.scores[j]
                    s_avg_norm = [self.a[i] / self.ta[self.tasks[i]] for i in range(len(self.a))]
                    s_avg_norm.sort()
                    s_min_l = s_avg_norm[0:self.l]
                max_episode_timesteps = int(self.timestep_coeff * self.performance_logger.worst_performing_task_timestep)
                self.amta.train_for_one_episode(self.tasks[j], max_episode_timesteps)
                r1 = 1 - self.a[j] / self.amta.episodes_learnt[self.tasks[j]]
                r2 = 1 - np.mean(np.clip(s_min_l, 0, 1))
                reward = self.lambda_ * r1 + (1 - self.lambda_) * r2
                self.ma.train(action, value, reward)
                self.p, value, neglogp = self.ma.sample(game_input=np.concatenate(
                    [np.asarray(list(self.amta.episodes_learnt.values())) / np.sum(np.asarray(list(self.amta.episodes_learnt.values()))), self.p,
                     one_hot(j, len(self.tasks))])) # Input 3*len(tasks)
                action = j = np.random.choice(np.arange(0, len(self.p)), p=self.p)

    def train(self):
        if self.algorithm == "A5C":
            self.__A5C_train()
        elif self.algorithm == "EA4C":
            self.__EA4C_train()

