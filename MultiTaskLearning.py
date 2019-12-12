import config
import datetime
import numpy as np

from utils import one_hot, read_params, softmax, make_date_id

from stable_baselines.common import SetVerbosity
from MultiTaskAgent import MultiTaskAgent
from MetaAgent import MetaAgent
from PerformanceTesterAndLogger import PerformanceTesterAndLogger


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
            self.algorithm = params['multitask_algorithm']
            n_steps = params['n_steps']
            self.uniform_policy_steps = params["uniform_policy_steps"]  # Number of tasks to consider in the computation of r2.
            n_episodes = params["number_of_episodes_for_estimating"]  # Number of episodes which are used for estimating current performance in any task Ti
            if n_episodes > n_cpus:
                n_episodes = n_cpus
                print("Number of episodes for estimating is overwritten by the number of cpus")
            self.lambda_ = params['lambda']
            self.tau = params['tau']
            env_kwargs = params['env_kwargs']
            self.meta_n_steps = params['meta_n_steps']
            self.best_harmonic_performance = params['best_qhm']
            self.best_avg_performance = params['best_qam']
        else:
            self.tasks = tasks
            if algorithm == "EA4C" or algorithm == "A5C":
                self.algorithm = algorithm
            else:
                raise ValueError("algorithm can only be A5C or EA4C.")
            now = make_date_id(datetime.datetime.now())
            n_steps = config.n_steps
            self.model_id = self.algorithm + "_" + now
            self.tau = config.tau  # Temperature hyper-parameter of the softmax task-selection non-parametric policy
            self.uniform_policy_steps = config.uniform_policy_steps  # Number of tasks to consider in the computation of r2.
            n_episodes = config.number_of_episodes_for_estimating  # Number of episodes which are used for estimating current performance in any task Ti
            if n_episodes > n_cpus:
                n_episodes = n_cpus
                print("Number of episodes for estimating is overwritten by the number of cpus")
            self.lambda_ = config.meta_lambda
            self.meta_n_steps = config.meta_n_steps
            self.best_avg_performance = 0.01
            self.best_harmonic_performance = 0.01
            env_kwargs = {
                'episode_life': True,
                'clip_rewards': True,
                'frame_stack': policy != "lstm",
                'scale': False,
            }

        self.ta = config.target_performances  # Target score in task Ti. This could be based on expert human performance or even published scores from other technical works
        self.verbose = verbose

        self.json_params = json_params
        self.json_params.update({
            'multitask_algorithm': self.algorithm,
            'tasks': self.tasks,
            "seed": config.seed,
            "model_id": self.model_id,
            "lambda": self.lambda_,
            "uniform_policy_steps": self.uniform_policy_steps,
            "number_of_episodes_for_estimating": n_episodes,
            "best_avg_performance": self.best_avg_performance,
            "best_harmonic_performance": self.best_harmonic_performance,
            "meta_n_steps": self.meta_n_steps,
            "evaluation_frequency_in_episodes": config.evaluation_frequency_in_episodes,
            "tau": self.tau,
            'env_kwargs': env_kwargs,
            "best_qam": self.best_avg_performance,
            "best_qhm": self.best_harmonic_performance,
        })

        self.amta = MultiTaskAgent(self.model_id, policy, self.tasks, n_steps, n_cpus, n_episodes, tensorboard_logging, logging, env_kwargs=env_kwargs)
        self.performance_logger = PerformanceTesterAndLogger(tasks=self.tasks, model_id=self.model_id, ta=self.ta, logging=logging)

        self.p = np.ones(len(self.tasks)) * (1 / len(self.tasks))  # Probability of training on an episode of task Ti next.
        self.a = np.zeros(len(self.tasks))  # Average scores for every task

        if self.algorithm == "A5C":
            self.__A5C_init()
        elif self.algorithm == "EA4C":
            self.__EA4C_init()

    def __A5C_init(self):
        self.m = []
        for _ in range(len(self.tasks)):
            self.m.append(1.0)

    def __A5C_train(self):
        with SetVerbosity(self.verbose):
            while 1:
                if self.amta.total_episodes_learnt % config.evaluation_frequency_in_episodes == 0:
                    avg_performance, harmonic_performance = self.performance_logger.performance_test(amta=self.amta, ta=self.ta)
                    self.performance_logger.log(self.amta.total_timesteps)
                    if self.amta.total_episodes_learnt % (config.evaluation_frequency_in_episodes*5) == 0:
                        self.performance_logger.dump()
                    if avg_performance > self.best_avg_performance or harmonic_performance > self.best_avg_performance:
                        if avg_performance > self.best_avg_performance:
                            self.json_params['best_qam'] = avg_performance
                        if harmonic_performance > self.best_harmonic_performance:
                            self.json_params['best_qhm'] = harmonic_performance
                        self.amta.save_model(avg_performance, harmonic_performance, self.json_params)
                        print("Model saved")
                    self.amta.flush_tbw()
                    if avg_performance > self.best_avg_performance:
                        self.best_avg_performance = avg_performance
                    if harmonic_performance > self.best_harmonic_performance:
                        self.best_harmonic_performance = harmonic_performance
                    self.a = self.performance_logger.scores
                    for i in range(len(self.tasks)):
                        self.m[i] = max(self.ta[self.tasks[i]] - self.a[i], 0) / (self.ta[self.tasks[i]] * self.tau)  # the less the better
                    if self.amta.total_timesteps > self.uniform_policy_steps:
                        self.p = softmax(np.asarray(self.m))
                        print(self.tasks)
                        print(self.p)
                j = np.random.choice(np.arange(0, len(self.p)), p=self.p)
                self.amta.train_for_one_episode(self.tasks[j])

    def __EA4C_init(self):
        self.l = len(self.tasks) // 2
        self.ma = MetaAgent(self.model_id, self.meta_n_steps, 3 * len(self.tasks), len(self.tasks))

    def __EA4C_train(self):
        with SetVerbosity(self.verbose):
            while 1:
                if self.amta.total_episodes_learnt % self.meta_n_steps == 0:
                    self.ma.train()
                if self.amta.total_episodes_learnt % config.evaluation_frequency_in_episodes == 0:
                    avg_performance, harmonic_performance = self.performance_logger.performance_test(amta=self.amta, ta=self.ta)
                    self.performance_logger.log(self.amta.total_timesteps)
                    if self.amta.total_episodes_learnt % (config.evaluation_frequency_in_episodes*5) == 0:
                        self.performance_logger.dump()
                    if avg_performance > self.best_avg_performance or harmonic_performance > self.best_avg_performance:
                        if avg_performance > self.best_avg_performance:
                            self.json_params['best_qam'] = avg_performance
                        if harmonic_performance > self.best_harmonic_performance:
                            self.json_params['best_qhm'] = harmonic_performance
                        self.amta.save_model(avg_performance, harmonic_performance, self.json_params)
                        print("Model saved")
                    self.ma.save_model(self.amta.model.total_train_steps)
                    self.amta.flush_tbw()
                    if avg_performance > self.best_avg_performance:
                        self.best_avg_performance = avg_performance
                    if harmonic_performance > self.best_harmonic_performance:
                        self.best_harmonic_performance = harmonic_performance
                    s_avg_norm = list(self.performance_logger.performance)
                    s_avg_norm.sort()
                    s_min_l = s_avg_norm[0:self.l]
                    r2 = np.mean(np.clip(s_min_l, 0, 1))
                    # mint a játékoknál a runner egy menetet változatlan súllyal játszik és itt a "súly" a kiértékelés eredménye.
                action = j = np.random.choice(np.arange(0, len(self.p)), p=self.p)
                self.amta.train_for_one_episode(self.tasks[j])
                r1 = 1 - self.performance_logger.performance[j]
                print("r1: {}".format(r1))
                print("r2: {}".format(r2))
                reward = self.lambda_ * r1 + (1 - self.lambda_) * r2
                game_input = np.concatenate(
                    [np.asarray(list(self.amta.episodes_learnt.values())) / np.sum(np.asarray(list(self.amta.episodes_learnt.values()))), self.p,
                     one_hot(j, len(self.tasks))])  # Input 3*len(tasks)
                self.p, value, neglogp = self.ma.sample(game_input=game_input, action=action, reward=reward)

    def train(self):
        if self.algorithm == "A5C":
            self.__A5C_train()
        elif self.algorithm == "EA4C":
            self.__EA4C_train()
