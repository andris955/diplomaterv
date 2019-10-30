import config
import numpy as np
from utils import one_hot, read_params
from stable_baselines.common import SetVerbosity
from MultiTaskAgent import MultiTaskAgent
from MetaAgent import MetaAgent
import utils
import datetime
from scipy.stats import hmean

class MultiTaskLearning:
    def __init__(self, set_of_tasks, algorithm, policy, target_performances, n_cpus,
                 logging=True, model_id=None, tensorboard_logging=False, verbose=1):
        """

        :param set_of_tasks:
        :param algorithm: Chosen multi-task algorithm. Available: 'A5C','EA4C' ...
        """
        self.algorithm = algorithm
        if model_id is not None:
            if set_of_tasks is not None:
                print("The given set of tasks is overwritten by the tasks used by the referred model (transfer_id).")
            params = read_params(model_id, "multitask")
            self.tasks = params['tasks']
            self.model_id = model_id
        else:
            self.tasks = set_of_tasks
            now = str(datetime.datetime.now())[2:16]
            now = now.replace(' ', '_')
            now = now.replace(':', '_')
            now = now.replace('-', '_')
            self.model_id = self.algorithm + "_" + now

        assert isinstance(target_performances, dict), "TargetPerformance must be a dictionary"
        self.ta = target_performances  # Target score in task Ti. This could be based on expert human performance or even published scores from other technical works

        self.verbose = verbose
        self.logging = logging
        self.best_avg_performance = 0.1
        self.best_harmonic_performance = 0.0

        self.n_steps = config.n_steps
        self.n_cpus = n_cpus
        self.performance = np.zeros(len(self.tasks))

        self.max_train_steps = config.max_train_steps  # Total number of training steps for the algorithm
        self.uniform_policy_steps = config.uniform_policy_steps  # Number of tasks to consider in the computation of r2.
        self.n = config.number_of_episodes_for_estimating  # Number of episodes which are used for estimating current average performance in any task Ti

        self.amta = MultiTaskAgent(self.model_id, policy, self.tasks, self.n_steps, self.max_train_steps,
                                   self.n_cpus, tensorboard_logging, self.logging)

        meta_n_steps = 5 #TODO ennek utána nézni, valamint a n_stepsnek a runnerben és a multitaska2cben és policyben tanitásnál és predictnél
        meta_decider = MetaAgent(self.model_id, self.max_train_steps, meta_n_steps, 3*len(self.tasks), len(self.tasks))

        self.p = np.ones(len(self.tasks)) * (1 / len(self.tasks))  # Probability of training on an episode of task Ti next.
        self.s = []  # List of last n scores that the multi-tasking agent scored during training on task Ti.
        self.a = []  # Average scores for every task

        for _ in range(len(self.tasks)):
            self.s.append([1.0]) # Harmonic performance miatt nem lehet 0.0
            self.a.append(0.0)

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
            episode_learn = 0
            avg_performance = 0
            harmonic_performance = 0
            while self.amta.model.train_step < self.max_train_steps:
                for j in range(len(self.tasks)):
                    self.a[j] = sum(self.s[j])/len(self.s[j])
                    self.m[j] = (self.ta[self.tasks[j]] - self.a[j]) / (self.ta[self.tasks[j]] * self.tau)  # minél kisebb annál jobban teljesít az ágens az adott gamen
                    self.performance[j] = min((self.a[j]) / (self.ta[self.tasks[j]]), 1)
                if self.amta.model.train_step > self.uniform_policy_steps:
                    self.p = utils.softmax(np.asarray(self.m))
                avg_performance = np.mean(self.performance)  # qam
                harmonic_performance = hmean(self.performance)
                if (episode_learn % config.file_logging_frequency_in_episodes == 0 or
                    (avg_performance > self.best_avg_performance or harmonic_performance > self.best_harmonic_performance)) \
                        and episode_learn > 0:
                    self.amta.save_model(avg_performance, harmonic_performance)
                    self.amta.flush_tbw()
                if avg_performance > self.best_avg_performance:
                    self.best_avg_performance = avg_performance
                if harmonic_performance > self.best_harmonic_performance:
                    self.best_harmonic_performance = harmonic_performance
                j = np.random.choice(np.arange(0, len(self.p)), p=self.p)
                ep_scores, train_steps = self.amta.train_for_one_episode(self.tasks[j])
                episode_learn += 1
                self.s[j].append(np.mean(ep_scores))
                if len(self.s[j]) > self.n:
                    self.s[j].pop(0)
            self.amta.save_model(avg_performance, harmonic_performance)
            self.amta.exit_tbw()

    def __EA4C_init(self, meta_decider, lambda_):
        self.l = len(self.tasks) // 2
        self.training_episodes = np.zeros(len(self.tasks))  # Count of the number of training episodes of task Ti.
        assert isinstance(meta_decider, MetaAgent)
        self.ma = meta_decider  # Meta Learning MultiTaskAgent.
        self.lambda_ = lambda_  # Lambda weighting.

    def __EA4C_train(self):
        with SetVerbosity(self.verbose):
            episode_learn = 0
            action = j = 0
            value = np.array([0])
            avg_performance = 0
            harmonic_performance = 0
            while self.amta.model.train_step < self.max_train_steps:
                ep_scores, train_steps = self.amta.train_for_one_episode(self.tasks[j])
                self.training_episodes[j] = self.training_episodes[j] + 1
                episode_learn += 1
                self.s[j].append(np.mean(ep_scores))
                if len(self.s[j]) > self.n:
                    self.s[j].pop(0)
                for i in range(len(self.a)):
                    self.a[i] = sum(self.s[i]) / len(self.s[i])
                    self.performance[i] = min((self.a[i]) / (self.ta[self.tasks[i]]), 1)
                avg_performance = np.mean(self.performance)  # qam
                harmonic_performance = hmean(self.performance)
                if (episode_learn % config.file_logging_frequency_in_episodes == 0 or
                    (avg_performance > self.best_avg_performance or harmonic_performance > self.best_harmonic_performance)) \
                        and episode_learn > 0:
                    self.amta.save_model(avg_performance, harmonic_performance)
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

            self.amta.save_model(avg_performance, harmonic_performance)
            self.ma.save_model(self.amta.model.train_step)
            self.amta.exit_tbw()

    def train(self):
        if self.algorithm == "A5C":
            self.__A5C_train()
        elif self.algorithm == "EA4C":
            self.__EA4C_train()
