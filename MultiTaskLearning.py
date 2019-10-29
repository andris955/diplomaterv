import global_config
import numpy as np
from utils import one_hot, read_params
from stable_baselines.common import SetVerbosity
from MultiTaskAgent import MultiTaskAgent
from MetaAgent import MetaAgent


class MultiTaskLearning:
    def __init__(self, set_of_tasks, algorithm, policy, target_performances,
                 n_cpus, logging=True, transfer_id=None, tensorboard_logging=None,
                 verbose=1, lambda_=None):
        """

        :param set_of_tasks:
        :param algorithm: Chosen multi-task algorithm. Available: 'A5C','EA4C' ...
        """
        if transfer_id is not None:
            if set_of_tasks is not None:
                print("The given set of tasks is overwritten by the tasks used by the referred model (transfer_id).")
            params = read_params(transfer_id)
            self.tasks = params['tasks']
        else:
            self.tasks = set_of_tasks

        assert isinstance(target_performances, dict), "TargetPerformance must be a dictionary"
        self.ta = target_performances  # Target score in task Ti. This could be based on expert human performance or even published scores from other technical works

        self.algorithm = algorithm
        self.verbose = verbose
        self.logging = logging

        self.max_train_steps = global_config.max_train_steps  # Total number of training steps for the algorithm
        self.uniform_policy_steps = global_config.uniform_policy_steps  # Number of tasks to consider in the computation of r2.
        self.n = global_config.number_of_episodes_for_estimating  # Number of episodes which are used for estimating current average performance in any task Ti

        self.amta = MultiTaskAgent(algorithm, policy, self.tasks, global_config.n_steps, self.max_train_steps,
                                   n_cpus, transfer_id, tensorboard_logging=tensorboard_logging)
        meta_decider = None

        if self.algorithm == "A5C":
            self.__A5C_init()
        elif self.algorithm == "EA4C":
            self.__EA4C_init(meta_decider, lambda_)

    def __A5C_init(self):
        self.s = []  # List of last n scores that the multi-tasking agent scored during training on task Ti.
        self.a = []  # Average scores for every task
        self.m = []
        self.p = []  # Probability of training on an episode of task Ti next.
        self.performance = np.zeros(len(self.tasks))
        self.tau = global_config.tau  # Temperature hyper-parameter of the softmax task-selection non-parametric policy

        for _ in range(len(self.tasks)):
            self.p.append(1 / len(self.tasks))
            self.a.append(0.0)
            self.m.append(0.000000001)
        for i in range(len(self.tasks)):
            self.s.append([0.0 for _ in range(self.n)])

    def __A5C_train(self):
        with SetVerbosity(self.verbose):
            episode_learn = 0
            performance = 0
            while self.amta.model.train_step < self.max_train_steps:
                if self.amta.model.train_step > self.uniform_policy_steps:
                    for j in range(len(self.tasks)):
                        self.a[j] = sum(self.s[j])/self.n
                        self.m[j] = (self.ta[self.tasks[j]] - self.a[j]) / (self.ta[self.tasks[j]] * self.tau) # minél kisebb annál jobban teljesít az ágens az adott gamen
                        self.performance[j] = np.min((self.a[j]) / (self.ta[self.tasks[j]]), 1)
                    for j in range(len(self.tasks)):
                        self.p[j] = np.exp(self.m[j]) / (sum(np.exp(self.m)))
                if episode_learn % global_config.logging_frequency == 0:
                    performance = np.mean(self.performance)  # qam
                    self.amta.save_model(performance)
                    self.amta.flush_tbw()
                j = np.random.choice(np.arange(0, len(self.p)), p=self.p)
                ep_scores, train_steps = self.amta.train_for_one_episode(self.tasks[j], logging=self.logging)
                episode_learn += 1
                self.s[j].append(np.mean(ep_scores))
                if len(self.s[j]) > self.n:
                    self.s[j].pop(0)
            self.amta.save_model(performance)
            self.amta.exit_tbw()

    def __EA4C_init(self, meta_learning_agent, lambda_):
        self.s = []  # List of last n scores that the multi-tasking agent scored during training on task Ti.
        self.p = []  # Probability of training on an episode of task Ti next.
        self.c = np.zeros(len(self.tasks))  # Count of the number of training episodes of task Ti.
        self.r1 = 0
        self.r2 = 0  # First & second component of the reward for meta-learner, defined in lines 93-94
        self.reward = 0  # Reward for meta-learner
        self.s_avg = []  # Average scores for every task
        assert isinstance(meta_learning_agent, MetaAgent)
        self.ma = meta_learning_agent # Meta Learning MultiTaskAgent.
        self.lambda_ = lambda_ # Lambda weighting.
        for _ in range(len(self.tasks)):
            self.p.append(1 / len(self.tasks))
        for i in range(len(self.tasks)):
            self.s.append([0.0 for _ in range(self.n)])
        for i in range(len(self.tasks)):
            self.s_avg.append(0.0)

    # TODO megcsinálni
    def __EA4C_train(self):
        performance = 0
        for train_step in range(self.max_train_steps):
            j = np.random.choice(np.arange(0, len(self.p)), p=self.p)
            self.c[j] = self.c[j] + 1
            ep_scores, train_steps = self.amta.train_for_one_episode(self.tasks[j], logging=self.logging)
            self.s[j].append(np.mean(ep_scores))
            if len(self.s[j]) > self.n:
                self.s[j].pop(0)
            #TODO eddig jó
            for i in range(len(self.s_avg)):
                self.s_avg[i] = sum(self.s[i])/len(self.s[i])
            s_avg_norm = [self.s_avg[i] / self.ta[self.tasks[i]] for i in range(len(self.s_avg))]
            s_avg_norm.sort()
            s_min_l = s_avg_norm[0:self.uniform_policy_steps]
            performance = None
            self.r1 = 1 - self.s_avg[j] / self.c[j]
            self.r2 = 1 - np.average(np.clip(s_min_l, 0, 1))
            self.reward = self.lambda_ * self.r1 + (1 - self.lambda_) * self.r2
            self.p = self.ma.train_and_sample(game_input=[self.c / sum(self.c), self.p, one_hot(j, len(self.tasks))], reward=self.reward)
        self.amta.save_model(performance)
        self.amta.exit_tbw()

    def train(self):
        if self.algorithm == "A5C":
            self.__A5C_train()
        elif self.algorithm == "EA4C":
            self.__EA4C_train()
