import global_config
import local_config
import numpy as np
import utils


class MultiTasking():
    def __init__(self, SetOfTasks, Algorithm, NumberOfEpisodesForEstimating, TargetPerformances, l, MaxSteps, ActiveSamplingMultiTaskAgent, lam=None, MetaDecider=None):
        """

        :param SetOfTasks:
        :param Algorithm: Chosen multi-task algorithm. Available: 'A5C','EA4C' ...
        """
        self.algorithm = Algorithm
        if self.algorithm == "A5C":
            self.__A5C_init(SetOfTasks, NumberOfEpisodesForEstimating, TargetPerformances, l, MaxSteps, ActiveSamplingMultiTaskAgent)
        elif self.algorithm == "EA4C":
            self.__EA4C_init(SetOfTasks, NumberOfEpisodesForEstimating, TargetPerformances, l, MaxSteps, ActiveSamplingMultiTaskAgent, MetaDecider, lam)

    def __A5C_init(self, SetOfTasks, NumberOfEpisodesForEstimating, TargetPerformances, l, MaxSteps, ActiveSamplingMultiTaskAgent):
        assert isinstance(SetOfTasks, list)
        self.T = SetOfTasks
        self.k = len(self.T)  # Number of tasks
        assert isinstance(TargetPerformances, list)
        self.ta = TargetPerformances  # Target score in task Ti. This could be based on expert human performance or even published scores from other technical works
        assert isinstance(NumberOfEpisodesForEstimating, int)
        self.n = NumberOfEpisodesForEstimating  # Number of episodes which are used for estimating current average performance in any task Ti
        self.l = l  # Number of training steps for which a uniformly random policy is executed for task selection. At the end of l training steps, the agent must have learned on ≥ n
        self.t = MaxSteps  # Total number of training steps for the algorithm
        self.s = []  # List of last n scores that the multi-tasking agent scored during training on task Ti.
        self.a = []  # Average scores for every task
        self.m = []
        self.amta = ActiveSamplingMultiTaskAgent  # The Active Sampling multi-tasking agent
        self.p = []  # Probability of training on an episode of task Ti next.
        self.tau = global_config.tau  # Temperature hyper-parameter of the softmax task-selection non-parametric policy

        for _ in range(self.k):
            self.p.append(1/self.k)
        for i in range(self.k):
            self.s.append([0 for _ in range(self.n)])

    def __A5C_train(self):
        for train_step in range(self.t):
            if train_step > self.l:
                for i, task in enumerate(self.T):
                    self.a[i] = sum(self.s[i])/self.n
                    self.m[i] = (self.ta[i] - self.a[i]) / (self.ta[i] * self.tau)
                    self.p[i] = np.exp(self.m[i]) / (sum(np.exp(self.m)))
            j = self.p.index(max(self.p))
            score = self.amta.train_for_one_episode(self.T[j])
            self.s[j].append(score)
            if len(self.s[j]) > self.n:
                self.s[j].pop(0)
        self.amta.save_model()

    def __EA4C_init(self, SetOfTasks, NumberOfEpisodesForEstimating, TargetPerformances, l, MaxSteps, ActiveSamplingMultiTaskAgent, MetaLearningAgent, lam):
        assert isinstance(SetOfTasks, list)
        self.T = SetOfTasks
        self.k = len(self.T)  # Number of tasks
        assert isinstance(TargetPerformances, list)
        self.ta = TargetPerformances  # Target score in task Ti. This could be based on expert human performance or even published scores from other technical works
        assert isinstance(NumberOfEpisodesForEstimating, int)
        self.n = NumberOfEpisodesForEstimating  # Number of episodes which are used for estimating current average performance in any task Ti
        self.t = MaxSteps  # Total number of training steps for the algorithm
        self.s = []  # List of last n scores that the multi-tasking agent scored during training on task Ti.
        self.p = []  # Probability of training on an episode of task Ti next.
        self.c = np.zeros(self.k)  # Count of the number of training episodes of task Ti.
        self.r1 = 0
        self.r2 = 0  # First & second component of the reward for meta-learner, defined in lines 93-94
        self.r = 0  # Reward for meta-learner
        self.l = l  # Number of tasks to consider in the computation of r2.
        self.s_avg = []  # Average scores for every task
        self.amta = ActiveSamplingMultiTaskAgent  # The Active Sampling multi-tasking agent
        self.ma = MetaLearningAgent # Meta Learning Agent.
        self.lam = lam
        for _ in range(self.k):
            self.p.append(1/self.k)
        for i in range(self.k):
            self.s.append([0 for _ in range(self.n)])
        for i in range(self.k):
            self.s_avg.append(0)

    def __EA4C_train(self):
        for train_step in range(self.t):
            j = self.p.index(max(self.p))
            self.c[j] = self.c[j] + 1
            score = self.amta.train_for_one_episode(self.T[j])
            self.s[j].append(score)
            if len(self.s[j]) > self.n:
                self.s[j].pop(0)
            for i in range(len(self.s_avg)):
                self.s_avg[i] = sum(self.s[i])/len(self.s[i])
            s_avg_norm = [self.s_avg[i]/self.ta[i] for i in range(len(self.s_avg))]
            s_avg_norm.sort()
            s_min_l = s_avg_norm[0:self.l]
            self.r1 = 1 - self.s_avg[j] / self.c[j]
            self.r2 = 1 - np.average(np.clip(s_min_l, 0, 1))
            self.r = self.lam * self.r1 + (1 - self.lam) * self.r2
            self.p = self.ma.train_and_sample(state=[self.c/sum(self.c), self.p, utils.one_hot(j, self.k)], reward=self.r)
        self.amta.save_model()

    def train(self):
        if self.algorithm == "A5C":
            self.__A5C_train()
        elif self.algorithm == "EA4C":
            self.__EA4C_train()

    def get_agent(self):
        return self.amta
