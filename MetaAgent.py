from MetaA2C import MetaA2CModel
import numpy as np


class MetaAgent:
    def __init__(self, total_timesteps, n_batch, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        self.n_batch = n_batch
        self.meta_learner = MetaA2CModel(total_timesteps, n_input, n_output, n_batch)

        self.inputs = [np.zeros(self.n_input)] * self.n_batch
        self.actions = [np.zeros(self.n_output)] * self.n_batch
        self.rewards = [0]*n_batch
        self.values = [0.0]*n_batch

    def train_and_sample(self, game_input, reward):
        self.inputs.append(game_input)
        self.inputs.pop(0)
        self.rewards.append(reward)
        self.rewards.pop(0)
        action, value, neglogp = self.meta_learner.policy_model.step(state=np.asarray(self.inputs)) #TODO eloszlást kell hogy visszaadjon
        self.actions.append(action)
        self.actions.pop(0)
        self.values.append(value)
        self.values.pop(0)
        _, _, _ = self.meta_learner.train_step(inputs=np.asarray(self.inputs), rewards=self.rewards, actions=self.actions, values=self.values)
        return action
