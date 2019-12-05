import numpy as np
import os
import utils
import config
import copy

from MetaA2C import MetaA2CModel

from stable_baselines import logger
from utils import read_params
from collections import deque
from stable_baselines.a2c.utils import discount_with_dones


class MetaAgent:
    def __init__(self, model_id: str, n_steps: int, input_len: int, output_len: int):
        self.model_id = model_id
        self.transfer = True if os.path.exists(os.path.join(config.model_path, model_id)) and model_id else False
        if self.transfer:
            params = read_params(model_id, "meta")
            self.input_len = params['input_length']
            self.output_len = params['output_length']
            self.n_steps = params['n_steps']
            self.window_size = self.n_steps//3
            self.gamma = params['gamma']
            self.meta_learner = MetaA2CModel.load(self.model_id, self.input_len, self.output_len)
        else:
            self.input_len = input_len
            self.output_len = output_len
            self.n_steps = n_steps
            self.window_size = self.n_steps//3
            self.gamma = 0.8
            self.meta_learner = MetaA2CModel(self.input_len, self.output_len, self.n_steps, window_size=self.window_size, gamma=self.gamma)

        self.input = deque([np.zeros(self.input_len)] * self.window_size, maxlen=self.window_size)
        self.input_batch = deque([np.zeros([self.window_size, self.input_len])] * self.n_steps, maxlen=self.n_steps)
        self.reward_batch = deque([0.0] * self.n_steps, maxlen=self.n_steps)
        self.action_batch = deque([0] * self.n_steps, maxlen=self.n_steps)
        self.value_batch = deque([0.0] * self.n_steps, maxlen=self.n_steps)
        self.train_step = 0

    def sample(self, game_input, action, reward):
        self.input.append(game_input)
        self.action_batch.append(action)
        self.reward_batch.append(reward)
        self.input_batch.append(copy.deepcopy(np.asarray(self.input)))
        flat_param, value, neglogp = self.meta_learner.step(input_state=np.asarray([self.input]))
        prob_dist = utils.softmax(flat_param[0, :])
        self.value_batch.append(value[0])
        return prob_dist, value, neglogp

    def train(self):
        inputs = np.asarray(self.input_batch)
        rewards = list(self.reward_batch)
        actions = np.asarray(self.action_batch)
        values = np.asarray(self.value_batch)
        discounted_rewards = np.asarray(discount_with_dones(rewards + [self.value_batch[-1]], [False]*(len(rewards)+1), self.gamma)[:-1])
        policy_loss, value_loss, policy_entropy = self.meta_learner.train_step(inputs=inputs, discounted_rewards=discounted_rewards,
                                                                               actions=actions, values=values)
        self.train_step += 1

        print("---------------------------{}---------------------------".format("Meta agent"))
        logger.record_tabular("value_loss", float(value_loss))
        logger.record_tabular("policy_loss", float(policy_loss))
        logger.dump_tabular()

    def save_model(self, train_step):
        try:
            base_path = os.path.join(config.model_path, self.model_id)
            id = "{:08}".format(train_step)
            if not os.path.exists(base_path):
                os.mkdir(base_path)
            self.meta_learner.save(base_path, id)
        except:
            print("Error saving the Meta model")
