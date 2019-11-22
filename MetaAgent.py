from MetaA2C import MetaA2CModel
import numpy as np
import os
import utils
import config
from stable_baselines import logger
from utils import read_params


class MetaAgent:
    def __init__(self, model_id: str, n_steps: int, input_len: int, output_len: int):
        self.model_id = model_id
        self.transfer = True if os.path.exists(os.path.join(config.model_path, model_id)) and model_id else False
        if self.transfer:
            params = read_params(model_id, "meta")
            self.input_len = params['input_length']
            self.output_len = params['output_length']
            self.n_steps = params['n_steps']
            self.meta_learner = MetaA2CModel.load(self.model_id, self.input_len, self.output_len)
        else:
            self.input_len = input_len
            self.output_len = output_len
            self.n_steps = n_steps
            self.meta_learner = MetaA2CModel(self.input_len, self.output_len, self.n_steps)

        self.inputs = [np.zeros(self.input_len)] * self.n_steps
        self.train_step = 0

    def sample(self, game_input):
        self.inputs.append(game_input)
        self.inputs.pop(0)
        flat_param, value, neglogp = self.meta_learner.step(input_state=np.asarray(self.inputs))
        prob_dist = utils.softmax(flat_param[0, :])
        return prob_dist, value, neglogp

    def train(self, action, value, reward):
        rewards = np.asarray([reward])
        actions = np.asarray([action])
        policy_loss, value_loss, policy_entropy = self.meta_learner.train_step(inputs=np.asarray(self.inputs), rewards=rewards,
                                                                               actions=actions, values=value)
        self.train_step += 1

        if self.train_step % (config.stdout_logging_frequency_in_train_steps // 10) == 0:
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
