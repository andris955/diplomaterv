import tensorflow as tf
import os
import numpy as np
import utils
import config

from stable_baselines.a2c.utils import mse
from stable_baselines.common import set_global_seeds

import tf_utils
from MetaPolicy import MetaLstmActorCriticPolicy
from utils import Scheduler


class MetaA2CModel:
    """
    The Meta A2C (Advantage Actor Critic) model class

    :param gamma: (float) Discount factor
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param ent_coef: (float) Entropy coefficient for the loss caculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param learning_rate: (float) The learning rate
    :param alpha: (float)  RMSProp decay parameter (default: 0.99)
    :param epsilon: (float) RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update)
        (default: 1e-5)
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                              'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
                              (used only for loading)
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, input_length: int, output_length: int, n_steps: int, window_size: int, seed=3, gamma=0.8, vf_coef=0.25, ent_coef=0,
                 max_grad_norm=0.5, learning_rate=1e-3, alpha=0.99, epsilon=1e-5, lr_schedule='linear', verbose=0, _init_setup_model=True):

        self.policy = MetaLstmActorCriticPolicy
        self.verbose = verbose
        self.input_length = input_length
        self.output_length = output_length
        self.num_train_steps = 0
        self.n_steps = n_steps
        self.window_size = window_size
        self.total_timesteps = config.max_timesteps/10_000

        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.epsilon = epsilon
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate

        self.graph = None
        self.sess = None
        self.learning_rate_ph = None
        self.actions_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.pg_loss = None
        self.vf_loss = None
        self.entropy = None
        self.apply_backprop = None
        self.policy_model = None
        self.step = None
        self.value = None
        self.learning_rate_schedule = None
        self.trainable_variables = None

        self.layers = config.meta_layers
        self.lstm_units = config.meta_lstm_units

        if seed is not None:
            set_global_seeds(seed)

        self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=self.total_timesteps,
                                                schedule=self.lr_schedule, init_step=self.num_train_steps)

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        """
        Create all the functions and tensorflow graphs necessary to train the model
        """

        assert issubclass(self.policy, MetaLstmActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
                                                                         "instance of MetaLstmActorCriticPolicy."

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf_utils.make_session(graph=self.graph)

            # azért nincs step model mert ugyanaz a lépés (n_batch) így felesleges.
            policy_model = self.policy(sess=self.sess, input_length=self.input_length, output_length=self.output_length, n_steps=self.n_steps,
                                       window_size=self.window_size, layers=self.layers, lstm_units=self.lstm_units)

            with tf.variable_scope("loss", reuse=False):
                self.actions_ph = policy_model.pdtype.sample_placeholder([self.n_steps], name="action_ph")
                self.advs_ph = tf.placeholder(tf.float32, [self.n_steps], name="advs_ph")
                self.rewards_ph = tf.placeholder(tf.float32, [self.n_steps], name="rewards_ph")
                self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                neglogpac = policy_model.proba_distribution.neglogp(self.actions_ph)
                self.entropy = tf.reduce_mean(policy_model.proba_distribution.entropy())
                self.pg_loss = tf.reduce_mean(self.advs_ph * neglogpac)
                self.vf_loss = mse(tf.squeeze(policy_model.value_fn), self.rewards_ph)
                loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                self.trainable_variables = tf_utils.find_trainable_variables("model")
                grads = tf.gradients(loss, self.trainable_variables)
                if self.max_grad_norm is not None:
                    grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads = list(zip(grads, self.trainable_variables))

            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha,
                                                epsilon=self.epsilon)
            self.apply_backprop = trainer.apply_gradients(grads)
            self.step = policy_model.step
            self.policy_model = policy_model
            self.value = self.policy_model.value
            tf.global_variables_initializer().run(session=self.sess)

    def train_step(self, inputs: np.ndarray, discounted_rewards, actions, values):
        """
        applies a training step to the model
        """
        advs = discounted_rewards - values

        cur_lr = None
        for _ in range(self.n_steps):
            cur_lr = self.learning_rate_schedule.value()

        td_map = {self.policy_model.input_ph: inputs, self.actions_ph: actions, self.advs_ph: advs,
                  self.rewards_ph: discounted_rewards, self.learning_rate_ph: cur_lr}

        policy_loss, value_loss, policy_entropy, _ = self.sess.run(
            [self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop], td_map)

        return policy_loss, value_loss, policy_entropy

    def save(self, save_path: str, id: str):
        """
        Save the current parameters to file

        :param save_path: (str or file-like object) the save location
        """

        params = {
            "gamma": self.gamma,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "num_train_steps": self.num_train_steps,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "n_steps": self.n_steps,
            "window_size": self.window_size,
            "total_timesteps": self.total_timesteps,
            "layers": self.layers,
            "lstm_units": self.lstm_units,
        }

        json_params = {
            "input_length": self.input_length,
            "output_length": self.output_length,
            "n_steps": self.n_steps,
            "window_size": self.window_size,
            "total_timesteps": self.total_timesteps,
            "gamma": self.gamma,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "layers": self.layers,
            "lstm_units": self.lstm_units,
        }

        weights = self.sess.run(self.trainable_variables)

        utils._save_model_to_file(save_path, id, 'meta', json_params=json_params, weights=weights, params=params)

    @classmethod
    def load(cls, model_id: str,  input_len: int, output_len: int):
        """
        Load the model from file

        """
        load_path = os.path.join(config.model_path, model_id)
        weights, params = utils._load_model_from_file(load_path, 'meta')

        if params['input_length'] != input_len or params['output_length'] != output_len:
            raise ValueError("The input and the output length must be the same as the model's that trying to load.")

        model = cls(input_length=params["input_length"], output_length=params["output_length"],
                    n_steps=params["n_steps"], window_size=params["window_size"], _init_setup_model=False)
        model.__dict__.update(params)
        model.setup_model()

        restores = []
        for param, loaded_weight in zip(model.trainable_variables, weights):
            restores.append(param.assign(loaded_weight))
        model.sess.run(restores)

        return model
