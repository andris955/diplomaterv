import tensorflow as tf
import numpy as np
import cloudpickle
import os

from stable_baselines.a2c.utils import mse, find_trainable_variables
from stable_baselines.common import set_global_seeds

import tf_util
from MetaPolicy import MetaLstmPolicyActorCriticPolicy
from utils import Scheduler


class MetaA2CModel:
    """
    The Meta A2C (Advantage Actor Critic) model class

    :param policy: (MetaActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnMetaLstmPolicy, ...)
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

    def __init__(self, total_timesteps, input_length, output_length, n_batch, seed=None, gamma=0.99, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
                 learning_rate=7e-4, alpha=0.99, epsilon=1e-5, lr_schedule='linear', verbose=0,
                 _init_setup_model=True):

        self.policy = MetaLstmPolicyActorCriticPolicy
        self.verbose = verbose
        self.input_length = input_length
        self.output_length = output_length
        self.num_timesteps = 0
        self.n_batch = n_batch

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
        self.initial_state = None
        self.learning_rate_ph = None
        self.actions_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.pg_loss = None
        self.vf_loss = None
        self.entropy = None
        self.apply_backprop = None
        self.train_model = None
        self.step_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.learning_rate_schedule = None
        self.summary = None
        self.episode_reward = None

        if seed is not None:
            set_global_seeds(seed)

        self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps, schedule=self.lr_schedule)

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        """
        Create all the functions and tensorflow graphs necessary to train the model
        """


        assert issubclass(self.policy, MetaLstmPolicyActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
                                                           "instance of MetaLstmPolicyActorCriticPolicy."

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf_util.make_session(graph=self.graph)

            # azért nincs step model mert ugyanaz a lépés (n_batch) így felesleges.
            policy_model = self.policy(sess=self.sess, input_length=self.input_length, output_length=self.output_length, n_batch=self.n_batch)

            with tf.variable_scope("loss", reuse=False):
                self.actions_ph = policy_model.pdtype.sample_placeholder([None], name="action_ph")
                self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                neglogpac = policy_model.proba_distribution.neglogp(self.actions_ph)
                self.entropy = tf.reduce_mean(policy_model.proba_distribution.entropy())
                self.pg_loss = tf.reduce_mean(self.advs_ph * neglogpac)
                self.vf_loss = mse(tf.squeeze(policy_model.value_fn), self.rewards_ph)
                loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                tf.summary.scalar('entropy_loss', self.entropy)
                tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                tf.summary.scalar('value_function_loss', self.vf_loss)
                tf.summary.scalar('loss', loss)

                self.params = find_trainable_variables("model")
                grads = tf.gradients(loss, self.params)
                if self.max_grad_norm is not None:
                    grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads = list(zip(grads, self.params))

            with tf.variable_scope("input_info", reuse=False):
                tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate))
                tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))

            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha,
                                                epsilon=self.epsilon)
            self.apply_backprop = trainer.apply_gradients(grads)

            self.policy_model = policy_model
            self.value = self.policy_model.value
            self.initial_state = self.policy_model.initial_state
            tf.global_variables_initializer().run(session=self.sess)


    def train_step(self, state, states, rewards, masks, actions, values):
        """
        applies a training step to the model

        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for recurrent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for recurrent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :return: (float, float, float) policy loss, value loss, policy entropy
        """
        advs = rewards - values
        cur_lr = self.learning_rate_schedule.value()

        td_map = {self.train_model.obs_ph: state, self.actions_ph: actions, self.advs_ph: advs,
                  self.rewards_ph: rewards, self.learning_rate_ph: cur_lr}

        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks

        policy_loss, value_loss, policy_entropy, _ = self.sess.run(
            [self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop], td_map)

        return policy_loss, value_loss, policy_entropy


    def predict(self, observation, state=None, deterministic=False):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """

        if state is None:
            state = self.initial_state

        observation = np.array(observation)

        actions, _, states, _ = self.step(observation, state, deterministic=deterministic)

        return actions, states

    def save(self, save_path):
        """
        Save the current parameters to file

        :param save_path: (str or file-like object) the save location
        """

        data = {
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
            "action_space": self.action_space,
            "n_envs": self.n_envs,
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

    @classmethod
    def load(cls, load_path):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param kwargs: extra arguments to change the model when loading
        """

        data, params = cls._load_from_file(load_path)

        model = cls(policy=data["policy"], _init_setup_model=False)
        model.__dict__.update(data)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model

    @staticmethod
    def _save_to_file(save_path, data=None, params=None):
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".pkl"

            with open(save_path, "wb") as file_:
                cloudpickle.dump((data, params), file_)
        else:
            # Here save_path is a file-like object, not a path
            cloudpickle.dump((data, params), save_path)

    @staticmethod
    def _load_from_file(load_path):
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".pkl"):
                    load_path += ".pkl"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

            with open(load_path, "rb") as file:
                data, params = cloudpickle.load(file)
        else:
            # Here load_path is a file-like object, not a path
            data, params = cloudpickle.load(load_path)

        return data, params

    @staticmethod
    def _softmax(x_input):
        """
        An implementation of softmax.

        :param x_input: (numpy float) input vector
        :return: (numpy float) output vector
        """
        x_exp = np.exp(x_input.T - np.max(x_input.T, axis=0))
        return (x_exp / x_exp.sum(axis=0)).T
