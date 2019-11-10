from abc import ABC

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution
from tf_utils import observation_input


def shared_network(scaled_images):
    """
    N: number of images in the batch
    H: height of the image
    W: width of the image
    C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)


    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2)))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2)))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2)))
    layer_3 = conv_to_fc(layer_3)
    layer_4 = activ(linear(layer_3, 'fc1', n_hidden=256, init_scale=np.sqrt(2))) #1024
    return activ(linear(layer_4, 'fc2', n_hidden=256, init_scale=np.sqrt(2))) #512


class BaseMultiTaskPolicy(ABC):
    """
    The base policy object

    :param sess: (TensorFlow session) The current TensorFlow session
    :param tasks: [str] Name of tasks
    :param ob_spaces: (Gym Space) The observation space of the environment
    :param ac_space_dict: (Dict of Gym Space) The action space of the environment
    :param n_envs_per_tasks: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    """

    def __init__(self, sess, tasks, ob_spaces, ac_space_dict, n_envs_per_task, n_steps, n_batch, reuse=False):
        self.n_envs_per_task = n_envs_per_task
        self.tasks = tasks
        self.n_steps = n_steps
        with tf.variable_scope("input", reuse=False):
            self.obs_ph, self.processed_obs = observation_input(ob_spaces, n_batch)
        if n_batch is None:
            self.n_batch = self.n_envs_per_task * self.n_steps
        else:
            self.n_batch = n_batch
        self.sess = sess
        self.reuse = reuse
        self.ob_spaces = ob_spaces
        self.ac_space_dict = ac_space_dict

    def step(self, task, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    def proba_step(self, task, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class MultiTaskActorCriticPolicy(BaseMultiTaskPolicy):
    """
    Policy object that implements actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space_dict: (Dict of Gym Spaces) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, tasks, ob_spaces, ac_space_dict, n_envs_per_task, n_steps, n_batch, reuse=False):
        super(MultiTaskActorCriticPolicy, self).__init__(sess, tasks, ob_spaces, ac_space_dict, n_envs_per_task, n_steps, n_batch, reuse=reuse)
        self.pdtype_dict = {}
        self.is_discrete_dict = {}
        for task in self.tasks:
            self.pdtype_dict[task] = make_proba_dist_type(self.ac_space_dict[task])
            self.is_discrete_dict[task] = isinstance(self.ac_space_dict[task], Discrete)
        self.policy_dict = {}
        self.proba_distribution_dict = {}
        self.value_fn_dict = {}
        self.q_value_dict = {}
        self.deterministic_action = None
        self.initial_state = None

    def _setup_init(self):
        """
        sets up the distibutions, actions, and values
        """
        self.action = {}
        self.deterministic_action = {}
        self.neglogp = {}
        self.policy_proba = {}
        self._value = {}
        for key in self.tasks:
            with tf.variable_scope(key + "_output", reuse=True):
                assert self.policy_dict is not {} and self.proba_distribution_dict is not {} and self.value_fn_dict is not {}
                self.action[key] = self.proba_distribution_dict[key].sample()
                self.deterministic_action[key] = self.proba_distribution_dict[key].mode()
                self.neglogp[key] = self.proba_distribution_dict[key].neglogp(self.action[key])
                if isinstance(self.proba_distribution_dict[key], CategoricalProbabilityDistribution):
                    self.policy_proba[key] = tf.nn.softmax(self.policy_dict[key])
                else:
                    self.policy_proba[key] = []  # it will return nothing, as it is not implemented
                self._value[key] = self.value_fn_dict[key][:, 0]

    def step(self, task, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    def proba_step(self, task, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError

    def value(self, task, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class MultiTaskFeedForwardA2CPolicy(MultiTaskActorCriticPolicy):
    def __init__(self, sess, tasks, ob_spaces, ac_space_dict, n_envs_per_task, n_steps, n_batch, reuse=False, feature_extractor=shared_network):
        super(MultiTaskFeedForwardA2CPolicy, self).__init__(sess, tasks, ob_spaces, ac_space_dict, n_envs_per_task, n_steps, n_batch, reuse=reuse)

        with tf.variable_scope("shared_model", reuse=reuse):
            self.pi_latent = vf_latent = feature_extractor(self.processed_obs)

        for key in self.pdtype_dict.keys():
            with tf.variable_scope(key + "_model", reuse=reuse):
                self.value_fn_dict[key] = linear(vf_latent, 'vf', 1)
                proba_distribution, policy, q_value = self.pdtype_dict[key].proba_distribution_from_latent(self.pi_latent, vf_latent, init_scale=0.01)
                self.proba_distribution_dict[key] = proba_distribution # distribution lehet vele sample neglog entropy a policy layeren
                self.policy_dict[key] = policy # egy linear layer
                self.q_value_dict[key] = q_value # linear layer

        self.initial_state = None

        self._setup_init()

    def step(self, task, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action[task], self._value[task], self.neglogp[task]],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action[task], self._value[task], self.neglogp[task]],
                                                   {self.obs_ph: obs})
        return action, value, state, neglogp

    def proba_step(self, task, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba[task], {self.obs_ph: obs})

    def value(self, task, obs, state=None, mask=None):
        return self.sess.run(self._value[task], {self.obs_ph: obs})


class MultiTaskLSTMA2CPolicy(MultiTaskActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_spaces: (Gym Space) The observation space of the environment
    :param ac_space_dict: (Gym Space) The action space of the environment
    :param n_envs_per_task: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param feature_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    """

    def __init__(self, sess, tasks, ob_spaces, ac_space_dict, n_envs_per_task, n_steps, n_batch, n_lstm=256, reuse=False,
                 feature_extractor=shared_network, layer_norm=False):
        super(MultiTaskLSTMA2CPolicy, self).__init__(sess, tasks, ob_spaces, ac_space_dict, n_envs_per_task, n_steps, n_batch, reuse)

        with tf.variable_scope("input", reuse=True):
            self.masks_ph = tf.placeholder(tf.float32, [n_batch], name="masks_ph")  # mask (done t-1)
            # n_lstm * 2 dim because of the cell and hidden states of the LSTM
            self.states_ph = tf.placeholder(tf.float32, [self.n_envs_per_task, n_lstm * 2], name="states_ph")  # states

        with tf.variable_scope("shared_model", reuse=reuse):
            extracted_features = feature_extractor(self.processed_obs)
            input_sequence = batch_to_seq(extracted_features, self.n_envs_per_task, self.n_steps) # n_steps x [n_env x feature extractore output shape]
            masks = batch_to_seq(self.masks_ph, self.n_envs_per_task, self.n_steps) # n_steps x [n_env x 1]
            rnn_output, self.state_new = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm, layer_norm=layer_norm)  # n_steps x [n_env x n_lstm]
            latent_vector = seq_to_batch(rnn_output) # (n_steps * n_envs) x n_lstm

        for key in self.pdtype_dict.keys():
            with tf.variable_scope(key + "_model", reuse=reuse):
                self.value_fn_dict[key] = linear(latent_vector, 'vf', 1)
                proba_distribution, policy, q_value = self.pdtype_dict[key].\
                    proba_distribution_from_latent(latent_vector, latent_vector, init_scale=0.01)
                self.proba_distribution_dict[key] = proba_distribution  # distribution lehet vele sample neglog entropy a policy layeren
                self.policy_dict[key] = policy  # egy linear layer
                self.q_value_dict[key] = q_value  # linear layer

        self.initial_state = np.zeros((self.n_envs_per_task, n_lstm * 2), dtype=np.float32)
        self._setup_init()

    def step(self, task, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, state, neglogp = self.sess.run([self.deterministic_action[task], self._value[task], self.state_new, self.neglogp[task]],
                                 {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})
        else:
            action, value, state, neglogp = self.sess.run([self.action[task], self._value[task], self.state_new, self.neglogp[task]],
                                 {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

        return action, value, state, neglogp

    def proba_step(self, task, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba[task], {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def value(self, task, obs, state=None, mask=None):
        return self.sess.run(self._value[task], {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})
