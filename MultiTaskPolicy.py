from abc import ABC

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution
from stable_baselines.common.input import observation_input


def cnn_from_paper(scaled_images, **kwargs):
    """
    CNN from the Multi task paper.

    N: number of images in the batch
    H: height of the image
    W: width of the image
    C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)


    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    layer_4 = activ(linear(layer_3, 'fc1', n_hidden=1024, init_scale=np.sqrt(2)))
    return activ(linear(layer_4, 'fc2', n_hidden=512, init_scale=np.sqrt(2)))


class BaseMultiTaskPolicy(ABC):
    """
    The base policy object

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space_dict: (Dict of Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param add_action_ph: (bool) whether or not to create an action placeholder
    """

    def __init__(self, sess, ob_space, ac_space_dict, n_env, n_steps, n_batch, reuse=False, scale=True,
                 obs_phs=None, add_action_ph=False):
        self.n_env = n_env
        self.n_steps = n_steps
        with tf.variable_scope("input", reuse=False):
            if obs_phs is None:
                self.obs_ph, self.processed_obs = observation_input(ob_space, n_batch, scale=scale)
            else:
                self.obs_ph, self.processed_obs = obs_phs

            if add_action_ph:
                for key, value in ac_space_dict.items():
                    self.action_ph = tf.placeholder(dtype=ac_space_dict[key].dtype, shape=(None,) + ac_space_dict[key].shape, name="action_ph")
                    break
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space_dict = ac_space_dict

    @staticmethod
    def _kwargs_check(feature_extraction, kwargs):
        """
        Ensure that the user is not passing wrong keywords
        when using policy_kwargs.

        :param feature_extraction: (str)
        :param kwargs: (dict)
        """
        # When using policy_kwargs parameter on model creation,
        # all keywords arguments must be consumed by the policy constructor except
        # the ones for the cnn_extractor network (cf nature_cnn()), where the keywords arguments
        # are not passed explicitely (using **kwargs to forward the arguments)
        # that's why there should be not kwargs left when using the mlp_extractor
        # (in that case the keywords arguments are passed explicitely)
        if feature_extraction == 'mlp' and len(kwargs) > 0:
            raise ValueError("Unknown keywords for policy: {}".format(kwargs))

    def step(self, game, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    def proba_step(self, game, obs, state=None, mask=None):
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

    def __init__(self, sess, ob_space, ac_space_dict, n_env, n_steps, n_batch, reuse=False, scale=False):
        super(MultiTaskActorCriticPolicy, self).__init__(sess, ob_space, ac_space_dict, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=scale)
        self.pdtype_dict = {}
        self.is_discrete_dict = {}
        for key, value in ac_space_dict.items():
            self.pdtype_dict[key] = make_proba_dist_type(ac_space_dict[key])
            self.is_discrete_dict[key] = isinstance(ac_space_dict[key], Discrete)
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
        for key in self.ac_space_dict.keys():
            with tf.variable_scope(key + "_output", reuse=True):
                assert self.policy_dict is not {} and self.proba_distribution_dict is not {} and self.value_fn_dict is not {}
                self.action[key] = self.proba_distribution_dict[key].sample()
                self.deterministic_action[key] = self.proba_distribution_dict[key].mode()
                self.neglogp[key] = self.proba_distribution_dict[key].neglogp(self.action[key])
                if isinstance(self.proba_distribution_dict[key], CategoricalProbabilityDistribution):
                    self.policy_proba[key] = tf.nn.softmax(self.policy_dict[key])
                # elif isinstance(self.proba_distribution, DiagGaussianProbabilityDistribution):
                #     self.policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
                # elif isinstance(self.proba_distribution, BernoulliProbabilityDistribution):
                #     self.policy_proba = tf.nn.sigmoid(self.policy)
                # elif isinstance(self.proba_distribution, MultiCategoricalProbabilityDistribution):
                #     self.policy_proba = [tf.nn.softmax(categorical.flatparam())
                #                          for categorical in self.proba_distribution.categoricals]
                else:
                    self.policy_proba[key] = []  # it will return nothing, as it is not implemented
                self._value[key] = self.value_fn_dict[key][:, 0]

    def step(self, game, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    def proba_step(self, game, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError

    def value(self, game, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class MultiTaskA2CPolicy(MultiTaskActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space_dict, n_env, n_steps, n_batch, reuse=False, cnn_extractor=cnn_from_paper, **kwargs):
        super(MultiTaskA2CPolicy, self).__init__(sess, ob_space, ac_space_dict, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("shared_model", reuse=reuse):
            self.pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)

        for key in self.pdtype_dict.keys():
            with tf.variable_scope(key + "_model", reuse=reuse):
                self.value_fn_dict[key] = linear(vf_latent, key + '_vf', 1)
                proba_distribution, policy, q_value = self.pdtype_dict[key].proba_distribution_from_latent(self.pi_latent, vf_latent, init_scale=0.01)
                self.proba_distribution_dict[key] = proba_distribution # distribution lehet vele sample neglog entropy
                self.policy_dict[key] = policy # egy linear layer
                self.q_value_dict[key] = q_value # linear layer

        self.initial_state = None

        self._setup_init()

    def step(self, game, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action[game], self._value[game], self.neglogp[game]],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action[game], self._value[game], self.neglogp[game]],
                                                   {self.obs_ph: obs})
        return action, value, neglogp

    def proba_step(self, game, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba[game], {self.obs_ph: obs})

    def value(self, game, obs, state=None, mask=None):
        return self.sess.run(self._value[game], {self.obs_ph: obs})
