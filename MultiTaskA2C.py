import time
import gym
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
import os
import warnings
import cloudpickle

from MultiTaskPolicy import MultiTaskActorCriticPolicy

from stable_baselines.common.policies import LstmPolicy, get_policy_from_name
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.common import set_global_seeds

from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, SetVerbosity
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.a2c.utils import discount_with_dones, Scheduler, mse
from TensorboardWriter import TensorboardWriter

import utils


class BaseMultitaskRLModel(ABC):
    """
    The base RL model

    :param policy: (BasePolicy) Policy object
    :param env_dict: (Dictionary of Gym environments) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param policy_base: (BasePolicy) the base policy used by this method
    """

    def __init__(self, policy, env_dict, verbose=0, *, requires_vec_env, policy_base, policy_kwargs=None):
        if isinstance(policy, str):
            self.policy = get_policy_from_name(policy_base, policy)
        else:
            self.policy = policy
        self.env_dict = env_dict
        self.verbose = verbose
        self._requires_vec_env = requires_vec_env
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None
        self.action_space_dict = {}
        self.n_envs = None
        self._vectorize_action = False
        self.num_timesteps = 0

        if env_dict is not None:
            if not isinstance(env_dict, dict):
                print("env_dict must be a dictionary with keys as the name of the game and values are SubprocVecEnv objects")
            for key, value in env_dict.items():
                self.observation_space = env_dict[key].observation_space
                self.action_space_dict[key] = env_dict[key].action_space
            if requires_vec_env:
                for key, value in self.env_dict.items():
                    if isinstance(self.env_dict[key], VecEnv):
                        self.n_envs = self.env_dict[key].num_envs
                        break
                    else:
                        raise ValueError("Error: the model requires a vectorized environment, please use a VecEnv wrapper.")
            else:
                for key, env in self.env_dict.items():
                    if isinstance(env, VecEnv):
                        if env.num_envs == 1:
                            self.env = _UnvecWrapper(env)
                            self._vectorize_action = True
                        else:
                            raise ValueError("Error: the model requires a non vectorized environment or a single vectorized"
                                             " environment.")
                    self.n_envs = 1

    def get_envs(self):
        """
        returns the current environment (can be None if not defined)

        :return: (Dict) The dictionary with current environments
        """
        return self.env_dict

    def set_env(self, env_dict):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.

        :param env_dict: (Dictionary of Gym Environments) The environment for learning a policy
        """
        if env_dict is None and self.env_dict is None:
            if self.verbose >= 1:
                print("Loading a model without an environment, "
                      "this model cannot be trained until it has a valid environment.")
            return
        elif env_dict is None:
            raise ValueError("Error: trying to replace the current environment with None")

        # sanity checking the environment
        for key, env in self.env_dict.items():
            assert self.observation_space == env.observation_space, \
                "Error: the environments passed must have at least the same observation space as the model was trained on."
            assert self.action_space_dict[key] == env.action_space, \
                "Error: the environment passed must have at least the same action space as the model was trained on."
            if self._requires_vec_env:
                assert isinstance(env, VecEnv), \
                    "Error: the environment passed is not a vectorized environment, however {} requires it".format(
                        self.__class__.__name__)
                assert not issubclass(self.policy, LstmPolicy) or self.n_envs == env.num_envs, \
                    "Error: the environment passed must have the same number of environments as the model was trained on." \
                    "This is due to the Lstm policy not being capable of changing the number of environments."
                self.n_envs = env.num_envs
            else:
                # for models that dont want vectorized environment, check if they make sense and adapt them.
                # Otherwise tell the user about this issue
                if isinstance(env, VecEnv):
                    if env.num_envs == 1:
                        env = _UnvecWrapper(env)
                        self._vectorize_action = True
                    else:
                        raise ValueError("Error: the model requires a non vectorized environment or a single vectorized "
                                         "environment.")
                else:
                    self._vectorize_action = False

                self.n_envs = 1

        self.env_dict = env_dict

    def _init_num_timesteps(self, reset_num_timesteps=True):
        """
        Initialize and resets num_timesteps (total timesteps since beginning of training)
        if needed. Mainly used logging and plotting (tensorboard).

        :param reset_num_timesteps: (bool) Set it to false when continuing training
            to not create new plotting curves in tensorboard.
        :return: (bool) Whether a new tensorboard log needs to be created
        """
        if reset_num_timesteps:
            self.num_timesteps = 0

        new_tb_log = self.num_timesteps == 0
        return new_tb_log

    @abstractmethod
    def setup_model(self):
        """
        Create all the functions and tensorflow graphs necessary to train the model
        """
        pass

    def _setup_learn(self, seed):
        """
        check the environment, set the seed, and set the logger

        :param seed: (int) the seed value
        """
        if self.env_dict is None:
            raise ValueError("Error: cannot train the model without a valid environment set, please set an environment set with"
                             "set_env(self, env) method.")
        if seed is not None:
            set_global_seeds(seed)

    @abstractmethod
    def predict(self, game, observation, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        pass

    @abstractmethod
    def action_probability(self, game, observation, state=None, mask=None, actions=None):
        """
        If ``actions`` is ``None``, then get the model's action probability distribution from a given observation

        depending on the action space the output is:
            - Discrete: probability for each possible action
            - Box: mean and standard deviation of the action output

        However if ``actions`` is not ``None``, this function will return the probability that the given actions are
        taken with the given parameters (observation, state, ...) on this model.

        .. warning::
            When working with continuous probability distribution (e.g. Gaussian distribution for continuous action)
            the probability of taking a particular action is exactly zero.
            See http://blog.christianperone.com/2019/01/ for a good explanation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param actions: (np.ndarray) (OPTIONAL) For calculating the likelihood that the given actions are chosen by
            the model for each of the given parameters. Must have the same number of actions and observations.
            (set to None to return the complete action probability distribution)
        :return: (np.ndarray) the model's action probability
        """
        pass

    @abstractmethod
    def save(self, save_path):
        """
        Save the current parameters to file

        :param save_path: (str or file-like object) the save location
        """
        # self._save_to_file(save_path, data={}, params=None)
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, load_path, env=None, **kwargs):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param kwargs: extra arguments to change the model when loading
        """
        # data, param = cls._load_from_file(load_path)
        raise NotImplementedError()

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

    @staticmethod
    def _is_vectorized_observation(observation, observation_space):
        """
        For every observation type, detects and validates the shape,
        then returns whether or not the observation is vectorized.

        :param observation: (np.ndarray) the input observation to validate
        :param observation_space: (gym.spaces) the observation space
        :return: (bool) whether the given observation is vectorized or not
        """
        if isinstance(observation_space, gym.spaces.Box):
            if observation.shape == observation_space.shape:
                return False
            elif observation.shape[1:] == observation_space.shape:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for ".format(observation.shape) +
                                 "Box environment, please use {} ".format(observation_space.shape) +
                                 "or (n_env, {}) for the observation shape."
                                 .format(", ".join(map(str, observation_space.shape))))
        elif isinstance(observation_space, gym.spaces.Discrete):
            if observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
                return False
            elif len(observation.shape) == 1:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for ".format(observation.shape) +
                                 "Discrete environment, please use (1,) or (n_env, 1) for the observation shape.")
        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            if observation.shape == (len(observation_space.nvec),):
                return False
            elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for MultiDiscrete ".format(observation.shape) +
                                 "environment, please use ({},) or ".format(len(observation_space.nvec)) +
                                 "(n_env, {}) for the observation shape.".format(len(observation_space.nvec)))
        elif isinstance(observation_space, gym.spaces.MultiBinary):
            if observation.shape == (observation_space.n,):
                return False
            elif len(observation.shape) == 2 and observation.shape[1] == observation_space.n:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for MultiBinary ".format(observation.shape) +
                                 "environment, please use ({},) or ".format(observation_space.n) +
                                 "(n_env, {}) for the observation shape.".format(observation_space.n))
        else:
            raise ValueError("Error: Cannot determine if the observation is vectorized with the space type {}."
                             .format(observation_space))


#--------------------------------------------------------------------------------------------------------------------------------------------------

class ActorCriticMultitaskRLModel(BaseMultitaskRLModel):
    """
    The base class for Actor critic model

    :param policy: (BasePolicy) Policy object
    :param env_dict: (Dictionary of Gym environments) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param policy_base: (BasePolicy) the base policy used by this method (default=ActorCriticPolicy)
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    """

    def __init__(self, policy, env_dict, _init_setup_model, verbose=0, policy_base=MultiTaskActorCriticPolicy,
                 requires_vec_env=False, policy_kwargs=None):
        super(ActorCriticMultitaskRLModel, self).__init__(policy, env_dict, verbose=verbose, requires_vec_env=requires_vec_env,
                                                          policy_base=policy_base, policy_kwargs=policy_kwargs)

        self.sess = None
        self.initial_state = None
        self.step = None
        self.proba_step = None
        self.params = None

    @abstractmethod
    def setup_model(self):
        pass

    def predict(self, game, observation, state=None, mask_dict=None, deterministic=False):
        if state is None:
            state = self.initial_state
        if mask_dict is None:
            mask_dict = {}
            for key in self.env_dict.keys():
                mask_dict[key] = [False for _ in range(self.n_envs)]

        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        assert isinstance(game, str), "Error: the game passed is not a string"

        try:
            self.env_dict[game]
        except:
            print("The game must be in the env_dict dictionary!")
            exit()

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions, _, states, _ = self.step(game, observation, state, mask_dict, deterministic=deterministic)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space_dict[game], gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space_dict[game].low, self.action_space_dict[game].high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        return clipped_actions, states

    def action_probability(self, game, observation, state=None, mask_dict=None, actions=None):
        if state is None:
            state = self.initial_state
        if mask_dict is None:
            mask_dict = {}
            for key in self.env_dict.key():
                mask_dict[key] = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(game, observation, state, mask_dict)

        if len(actions_proba) == 0:  # empty list means not implemented
            warnings.warn("Warning: action probability is not implemented for {} action space. Returning None."
                          .format(type(self.action_space_dict[game]).__name__))
            return None

        if actions is not None:  # comparing the action distribution, to given actions
            actions = np.array([actions])
            if isinstance(self.action_space_dict[game], gym.spaces.Discrete):
                actions = actions.reshape((-1,))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                actions_proba = actions_proba[np.arange(actions.shape[0]), actions]

            elif isinstance(self.action_space_dict[game], gym.spaces.MultiDiscrete):
                actions = actions.reshape((-1, len(self.action_space_dict[game].nvec)))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                # Discrete action probability, over multiple categories
                actions = np.swapaxes(actions, 0, 1)  # swap axis for easier categorical split
                actions_proba = np.prod([proba[np.arange(act.shape[0]), act]
                                         for proba, act in zip(actions_proba, actions)], axis=0)

            elif isinstance(self.action_space_dict[game], gym.spaces.MultiBinary):
                actions = actions.reshape((-1, self.action_space_dict[game].n))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                # Bernoulli action probability, for every action
                actions_proba = np.prod(actions_proba * actions + (1 - actions_proba) * (1 - actions), axis=1)

            elif isinstance(self.action_space_dict[game], gym.spaces.Box):
                warnings.warn("The probabilty of taken a given action is exactly zero for a continuous distribution."
                              "See http://blog.christianperone.com/2019/01/ for a good explanation")
                actions_proba = np.zeros((observation.shape[0], 1), dtype=np.float32)
            else:
                warnings.warn("Warning: action_probability not implemented for {} actions space. Returning None."
                              .format(type(self.action_space_dict[game]).__name__))
                return None
            # normalize action proba shape for the different gym spaces
            actions_proba = actions_proba.reshape((-1, 1))

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba

    @abstractmethod
    def save(self, save_path):
        pass

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env_dict=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model


#---------------------------------------------------------------------------------------------------------------------------------------

class MultitaskA2C(ActorCriticMultitaskRLModel):
    """
    The A2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env_dict: (Dictionary of environments) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
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
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
                              (used only for loading)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env_dict, gamma=0.99, n_steps=4, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
                 learning_rate=7e-4, alpha=0.99, epsilon=1e-5, lr_schedule='linear', verbose=1, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False):

        super(MultitaskA2C, self).__init__(policy=policy, env_dict=env_dict, verbose=verbose, requires_vec_env=True,
                                           _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)

        self.n_steps = n_steps
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.epsilon = epsilon
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.graph = None
        # self.sess = None
        self.learning_rate_ph = None
        self.n_batch = None
        self.actions_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.pg_loss = {}
        self.vf_loss = {}
        self.entropy = {}
        self.params = None
        self.apply_backprop = None
        self.train_model = None
        self.step_model = None
        # self.step = None
        # self.proba_step = None
        self.value = None
        # self.initial_state = None
        self.learning_rate_schedule = None
        self.summary = None
        self.episode_reward = None
        self.updates = None

        # if we are loading, it is possible the environment is not known, however the obs and action space are known
        if _init_setup_model:
            self.setup_model()

    def _setup_multitask_learn(self, number_of_steps, seed=None):
        new_tb_log = self._init_num_timesteps(True)
        tbw = TensorboardWriter(self.graph, self.tensorboard_log, "A2C", new_tb_log)

        self._setup_learn(seed)

        self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=number_of_steps,
                                                schedule=self.lr_schedule)
        self.episode_reward = {}
        for key in self.env_dict.keys():
            self.episode_reward[key] = np.zeros((self.n_envs,))

        return tbw

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, MultiTaskActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
                                                               "instance of common.policies.ActorCriticPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(graph=self.graph)

                self.n_batch = self.n_envs * self.n_steps

                n_batch_step = None
                n_batch_train = None

                step_model = self.policy(self.sess, self.observation_space, self.action_space_dict, self.n_envs, 1,
                                         n_batch_step, reuse=False, **self.policy_kwargs)

                with tf.variable_scope("train_model", reuse=True, custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space_dict, self.n_envs,
                                              self.n_steps, n_batch_train, reuse=True, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    # for key in self.env_dict.keys():
                    self.actions_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="actions_ph") #TODO leellenőrizni, h minnden ugyanolyan típusú distribution-e
                    # self.actions_ph = train_model.pdtype_dict[key].sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph") # advantages
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                    neglogpac = {}
                    losses = {}
                    for key in self.env_dict.keys():
                        neglogpac[key] = train_model.proba_distribution_dict[key].neglogp(self.actions_ph) #TODO lehet nem jó játék actionjeit kapja
                        self.entropy[key] = tf.reduce_mean(train_model.proba_distribution_dict[key].entropy())
                        self.pg_loss[key] = tf.reduce_mean(self.advs_ph * neglogpac[key]) # policy gradient loss
                        self.vf_loss[key] = mse(tf.squeeze(train_model.value_fn_dict[key]), self.rewards_ph)
                        losses[key] = self.pg_loss[key] - self.entropy[key] * self.ent_coef + self.vf_loss[key] * self.vf_coef

                        tf.summary.scalar(key + '_entropy_loss', self.entropy[key])
                        tf.summary.scalar(key + '_policy_gradient_loss', self.pg_loss[key])
                        tf.summary.scalar(key + '_value_function_loss', self.vf_loss[key])
                        tf.summary.scalar(key + '_loss', losses[key])

                    # self.params = find_trainable_variables("model")
                    # grads = tf.gradients(loss, self.params)
                    # if self.max_grad_norm is not None:
                    #     grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    # grads = list(zip(grads, self.params))

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                            pass
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                optimizers = {}
                grads_and_vars = {}
                self.apply_backprop = {}
                for key in self.env_dict.keys():
                    optimizers[key] = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha, epsilon=self.epsilon)
                    # self.apply_backprop[key] = optimizers[key].minimize(losses[key])
                    grads_and_vars[key] = optimizers[key].compute_gradients(losses[key])
                    if self.max_grad_norm is not None:
                        grads = [grad for grad, var in grads_and_vars[key]]
                        vars = [var for grad, var in grads_and_vars[key]]
                        clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                        grads_and_vars[key] = list(zip(clipped_grads, vars))
                    self.apply_backprop[key] = optimizers[key].apply_gradients(grads_and_vars[key])

                self.train_model = train_model
                self.step_model = step_model
                self.step = step_model.step
                self.proba_step = step_model.proba_step
                self.value = step_model.value
                self.initial_state = step_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)

                self.summary = tf.summary.merge_all()

    def _train_step(self, game, obs, states, rewards, masks, actions, values, update, writer=None):
        """
        applies a training step to the model

        :param game: (str) Name of the game
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
        advs = rewards - values #TODO átnézni
        cur_lr = None
        for _ in range(len(obs)):
            cur_lr = self.learning_rate_schedule.value()
        assert cur_lr is not None, "Error: the observation input array cannon be empty"

        td_map = {self.train_model.obs_ph: obs, self.actions_ph: actions, self.advs_ph: advs,
                  self.rewards_ph: rewards, self.learning_rate_ph: cur_lr}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss[game], self.vf_loss[game], self.entropy[game], self.apply_backprop[game]],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * (self.n_batch + 1)))
            else:
                summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss[game], self.vf_loss[game], self.entropy[game], self.apply_backprop[game]], td_map)
            writer.add_summary(summary, update * (self.n_batch + 1))

        else:
            policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                [self.pg_loss[game], self.vf_loss[game], self.entropy[game], self.apply_backprop[game]], td_map)

        return policy_loss, value_loss, policy_entropy

    def multi_task_learn_for_one_episode(self, game, runner, writer, callback=None, seed=None, log_interval=100, tb_log_name="A2C",
                                         reset_num_timesteps=True):

        t_start = time.time()
        # true_reward is the reward without discount
        self.updates = self.num_timesteps // (self.n_batch + 1)
        obs, states, rewards, masks, actions, values, true_rewards = runner.run()
        _, value_loss, policy_entropy = self._train_step(game, obs, states, rewards, masks, actions, values, self.updates, writer) #TODO writer
        n_seconds = time.time() - t_start
        fps = int(self.n_batch / n_seconds)

        if writer is not None:
            self.episode_reward[game] = utils.total_episode_reward_logger(game, self.episode_reward[game],
                                                              true_rewards.reshape((self.n_envs, self.n_steps)),
                                                              masks.reshape((self.n_envs, self.n_steps)),
                                                              writer, self.num_timesteps)

        self.num_timesteps += self.n_batch + 1

        if callback is not None:
            # Only stop training if return value is False, not when it is None. This is for backwards
            # compatibility with callbacks that have no return statement.
            if callback(locals(), globals()) is False:
                return self

        if self.verbose >= 1 and (self.updates % log_interval == 0):
            explained_var = explained_variance(values, rewards)
            logger.record_tabular("updates", self.updates)
            logger.record_tabular("total_timesteps", self.num_timesteps)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(explained_var))
            logger.dump_tabular()

        return self.episode_reward

    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space_dict,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)


class myA2CRunner(AbstractEnvRunner):
    def __init__(self, env_name, env, model, n_steps=5, gamma=0.99):
        """
        A runner to learn the policy of an environment for an a2c model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        super(myA2CRunner, self).__init__(env=env, model=model, n_steps=n_steps)
        self.gamma = gamma
        self.env_name = env_name

    def run(self):
        """
        Run a learning step of the model

        :return: ([float], [float], [float], [bool], [float], [float])
                 observations, states, rewards, masks, actions, values
        """
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        for _ in range(self.n_steps):
            actions, values, _ = self.model.step(self.env_name, self.obs)
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = np.clip(actions, 0, self.env.action_space.n)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            obs, rewards, dones, _ = self.env.step(actions)
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)

        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(0, 1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        true_rewards = np.copy(mb_rewards)
        last_values = self.model.value(self.env_name, self.obs).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        # convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
        mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_values = mb_values.reshape(-1, *mb_values.shape[2:])
        mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])
        true_rewards = true_rewards.reshape(-1, *true_rewards.shape[2:])
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, true_rewards
