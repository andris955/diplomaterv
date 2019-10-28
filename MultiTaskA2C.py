import time
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
import os
import cloudpickle

from MultiTaskPolicy import MultiTaskActorCriticPolicy, MultiTaskLSTMA2CPolicy

from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.common import set_global_seeds

from stable_baselines import logger
from stable_baselines.common import explained_variance, SetVerbosity
from stable_baselines.a2c.utils import mse
from TensorboardWriter import TensorboardWriter

import json
import utils
import tf_utils
import global_config

from episode_reward import EpisodeRewardCalculator


class BaseMultitaskRLModel(ABC):
    """
    The base RL model

    :param policy: (BasePolicy) Policy object
    :param env_dict: (Dictionary of Gym environments) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    """

    def __init__(self, policy_name, env_dict):
        self.policy_name = policy_name
        self.policy = utils.get_policy_from_string(self.policy_name)
        self.env_dict = env_dict
        self.tasks = [key for key in self.env_dict.keys()] if self.env_dict is not None else None
        self.verbose = global_config.verbose
        self.observation_spaces = []
        self.action_space_dict = {}
        self.n_envs_per_task = None
        # self._vectorize_action = False
        self.num_timesteps = 0

        if env_dict is not None:
            if not isinstance(env_dict, dict):
                print("env_dict must be a dictionary with keys as the name of the game and values are SubprocVecEnv objects")
            for key in env_dict.keys():
                self.observation_spaces.append(env_dict[key].observation_space)
                self.action_space_dict[key] = env_dict[key].action_space
            for key in self.env_dict.keys():
                if isinstance(self.env_dict[key], VecEnv):
                    if env_dict[key].num_envs == 1:
                        self.env_dict[key] = _UnvecWrapper(env_dict[key])
                        self._vectorize_action = True
                    if self.n_envs_per_task is None:
                        self.n_envs_per_task = self.env_dict[key].num_envs
                    else:
                        if self.n_envs_per_task != self.env_dict[key].num_envs:
                            raise ValueError("All tasks must have the same number of environments ")
                    break
                else:
                    raise ValueError("Error: the model requires a vectorized environment, please use a VecEnv wrapper.")

    def get_envs(self):
        """
        returns the current environment (can be None if not defined)

        :return: (Dict) The dictionary with current environments
        """
        return self.env_dict

    def set_env(self, env_dict, tasks):
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
        for key, env in env_dict.items():
            assert self.action_space_dict[key] == env.action_space, \
                "Error: the environment passed must have at least the same action space as the model was trained on."
            assert isinstance(env, VecEnv), \
                "Error: the environment passed is not a vectorized environment, however {} requires it".format(
                    self.__class__.__name__)
            assert not issubclass(self.policy, LstmPolicy) or self.n_envs_per_task == env.num_envs, \
                "Error: the environment passed must have the same number of environments as the model was trained on." \
                "This is due to the Lstm policy not being capable of changing the number of environments."
            self.n_envs_per_task = env.num_envs

        self.env_dict = env_dict
        self.tasks = tasks
        self.observation_spaces = [self.env_dict[key].observation_space for key in self.tasks]
        for key in self.tasks:
            self.action_space_dict[key] = self.env_dict[key].action_space

    @abstractmethod
    def setup_train_model(self, transfer=False):
        """
        Create all the functions and tensorflow graphs necessary to train the model
        """
        pass

    @abstractmethod
    def setup_step_model(self):
        """
        Create all the functions and tensorflow graphs necessary to play with the model
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
    def save(self, save_path, name):
        """
        Save the current parameters to file

        :param save_path: (str or file-like object) the save location
        """
        # self._save_to_file(save_path, data={}, params=None)
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, model_id, envs_to_set=None, transfer=False):
        raise NotImplementedError()

    @staticmethod
    def _save_to_file(save_path, name, json_params=None, weights=None, params=None):
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                model_path = os.path.join(save_path, 'model_' + name + '.pkl')
                param_path = os.path.join(save_path, 'params.json')

                with open(model_path, "wb") as file_:
                    cloudpickle.dump((weights, params), file_)

                # if not os.path.exists(param_path):
                with open(param_path, "w") as file_:
                    json.dump(json_params, file_)
            else:
                raise ValueError("Error save_path must be a directory path")

        else:
            raise ValueError("Error: save_path must be a string")

    @staticmethod
    def _load_from_file(load_path):
        if isinstance(load_path, str):
            model_path = os.path.join(load_path, 'model.pkl')
            if os.path.exists(model_path):
                with open(model_path, "rb") as file:
                    weights, params = cloudpickle.load(file)
            else:
                raise ValueError("Error: No such file {}".format(model_path))
        else:
            raise ValueError("Error: load_path must be a string")

        return weights, params

    @staticmethod
    def _softmax(x_input):
        """
        An implementation of softmax.

        :param x_input: (numpy float) input vector
        :return: (numpy float) output vector
        """
        x_exp = np.exp(x_input.T - np.max(x_input.T, axis=0))
        return (x_exp / x_exp.sum(axis=0)).T


# --------------------------------------------------------------------------------------------------------------------------------------------------


class ActorCriticMultitaskRLModel(BaseMultitaskRLModel):
    """
    The base class for Actor critic model

    :param policy: (str)
    :param _init_steup_model: (Bool) A loadhoz kell!
    :param env_dict: (Dictionary of Gym environments) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: (BasePolicy) the base policy used by this method (default=ActorCriticPolicy)
    """

    def __init__(self, policy, env_dict, _init_setup_model):
        super(ActorCriticMultitaskRLModel, self).__init__(policy, env_dict)

        self.sess = None
        self.initial_state = None
        self.step = None
        self.trainable_variables = None
        # self.transfer_id = []

    @abstractmethod
    def setup_train_model(self, transfer=False):
        pass

    @abstractmethod
    def setup_step_model(self):
        pass

    def predict(self, game, observation, state=None, mask=None, deterministic=False):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs_per_task)]

        observation = np.array(observation)
        vectorized_env = utils._is_vectorized_observation(observation, self.observation_spaces[game])
        assert isinstance(game, str), "Error: the game passed is not a string"

        try:
            self.env_dict[game]
        except:
            print("The game must be in the env_dict dictionary!")
            exit()

        observation = observation.reshape((-1,) + observation.shape)
        actions, value, state, neglogp = self.step(game, observation, state, mask, deterministic=deterministic)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions[0]

        return actions, state

    @abstractmethod
    def save(self, save_path, name):
        pass

    @classmethod
    def load(cls, model_id, envs_to_set=None, transfer=False):
        load_path = os.path.join('./data/models', model_id)
        weights, params = cls._load_from_file(load_path)

        model = cls(policy=params['policy'], env_dict=None, _init_setup_model=False)
        model.__dict__.update(params)
        # model.transfer_id.append(model_id)

        tasks = params["tasks"]

        if not transfer:
            model.setup_step_model()
        else:
            tasks_to_set = [key for key in envs_to_set.keys()]
            if tasks == tasks_to_set:
                model.set_env(envs_to_set, tasks)
                model.setup_train_model(transfer=True)
            else:
                print("The envs passed as argument is not corresponding to the envs that the model "
                      "is trained on.\n Trained on: {} \n Passed: {}".format(tasks, tasks_to_set))
                exit()

        restores = []
        for param, loaded_weight in zip(model.trainable_variables, weights):
            restores.append(param.assign(loaded_weight))
        model.sess.run(restores)

        model.sess.graph.finalize()

        return model, tasks


# ---------------------------------------------------------------------------------------------------------------------------------------


class MultitaskA2C(ActorCriticMultitaskRLModel):
    """
    The A2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

    :param policy: (str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
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
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
                              (used only for loading)
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env_dict, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
                 learning_rate=1e-3, alpha=0.99, epsilon=1e-5, lr_schedule='linear', tensorboard_log=None,
                 _init_setup_model=True, full_tensorboard_log=False):

        super(MultitaskA2C, self).__init__(policy=policy, env_dict=env_dict, _init_setup_model=_init_setup_model)

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
        self.sess = None
        self.learning_rate_ph = None
        self.n_batch = None
        self.actions_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.pg_loss = {}
        self.vf_loss = {}
        self.entropy = {}
        self.apply_backprop = None
        self.train_model = None
        self.step_model = None
        self.value = None
        self.learning_rate_schedule = None
        self.summary = None
        self.episode_reward = None
        self.total_train_steps = None
        self.train_step = 0

        # if we are loading, it is possible the environment is not known, however the obs and action space are known
        if _init_setup_model:
            self.setup_train_model()

    def _setup_multitask_learn(self, model_name, max_train_steps, seed=3):
        self._setup_learn(seed)
        self.total_train_steps = max_train_steps
        self.learning_rate_schedule = utils.Scheduler(initial_value=self.learning_rate, n_values=self.total_train_steps,
                                                      schedule=self.lr_schedule, init_step=self.train_step)

        tbw = TensorboardWriter(self.graph, self.tensorboard_log, model_name)

        if tbw is not None:
            self.episode_reward = EpisodeRewardCalculator([key for key in self.env_dict.keys()], self.n_envs_per_task, tbw.writer)
        else:
            self.episode_reward = EpisodeRewardCalculator([key for key in self.env_dict.keys()], self.n_envs_per_task, None)

        return tbw

    def setup_step_model(self):
        assert issubclass(self.policy, MultiTaskActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
                                                                    "instance of common.policies.ActorCriticPolicy."

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf_utils.make_session(graph=self.graph)

            step_model = self.policy(self.sess, self.observation_spaces, self.action_space_dict, self.n_envs_per_task, n_steps=1,
                                     reuse=False)

            self.trainable_variables = tf_utils.find_trainable_variables("model")  # a modell betöltéséhez kell.
            self.step = step_model.step

    def setup_train_model(self, transfer=False):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, MultiTaskActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
                                                                        "instance of MultiTaskActorCriticPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_utils.make_session(graph=self.graph)

                self.n_batch = self.n_envs_per_task * self.n_steps

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, MultiTaskLSTMA2CPolicy):
                    n_batch_step = self.n_envs_per_task
                    n_batch_train = self.n_envs_per_task * self.n_steps

                step_model = self.policy(self.sess, self.tasks, self.observation_spaces, self.action_space_dict, self.n_envs_per_task, n_steps=1,
                                         n_batch=n_batch_step, reuse=False)

                with tf.variable_scope("train_model", reuse=True, custom_getter=tf_utils.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.tasks, self.observation_spaces, self.action_space_dict, self.n_envs_per_task,
                                              self.n_steps, n_batch_train, reuse=True)

                with tf.variable_scope("loss", reuse=False):
                    self.actions_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="actions_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")  # advantages
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                    neglogpac = {}
                    losses = {}
                    for key in self.env_dict.keys():
                        neglogpac[key] = train_model.proba_distribution_dict[key].neglogp(self.actions_ph)
                        self.entropy[key] = tf.reduce_mean(train_model.proba_distribution_dict[key].entropy())
                        self.pg_loss[key] = tf.reduce_mean(self.advs_ph * neglogpac[key])  # policy gradient loss
                        self.vf_loss[key] = mse(tf.squeeze(train_model.value_fn_dict[key]), self.rewards_ph)
                        losses[key] = self.pg_loss[key] - self.entropy[key] * self.ent_coef + self.vf_loss[key] * self.vf_coef

                        # tf.summary.scalar(key + '_entropy_loss', self.entropy[key])
                        tf.summary.scalar(key + '_policy_gradient_loss', self.pg_loss[key])
                        tf.summary.scalar(key + '_value_function_loss', self.vf_loss[key])
                        # tf.summary.scalar(key + '_loss', losses[key])

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                optimizers = {}
                grads_and_vars = {}
                self.apply_backprop = {}
                for key in self.env_dict.keys():
                    optimizers[key] = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha, epsilon=self.epsilon)
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
                self.value = step_model.value
                self.initial_state = step_model.initial_state

                self.trainable_variables = tf_utils.find_trainable_variables("model")

                tf.global_variables_initializer().run(session=self.sess)

                self.summary = tf.summary.merge_all()

                if not transfer:
                    self.sess.graph.finalize()

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
        advs = rewards - values

        cur_lr = self.learning_rate_schedule.value()
        assert cur_lr is not None, "Error: the observation input array cannon be empty"

        if writer is not None and (self.num_timesteps % 1000 == 0):
            tf_utils.tensorboard_logger(game, rewards, advs, writer, self.num_timesteps, obs=None)

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

    def multi_task_learn_for_one_episode(self, game, runner, writer, log_interval=100):
        """
        Trains until game over.

        :param game: (str) name of the game
        :param runner: (myA2CRunnner)
        :param writer: (Tensorboard writer)
        :param log_interval: (int)
        :return:
        """
        print("-----------------------------------------------------------------")
        print("---------------------------{}---------------------------".format(game))

        mask = [False]*self.n_envs_per_task
        ep_scores = [-1]*self.n_envs_per_task
        policy_loss = value_loss = None
        ep_train_step = 0

        while mask != [True]*self.n_envs_per_task:
            t_start = time.time()
            # true_reward is the reward without discount
            # self.updates = self.num_timesteps // self.n_batch + 1
            self.train_step += 1
            obs, states, rewards, masks, actions, values, true_rewards, dones = runner.run()
            policy_loss, value_loss, policy_entropy = self._train_step(game, obs, states, rewards, masks, actions, values, self.train_step, writer)
            n_seconds = time.time() - t_start
            fps = int(self.n_batch / n_seconds)
            train_step_per_sec = int(1 / n_seconds)

            tmp_ep_scores = self.episode_reward.get_reward(game, true_rewards.reshape((self.n_envs_per_task, self.n_steps)),
                                                           masks.reshape((self.n_envs_per_task, self.n_steps)), self.train_step)

            masks_reshaped = masks.reshape((self.n_envs_per_task, self.n_steps))
            assert masks_reshaped.shape[0] == self.n_envs_per_task, "dones.shape[0] must be n_envs_per_task"
            for i in range(masks_reshaped.shape[0]):
                if True in masks_reshaped[i, :]:
                    mask[i] = True
                    if ep_scores[i] == -1 and tmp_ep_scores[i] is not None:
                        ep_scores[i] = tmp_ep_scores[i]

            self.num_timesteps += self.n_batch
            ep_train_step += 1

            if self.verbose >= 1 and ((self.train_step % log_interval == 0) or (mask == [True]*self.n_envs_per_task)):
                explained_var = explained_variance(values, rewards)
                logger.record_tabular("training_updates", self.train_step)
                logger.record_tabular("train_step_per_sec", train_step_per_sec)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_loss", float(policy_loss))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("explained_variance", float(explained_var))
                logger.dump_tabular()

        print("Game over: {}".format(mask))

        return ep_scores, policy_loss, value_loss, ep_train_step

    def save(self, save_path, name):
        params = {
            "tasks": self.tasks,
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
            "policy": self.policy_name,
            #'env_dict': self.env_dict, # nem lehet lementeni a TypeError: Pickling an AuthenticationString object is disallowed for security reasons miatt.
            # "observation_spaces": self.observation_spaces,
            "action_space_dict": self.action_space_dict,
            "n_envs_per_task": self.n_envs_per_task,
            # "_vectorize_action": self._vectorize_action,
            'tensorboard_log': self.tensorboard_log,
            "full_tensorboard_log": self.full_tensorboard_log,
            # "transfer_id": self.transfer_id,
            "total_train_steps": self.total_train_steps,
            "train_step": self.train_step,
        }

        json_params = {
            "policy": self.policy_name,
            "tasks": self.tasks,
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            # "verbose": self.verbose,
            "observation_spaces": [ob_space.shape for ob_space in self.observation_spaces],
            "action_spaces": {},
            "n_envs_per_task": self.n_envs_per_task,
            # "_vectorize_action": self._vectorize_action,
            # "transfer_id": self.transfer_id,
            "max_training_step": global_config.MaxTrainSteps,
            "train_step": self.train_step
        }

        for game, value in self.action_space_dict.items():
            json_params["action_spaces"][game] = value.n

        weights = self.sess.run(self.trainable_variables)

        self._save_to_file(save_path, name, json_params=json_params, weights=weights, params=params)
