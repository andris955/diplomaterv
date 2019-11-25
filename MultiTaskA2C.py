import time
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
import os
import sys

from MultiTaskPolicy import MultiTaskActorCriticPolicy, get_policy_from_string
from MultiTaskRunner import MultiTaskA2CRunner
from EpisodeRewardCalculator import EpisodeRewardCalculator

from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.common import set_global_seeds


from stable_baselines import logger
from stable_baselines.common import explained_variance, SetVerbosity
from stable_baselines.a2c.utils import mse
from TensorboardWriter import TensorboardWriter

import env_utils
import utils
import tf_utils
import config


class BaseMultitaskRLModel(ABC):
    """
    The base RL model

    :param policy_name: (str) Name of the policy can be ff (feed forward) or lstm
    :param env_dict: (Dictionary of Gym environments) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    """

    def __init__(self, policy_name: str, env_dict):
        self.policy_name = policy_name
        self.policy = get_policy_from_string(self.policy_name)
        self.env_dict = env_dict
        self.tasks = [key for key in self.env_dict.keys()] if self.env_dict is not None else None
        self.verbose = config.verbose
        self.observation_space_dict = {}
        self.action_space_dict = {}
        self.n_envs_per_task = None
        self.num_timesteps = 0

        if env_dict is not None:
            if not isinstance(env_dict, dict):
                print("env_dict must be a dictionary with keys as the name of the game and values are SubprocVecEnv objects")
            for key in env_dict.keys():
                self.observation_space_dict[key] = env_dict[key].observation_space
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

    def set_envs_by_name(self, tasks: list, env_kwargs: dict):
        self.env_dict = {}
        for task in tasks:
            self.env_dict[task] = env_utils.make_atari_env(task, 1, config.seed, wrapper_kwargs=env_kwargs)

    def set_envs(self, env_dict, tasks: list):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.

        :param env_dict: (Dictionary of Gym Environments) The environment for learning a policy
        :param tasks: (list)
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
                "Error: the environment passed must have the same action space as the model was trained on."
            assert self.observation_space_dict[key] == env.observation_space, \
                "Error: the environment passed must have the same observation space as the model was trained on."
            assert isinstance(env, VecEnv), \
                "Error: the environment passed is not a vectorized environment, however {} requires it".format(
                    self.__class__.__name__)
            self.n_envs_per_task = env.num_envs

        self.env_dict = env_dict
        self.tasks = tasks
        for key in self.tasks:
            self.action_space_dict[key] = self.env_dict[key].action_space
            self.observation_space_dict[key] = self.env_dict[key].observation_space

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

    def _setup_learn(self, seed: int):
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
    def predict(self, task: str, observation: np.ndarray, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation

        :param task: (str) Name of task
        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        pass

    @abstractmethod
    def save(self, save_path: str, id: str, json_params: dict):
        """
        Save the current parameters to file

        :param save_path: (str or file-like object) the save location
        """
        # self._save_to_file(save_path, data={}, params=None)
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, model_id: str, envs_to_set=None, transfer=False, total_train_steps=None, num_timesteps=None):
        raise NotImplementedError()


# --------------------------------------------------------------------------------------------------------------------------------------------------


class ActorCriticMultitaskRLModel(BaseMultitaskRLModel):
    """
    The base class for Actor critic model

    :param policy: (str)
    :param env_dict: (Dictionary of Gym environments) The environment to learn from
            (if registered in Gym, can be str. Can be None for loading trained models)
    :param _init_setup_model: (bool) Necessary for transfer learning
    """

    def __init__(self, policy: str, env_dict, _init_setup_model: bool):
        super(ActorCriticMultitaskRLModel, self).__init__(policy, env_dict)

        self.sess = None
        self.step_model = None
        self.step = None
        self.trainable_variables = None

    @abstractmethod
    def setup_train_model(self, transfer=False):
        pass

    @abstractmethod
    def setup_step_model(self):
        pass

    def predict(self, task: str, observation, state=None, mask=None, deterministic=False):
        if task not in self.tasks:
            raise ValueError("Error model was not trained on the game that you are trying to predict on!")

        n_env = observation.shape[0]
        if state is None:
            if self.step_model.n_lstm is not None:
                state = np.zeros((n_env, 2*self.step_model.n_lstm))
            else:
                state = None
        if mask is None:
            mask = [False for _ in range(n_env)]

        observation = np.array(observation)
        vectorized_env = env_utils._is_vectorized_observation(observation, self.env_dict[task].observation_space)
        if not vectorized_env:
            observation = np.zeros((1,) + observation.shape)

        actions, value, state, neglogp = self.step(task, observation, state, mask, deterministic=deterministic)
        actions = np.clip(actions, 0, self.action_space_dict[task].n-1)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions[0]

        return actions, state

    @abstractmethod
    def save(self, save_path: str, name: str, json_params: dict):
        pass

    @classmethod
    def load(cls, model_id: str, envs_to_set=None, transfer=False, total_training_updates=None, total_timesteps=None):
        #TODO This function does not update trainer/optimizer variables (e.g. momentum). As such training after using this function may lead to less-than-optimal results.
        if transfer:
            if total_training_updates is None or total_timesteps is None:
                raise ValueError("If transfer learning is active total_train_steps and num_timesteps must be provided!")
            if not (total_timesteps == int(total_timesteps) and total_training_updates == int(total_training_updates)):
                    raise TypeError("total_train_steps and num_timesteps must be integers")

        load_path = os.path.join(config.model_path, model_id)
        weights, params = utils._load_model_from_file(load_path, "multitask")

        model = cls(policy=params['policy_name'], env_dict=None, _init_setup_model=False)
        model.__dict__.update(params)

        model.num_timesteps = total_timesteps
        model.total_train_steps = total_training_updates
        tasks = params["tasks"]

        params = utils.read_params(model_id, "multitask")
        env_kwargs = params['env_kwargs']

        if transfer:
            tasks_to_set = list(envs_to_set.keys())
            if tasks == tasks_to_set:
                model.set_envs(envs_to_set, tasks)
                model.setup_train_model(transfer=True)

            else:
                print("The envs passed as argument is not corresponding to the envs that the model "
                      "is trained on.\n Trained on: {} \n Passed: {}".format(tasks, tasks_to_set))
                sys.exit()
        else:
            model.setup_step_model()
            env_kwargs['episode_life'] = False
            env_kwargs['clip_rewards'] = False
            model.set_envs_by_name(tasks, env_kwargs=env_kwargs)

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
    :param ent_coef: (float) Entropy coefficient for the loss calculation
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

    def __init__(self, policy: str, env_dict, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.02, max_grad_norm=0.5,
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
        self.initial_learning_rate = learning_rate
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
        self.total_train_steps = 0
        self.max_scheduler_timesteps = None

        # if we are loading, it is possible the environment is not known, however the obs and action space are known
        if _init_setup_model:
            self.setup_train_model()

    def _setup_multitask_learn(self, model_name: str):
        self._setup_learn(config.seed)
        if self.max_scheduler_timesteps is None:
            self.max_scheduler_timesteps = config.max_timesteps
        self.learning_rate_schedule = utils.Scheduler(initial_value=self.initial_learning_rate, n_values=self.max_scheduler_timesteps,
                                                      schedule=self.lr_schedule, init_step=self.num_timesteps)

        tbw = TensorboardWriter(self.graph, self.tensorboard_log, model_name)

        if tbw is not None:
            self.episode_reward = EpisodeRewardCalculator([key for key in self.env_dict.keys()], self.n_envs_per_task, tbw.writer)
        else:
            self.episode_reward = EpisodeRewardCalculator([key for key in self.env_dict.keys()], self.n_envs_per_task, None)

        return tbw

    def setup_step_model(self):
        assert issubclass(self.policy, MultiTaskActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
                                                                    "instance of MultiTaskActorCriticPolicy."

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf_utils.make_session(graph=self.graph)

            self.step_model = self.policy(self.sess, self.tasks, self.observation_space_dict, self.action_space_dict, self.n_envs_per_task, n_steps=1,
                                          reuse=False)

            self.trainable_variables = tf_utils.find_trainable_variables("model")  # a modell betöltéséhez kell.
            self.step = self.step_model.step
            self.value = self.step_model.value

    def setup_train_model(self, transfer=False):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, MultiTaskActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
                                                                        "instance of MultiTaskActorCriticPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_utils.make_session(graph=self.graph)

                self.n_batch = self.n_envs_per_task * self.n_steps

                step_model = self.policy(self.sess, self.tasks, self.observation_space_dict, self.action_space_dict, self.n_envs_per_task, n_steps=1,
                                         reuse=False)

                with tf.variable_scope("train_model", reuse=True, custom_getter=tf_utils.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.tasks, self.observation_space_dict, self.action_space_dict, self.n_envs_per_task,
                                              self.n_steps, reuse=True)

                if self.policy_name.split('_')[0] == 'diff':
                    with tf.variable_scope("loss", reuse=False):
                        self.actions_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="actions_ph")
                        self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")  # advantages
                        self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                        self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                        neglogpac = {}
                        losses = {}
                        for task in self.tasks:
                            neglogpac[task] = train_model.proba_distribution_dict[task].neglogp(self.actions_ph)
                            self.entropy[task] = tf.reduce_mean(train_model.proba_distribution_dict[task].entropy())
                            self.pg_loss[task] = tf.reduce_mean(self.advs_ph * neglogpac[task])  # policy gradient loss
                            self.vf_loss[task] = mse(tf.squeeze(train_model.value_fn_dict[task]), self.rewards_ph)
                            losses[task] = self.pg_loss[task] - self.entropy[task] * self.ent_coef + self.vf_loss[task] * self.vf_coef

                            tf.summary.scalar(task + '_policy_gradient_loss', self.pg_loss[task])
                            tf.summary.scalar(task + '_value_function_loss', self.vf_loss[task])

                    with tf.variable_scope("input_info", reuse=False):
                        tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                    optimizers = {}
                    grads_and_vars = {}
                    self.apply_backprop = {}
                    for task in self.tasks:
                        optimizers[task] = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha, epsilon=self.epsilon)
                        grads_and_vars[task] = optimizers[task].compute_gradients(losses[task])
                        if self.max_grad_norm is not None:
                            grads = [grad for grad, var in grads_and_vars[task]]
                            vars = [var for grad, var in grads_and_vars[task]]
                            clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                            grads_and_vars[task] = list(zip(clipped_grads, vars))
                        self.apply_backprop[task] = optimizers[task].apply_gradients(grads_and_vars[task])
                else:
                    with tf.variable_scope("loss", reuse=False):
                        self.actions_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="actions_ph")
                        self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")  # advantages
                        self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                        self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                        for task in self.tasks:
                            neglogpac = train_model.proba_distribution.neglogp(self.actions_ph)
                            self.entropy[task] = tf.reduce_mean(train_model.proba_distribution.entropy())
                            self.pg_loss[task] = tf.reduce_mean(self.advs_ph * neglogpac)  # policy gradient loss
                            self.vf_loss[task] = mse(tf.squeeze(train_model.value_fn), self.rewards_ph)
                        loss = self.pg_loss[self.tasks[0]] - self.entropy[self.tasks[0]] * self.ent_coef + self.vf_loss[self.tasks[0]] * self.vf_coef

                        tf.summary.scalar('policy_gradient_loss', self.pg_loss[self.tasks[0]])
                        tf.summary.scalar('value_function_loss', self.vf_loss[self.tasks[0]])

                    with tf.variable_scope("input_info", reuse=False):
                        tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                    self.apply_backprop = {}
                    optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha, epsilon=self.epsilon)
                    grads_and_vars = optimizer.compute_gradients(loss)
                    if self.max_grad_norm is not None:
                        grads = [grad for grad, var in grads_and_vars]
                        vars = [var for grad, var in grads_and_vars]
                        clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                        grads_and_vars = list(zip(clipped_grads, vars))
                    for task in self.tasks:
                        self.apply_backprop[task] = optimizer.apply_gradients(grads_and_vars)

                self.train_model = train_model
                self.step_model = step_model
                self.step = step_model.step
                self.value = step_model.value

                self.trainable_variables = tf_utils.find_trainable_variables("model")

                tf.global_variables_initializer().run(session=self.sess)

                self.summary = tf.summary.merge_all()

                if not transfer:
                    self.sess.graph.finalize()

    def _train_step(self, task: str, obs: list, states: list, rewards, masks: list, actions: list, values, writer=None):
        """
        applies a training step to the model

        :param task: (str) Name of the game
        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for recurrent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for recurrent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :return: (float, float, float) policy loss, value loss, policy entropy
        """
        advs = rewards - values

        cur_lr = None
        for _ in range(len(obs)):
            cur_lr = self.learning_rate_schedule.value()
        assert cur_lr is not None, "Error: the observation input array cannot be empty!"

        if writer is not None and (self.num_timesteps % 1000 == 0):
            tf_utils.tensorboard_logger(task, rewards, advs, writer, self.num_timesteps)

        td_map = {self.train_model.obs_ph: obs, self.actions_ph: actions, self.advs_ph: advs,
                  self.rewards_ph: rewards, self.learning_rate_ph: cur_lr}

        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + self.total_train_steps) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss[task], self.vf_loss[task], self.entropy[task], self.apply_backprop[task]],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (self.total_train_steps * (self.n_batch + 1)))
            else:
                summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss[task], self.vf_loss[task], self.entropy[task], self.apply_backprop[task]], td_map)
            writer.add_summary(summary, self.total_train_steps * (self.n_batch + 1))

        else:
            policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                [self.pg_loss[task], self.vf_loss[task], self.entropy[task], self.apply_backprop[task]], td_map)

        return policy_loss, value_loss, policy_entropy

    def multi_task_learn_for_one_episode(self, task: str, runner: MultiTaskA2CRunner, writer: TensorboardWriter):
        """
        Trains until game over.

        :param task: (str) name of the game
        :param runner: (MultiTaskA2CRunnner)
        :param writer: (TensorboardWriter)
        :return:
        """
        print("-----------------------------------------------------------------")
        print("---------------------------{}---------------------------".format(task))

        mask = [False]*self.n_envs_per_task
        episode_scores = np.zeros(self.n_envs_per_task)
        policy_loss = value_loss = None
        episode_training_updates = 0
        episode_timesteps = 0
        all_process_ended = False
        tmp_scores = np.zeros(self.n_batch)

        while not all_process_ended:
            t_start = time.time()
            # self.updates = self.num_timesteps // self.n_batch + 1
            self.total_train_steps += 1
            # true_reward is the reward without discount
            obs, states, rewards, masks, actions, values, true_rewards, dones = runner.run()
            policy_loss, value_loss, policy_entropy = self._train_step(task, obs, states, rewards, masks, actions, values, writer)
            n_seconds = time.time() - t_start
            fps = int(self.n_batch / n_seconds)
            train_step_per_sec = int(1 / n_seconds)
            tmp_ep_scores = self.episode_reward.get_reward(task, true_rewards.reshape((self.n_envs_per_task, self.n_steps)),
                                                           masks.reshape((self.n_envs_per_task, self.n_steps)), self.total_train_steps)
            tmp_scores += true_rewards
            masks_reshaped = masks.reshape((self.n_envs_per_task, self.n_steps))
            assert masks_reshaped.shape[0] == self.n_envs_per_task, "dones.shape[0] must be n_envs_per_task"
            for i in range(masks_reshaped.shape[0]):
                if True in masks_reshaped[i, :]:
                    mask[i] = True
                    episode_scores[i] = tmp_ep_scores[i]
            all_process_ended = all(mask)

            self.num_timesteps += self.n_batch
            episode_timesteps += self.n_steps
            episode_training_updates += 1

            if self.verbose >= 1 and ((self.total_train_steps % config.stdout_logging_frequency_in_train_steps == 0) or all_process_ended):
                explained_var = explained_variance(values, rewards)
                logger.record_tabular("training_updates", self.total_train_steps)
                logger.record_tabular("total_timesteps", self.num_timesteps)
                logger.record_tabular("train_step_per_sec", train_step_per_sec)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_loss", float(policy_loss))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("explained_variance", float(explained_var))
                logger.dump_tabular()

        print("Game over: {}".format(all_process_ended))

        episode_score = np.mean(episode_scores)

        return episode_score, policy_loss, value_loss, episode_training_updates

    def save(self, save_path: str, id: str, json_params: dict):
        params = {
            "policy_name": self.policy_name,
            "tasks": self.tasks,
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "initial_learning_rate": self.initial_learning_rate,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "observation_space_dict": self.observation_space_dict,
            "action_space_dict": self.action_space_dict,
            'tensorboard_log': self.tensorboard_log,
            "full_tensorboard_log": self.full_tensorboard_log,
            "max_scheduler_timesteps": self.max_scheduler_timesteps,
        }

        json_params.update({
            "policy": self.policy_name,
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "initial_learning_rate": self.initial_learning_rate,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "observation_spaces": [ob_space.shape for ob_space in self.observation_space_dict.values()],
            "action_spaces": {},
            "n_envs_per_task": self.n_envs_per_task,
            "max_scheduler_timesteps": self.max_scheduler_timesteps,
            "n_lstm": config.n_lstm,
        })

        for game, value in self.action_space_dict.items():
            json_params["action_spaces"][game] = value.n

        weights = self.sess.run(self.trainable_variables)  # Only works with python > 3.6 because of default insertion order dictionaries.

        utils._save_model_to_file(save_path, id, 'multitask', json_params=json_params, weights=weights, params=params)
