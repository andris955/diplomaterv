from stable_baselines.common.atari_wrappers import wrap_deepmind, NoopResetEnv
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
import gym
import config

def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None,
                   start_index=1, start_method=None):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :return: (Gym Environment) The atari environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id, frameskip=config.frameskip)
            env = NoopResetEnv(env, noop_max=30)
            env.seed(seed + rank)
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)

    # When using one environment, no need to start subprocesses
    if num_env == 1:
        return DummyVecEnv([make_env(0)])

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)],
                         start_method=start_method)


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
