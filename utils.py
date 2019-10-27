import numpy as np
import os
import json
import gym
from MultiTaskPolicy import MultiTaskFeedForwardA2CPolicy, MultiTaskLSTMA2CPolicy


def get_policy_from_string(policy):
    if policy == "lstm":
        return MultiTaskLSTMA2CPolicy
    elif policy == "ff":
        return MultiTaskFeedForwardA2CPolicy
    else:
        return MultiTaskFeedForwardA2CPolicy


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

def one_hot(k, n):
    """
    :param k: Index of the 1.
    :param n: Length of the one hot
    :return: A one-hot np array.
    """
    v = np.zeros(n)
    v[k] = 1
    return v


def read_params(transfer_id):
    with open(os.path.join(os.path.join("./data/models/", transfer_id), "params.json"), "r") as file:
        params = json.load(file)
    return params


def constant(_):
    """
    Returns a constant value for the Scheduler

    :param _: ignored
    :return: (float) 1
    """
    return 1.


def linear_schedule(progress):
    """
    Returns a linear value for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) 1 - progress
    """
    return 1 - progress


def middle_drop(progress):
    """
    Returns a linear value with a drop near the middle to a constant value for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) 1 - progress if (1 - progress) >= 0.75 else 0.075
    """
    eps = 0.75
    if 1 - progress < eps:
        return eps * 0.1
    return 1 - progress


def double_linear_con(progress):
    """
    Returns a linear value (x2) with a flattened tail for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) 1 - progress*2 if (1 - progress*2) >= 0.125 else 0.125
    """
    progress *= 2
    eps = 0.125
    if 1 - progress < eps:
        return eps
    return 1 - progress


def double_middle_drop(progress):
    """
    Returns a linear value with two drops near the middle to a constant value for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) if 0.75 <= 1 - p: 1 - p, if 0.25 <= 1 - p < 0.75: 0.75, if 1 - p < 0.25: 0.125
    """
    eps1 = 0.75
    eps2 = 0.25
    if 1 - progress < eps1:
        if 1 - progress < eps2:
            return eps2 * 0.5
        return eps1 * 0.1
    return 1 - progress


SCHEDULES = {
    'linear': linear_schedule,
    'constant': constant,
    'double_linear_con': double_linear_con,
    'middle_drop': middle_drop,
    'double_middle_drop': double_middle_drop
}


class Scheduler(object):
    def __init__(self, initial_value, n_values, schedule, init_step=None):
        """
        Update a value every iteration, with a specific curve

        :param initial_value: (float) initial value
        :param n_values: (int) the total number of iterations
        :param schedule: (function) the curve you wish to follow for your value
        """
        self.step = 0 if init_step is None else init_step
        self.initial_value = initial_value
        self.nvalues = n_values
        self.schedule = SCHEDULES[schedule]

    def value(self):
        """
        Update the Scheduler, and return the current value

        :return: (float) the current value
        """
        schedule = self.schedule(self.step / self.nvalues)
        if schedule < 0.05:
            schedule = 0.05
        current_value = self.initial_value * schedule
        self.step += 1.
        return current_value

    def value_steps(self, steps):
        """
        Get a value for a given step

        :param steps: (int) The current number of iterations
        :return: (float) the value for the current number of iterations
        """
        return self.initial_value * self.schedule(steps / self.nvalues)


def dir_check():
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./data/models'):
        os.mkdir('./data/models')
    if not os.path.exists('./data/logs'):
        os.mkdir('./data/logs')
    if not os.path.exists('./data/tb_logs'):
        os.mkdir('./data/tb_logs')
    return

if __name__ == '__main__':
    dir_check()
