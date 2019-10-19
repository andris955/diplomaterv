import numpy as np
import tensorflow as tf
import os
import json
from gym.spaces import Box


def one_hot(k, n):
    """
    :param k: Index of the 1.
    :param n: Length of the one hot
    :return: A one-hot np array.
    """
    v = np.zeros(n)
    v[k] = 1
    return v


def tensorboard_logger(game, rews, advs, writer, step, obs=None):
    summary = tf.Summary()
    summary.value.add(tag="discounted_reward/" + game, simple_value=None)
    summary.value.add(tag="advantage/" + game, simple_value=None)
    for i in range(len(rews)):
        summary.value[0].simple_value = rews[i]
        summary.value[1].simple_value = advs[i]
        writer.add_summary(summary, step+i)


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
    def __init__(self, initial_value, n_values, schedule):
        """
        Update a value every iteration, with a specific curve

        :param initial_value: (float) initial value
        :param n_values: (int) the total number of iterations
        :param schedule: (function) the curve you wish to follow for your value
        """
        self.step = 0.
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

def observation_input(batch_size, ob_spaces):
    for ob_space in ob_spaces:
        if isinstance(ob_space, Box):
            continue
        else:
            raise ValueError("All observation space must be a box!")
    observation_ph = tf.placeholder(shape=(batch_size, None, None, None), dtype=ob_spaces[0].dtype, name='Ob')
    processed_observations = tf.cast(observation_ph, tf.float32)
    processed_observations = tf.image.resize(processed_observations, (210, 160))
    # rescale to [1, 0] if the bounds are defined
    if (not np.any(np.isinf(ob_spaces[0].low)) and not np.any(np.isinf(ob_spaces[0].high)) and
       np.any((ob_spaces[0].high - ob_spaces[0].low) != 0)):
        processed_observations = ((processed_observations - ob_spaces[0].low) / (ob_spaces[0].high - ob_spaces[0].low))
    return observation_ph, processed_observations



def find_trainable_variables(key):
    """
    Returns the trainable variables within a given scope
    :param key: (str) The variable scope
    :return: ([TensorFlow Tensor]) the trainable variables

    - **removed** ``a2c.utils.find_trainable_params`` please use ``common.tf_util.get_trainable_vars`` instead.
  ``find_trainable_params`` was returning all trainable variables, discarding the scope argument.
  This bug was causing the model to save duplicated parameters (for DDPG and SAC)
  but did not affect the performance.
    """
    with tf.variable_scope(key):
        return tf.trainable_variables()


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
