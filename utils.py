import numpy as np
import os
import json
import cloudpickle
import config


class CustomMessengerClass:
    def __init__(self, *args, **kwargs):
        self.__dict__ = kwargs
        for i, arg in enumerate(args):
            if not isinstance(arg, str):
                raise TypeError("Error: {}. arg ({}) is not a string. All args must be a string".format(i, arg))
            if arg not in self._fields:
                update_dict = {arg: None}
                self.__dict__.update(update_dict)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError("Index must be an integer")
        if key >= len(self._fields):
            raise IndexError("Index is out of bound")
        return self.__dict__[self._fields[key]]

    def __delitem__(self, key):
        if not isinstance(key, int):
            raise TypeError("Index must be an integer")
        if key >= len(self._fields):
            raise IndexError("Index is out of bound")
        del self.__dict__[self._fields[key]]
        self._fields.pop(key)

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise TypeError("Index must be an integer")
        if key >= len(self._fields):
            raise IndexError("Index is out of bound")
        self.__dict__[self._fields[key]] = value

    @property
    def _fields(self):
        return list(self.__dict__.keys())


def one_hot(k, n):
    """
    :param k: Index of the 1.
    :param n: Length of the one hot
    :return: A one-hot np array.
    """
    v = np.zeros(n)
    v[k] = 1
    return v


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


SCHEDULES = {
    'linear': linear_schedule,
    'constant': constant,
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
        if schedule < 0.01:
            schedule = 0.01
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


def _save_model_to_file(save_path, id, model, json_params=None, weights=None, params=None):
    if model != "multitask" and model != "meta":
        raise ValueError("model must be either str(multitask) or str(meta)")
    if isinstance(save_path, str):
        _, ext = os.path.splitext(save_path)
        if ext == "":
            if model == "multitask":
                model_path = os.path.join(save_path, model + '_model-' + id + '.pkl')
            else:
                model_path = os.path.join(save_path, model + '_model.pkl')
            param_path = os.path.join(save_path, model + '_params' + '.json')

            with open(model_path, "wb") as file_:
                cloudpickle.dump((weights, params), file_)

            with open(param_path, "w") as file_:
                json.dump(json_params, file_)
        else:
            raise ValueError("Error save_path must be a directory path")

    else:
        raise ValueError("Error: save_path must be a string")


def _load_model_from_file(load_path, model):
    if model != "multitask" and model != "meta":
        raise ValueError("model must be either str(multitask) or str(meta)")
    if isinstance(load_path, str):
        model_path = os.path.join(load_path, model + '_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, "rb") as file:
                weights, params = cloudpickle.load(file)
        else:
            raise ValueError("Error: No such file {}".format(model_path))
    else:
        raise ValueError("Error: load_path must be a string")

    return weights, params


def read_params(model_id, model):
    if model != "multitask" or model != "meta":
        raise ValueError("model must be either str(multitask) or str(meta)")
    with open(os.path.join(os.path.join(config.model_path, model_id), model + '_params.json'), "r") as file:
        params = json.load(file)
    return params


def softmax(x_input):
    """
    An implementation of softmax.

    :param x_input: (numpy float) input vector
    :return: (numpy float) output vector
    """
    x_exp = np.exp(x_input)
    return x_exp / x_exp.sum(axis=0)


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
