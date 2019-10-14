import numpy as np
import tensorflow as tf
import os
import json


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
