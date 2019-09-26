import numpy as np
import tensorflow as tf
import os


def one_hot(k, n):
    """
    :param n: Index of the 1.
    :param k: Length of the one hot
    :return: A one-hot np array.
    """
    v = np.zeros(k)
    v[n] = 1
    return v


def total_episode_reward_logger(game, rew_acc, rewards, masks, writer, steps):
    """
    calculates the cumulated episode reward, and prints to tensorflow log the output

    :param rew_acc: (np.array float) the total running reward
    :param rewards: (np.array float) the rewards (n_envs x n_steps) (2x5)
    :param masks: (np.array bool) the end of episodes
    :param writer: (TensorFlow Session.writer) the writer to log to
    :param steps: (int) the current timestep
    :return: (np.array float) the updated total running reward
    :return: (np.array float) the updated total running reward
    """
    with tf.variable_scope("environment_info", reuse=True):
        for env_idx in range(rewards.shape[0]):
            dones_idx = np.sort(np.argwhere(masks[env_idx]))

            if len(dones_idx) == 0:
                rew_acc[env_idx] += sum(rewards[env_idx])
            else:
                rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
                summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward/" + game, simple_value=rew_acc[env_idx])])
                writer.add_summary(summary, steps + dones_idx[0, 0])
                for k in range(1, len(dones_idx[:, 0])):
                    rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k-1, 0]:dones_idx[k, 0]])
                    summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward/" + game, simple_value=rew_acc[env_idx])])
                    writer.add_summary(summary, steps + dones_idx[k, 0])
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])

    return rew_acc


def tensorboard_logger(game, rews, advs, writer, step, obs=None):
    summary = tf.Summary()
    summary.value.add(tag="discounted_reward/" + game, simple_value=None)
    summary.value.add(tag="advantage/" + game, simple_value=None)
    for i in range(len(rews)):
        summary.value[0].simple_value = rews[i]
        summary.value[1].simple_value = advs[i]
        writer.add_summary(summary, step+i)


def dir_check():
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./data/models'):
        os.mkdir('./data/models')
    if not os.path.exists('./data/logs'):
        os.mkdir('./data/logs')
    return

if __name__ == '__main__':
    dir_check()
