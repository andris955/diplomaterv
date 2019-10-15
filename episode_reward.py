import numpy as np
import tensorflow as tf


class EpisodeRewardCalculator:
    def __init__(self, envs, n_envs, writer):
        self.envs = envs
        self.n_envs = n_envs
        self.ep_rewards = {}
        self.writer = writer

        for env in self.envs:
            self.ep_rewards[env] = np.zeros(n_envs)

    def get_reward(self, game, true_rewards, masks, step):
        """
        calculates the cumulated episode reward, and prints to tensorflow log the output

        :param game: (str) name of the game
        :param true_rewards: (np.array float) the rewards (n_envs x n_steps) (2x5)
        :param masks: (np.array bool) the end of episodes
        :param step: (int) the current timestep
        :return: (np.array float) the updated total running reward
        :return: (np.array float) the updated total running reward
        """
        ep_reward = [None]*true_rewards.shape[0]
        for env_idx in range(true_rewards.shape[0]):
            masks_idx = np.sort(np.argwhere(masks[env_idx])) # meglépés előtti állapot
            if len(masks_idx) == 0:
                self.ep_rewards[game][env_idx] += sum(true_rewards[env_idx])
            else:
                self.ep_rewards[game][env_idx] += sum(true_rewards[env_idx, :masks_idx[0, 0]])
                ep_reward[env_idx] = self.ep_rewards[game][env_idx]
                self.tb_log_reward(game, step, masks_idx)
                for k in range(1, len(masks_idx[:])): # if two or more done (aka game over) in dones_idx
                    self.ep_rewards[game][env_idx] = sum(true_rewards[env_idx, masks_idx[k - 1, 0]:masks_idx[k, 0]]) # közbeeső episodeok logolása
                    self.tb_log_reward(game, step, masks_idx, k)
                self.ep_rewards[game][env_idx] = sum(true_rewards[env_idx, masks_idx[-1, 0]:]) # az episode reward felülíródik az új episode scorejával

        return ep_reward

    def tb_log_reward(self, game, step, dones_idx, k=0):
        if self.writer is not None:
            with tf.variable_scope("environment_info", reuse=True):
                summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward/" + game, simple_value=np.mean(self.ep_rewards[game]))])
                self.writer.add_summary(summary, step + dones_idx[k, 0])
