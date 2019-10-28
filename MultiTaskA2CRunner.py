import numpy as np
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.a2c.utils import discount_with_dones
import gym


class MultiTaskA2CRunner(AbstractEnvRunner):
    def __init__(self, env_name, env, model, n_steps=5, gamma=0.99):
        """
        A runner to learn the policy of an environment for an a2c model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        super(MultiTaskA2CRunner, self).__init__(env=env, model=model, n_steps=n_steps)
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
            actions, values, states, _ = self.model.step(self.env_name, self.obs, self.states, self.dones)
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = np.clip(actions, 0, self.env.action_space.n)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            obs, rewards, dones, _ = self.env.step(actions)

            self.states = states
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
        mb_masks = mb_dones[:, :-1] # lépés meglépése előtt mi volt az állapot
        mb_dones = mb_dones[:, 1:] # lépés meglépése után mi az állapot
        true_rewards = np.copy(mb_rewards)
        last_values = self.model.value(self.env_name, self.obs, self.states, self.dones).tolist()
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
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, true_rewards, mb_dones
