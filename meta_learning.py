from stable_baselines.common.policies import MlpLstmPolicy
from MetaPolicy import MetaLstmPolicyActorCriticPolicy
from stable_baselines import A2C
import gym
from stable_baselines.common.vec_env import DummyVecEnv

import tensorflow as tf


if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    #
    # model = A2C(MlpMetaLstmPolicy, env, verbose=1)
    # model.learn(total_timesteps=10000)
    #
    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
    #
    # env.close()

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    n_env = 1
    n_steps = 5
    input_length = env.action_space.n*3
    output_length = env.action_space.n
    policy = MetaLstmPolicyActorCriticPolicy(sess, input_length, output_length, n_batch=n_env*n_steps, layers=(128,128,"lstm"), lstm_units=64)

