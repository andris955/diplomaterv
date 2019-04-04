from myA2C import myA2C
from myA2C import myA2CRunner
import local_config
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
import os
import datetime

class Agent:
    def __init__(self, listOfGames):
        self.listOfGames = listOfGames
        self.env = {}
        self.__setup_environments()
        self.__setup_model()
        self.__setup_runners()

    def __setup_environments(self):
        n_cpu = local_config.number_of_cpus
        for game in self.listOfGames:
            env = SubprocVecEnv([lambda: gym.make(game) for i in range(n_cpu)])
            self.env[game] = env

    def __setup_model(self):
        self.model = myA2C(MlpPolicy, self.env, verbose=1)
        self.writer = self.model._setup_multitask_learn(10000)


    def __setup_runners(self):
        environment_list = list(self.env)
        self.runners = {}
        for environment in environment_list:
            self.runners[environment] = myA2CRunner(self.env[environment], self.model, n_steps=4, gamma=0.99)


    def play(self, model_path, game, number_of_steps, show_render=False):
        model = myA2C.load(model_path)
        env = self.env[game]
        obs = env.reset()
        for step in range(number_of_steps):
            action, states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            print(rewards)
            if show_render is True:
                env.render()


    def save_model(self):
        now = str(datetime.datetime.now())[:16]
        now = now.replace(' ', '_')
        now = now.replace(':', '_')
        now = now.replace('-', '_')
        try:
            os.mkdir("./data/models/" + now)
            self.model.save("./data/models" + now)
        except:
            print("Error at saving the model")

    def train_for_one_episode(self, game):
        #TODO
        runner = self.runners[game]
        self.model.multi_task_learn_for_one_episode(runner, self.writer)
