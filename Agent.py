from MultiTaskA2C import MultitaskA2C
from MultiTaskA2C import myA2CRunner
from MultiTaskPolicy import MultiTaskA2CPolicy
import local_config
import gym
from stable_baselines.common.vec_env import SubprocVecEnv
import os
import datetime

class Agent:
    def __init__(self, listOfGames):
        self.listOfGames = listOfGames
        self.sub_proc_environments = {}
        self.policy = MultiTaskA2CPolicy
        self.__setup_environments()
        self.__setup_model()
        self.__setup_runners()

    def __setup_environments(self):
        n_cpu = local_config.number_of_cpus
        for game in self.listOfGames:
            env = SubprocVecEnv([lambda: gym.make(game) for i in range(n_cpu)])
            self.sub_proc_environments[game] = env

    def __setup_model(self):
        self.model = MultitaskA2C(self.policy, self.sub_proc_environments, verbose=1, tensorboard_log="/home/andris955/Documents/Dipterv/diplomaterv/data/logs", full_tensorboard_log=True, n_steps=4)
        self.tbw = self.model._setup_multitask_learn(10000)
        self.writer = self.tbw.enter()

    def __setup_runners(self):
        self.runners = {}
        for environment in self.listOfGames:
            self.runners[environment] = myA2CRunner(environment, self.sub_proc_environments[environment], self.model, n_steps=4, gamma=0.99)

    def train_for_one_episode(self, game):
        runner = self.runners[game]
        score = self.model.multi_task_learn_for_one_episode(game, runner, self.writer)
        return score

    def play(self, model_path, game, max_number_of_games, show_render=False):
        number_of_games = 0
        model = MultitaskA2C.load(model_path)
        env = gym.make(game)
        obs = env.reset()
        while number_of_games < max_number_of_games:
            action, states = model.predict(game, obs)
            obs, rewards, done, info = env.step(action)
            if done:
                obs = env.reset()
                number_of_games += 1
            print(rewards)
            if show_render is True:
                env.render()

    def save_model(self):
        self.tbw.exit()
        now = str(datetime.datetime.now())[:16]
        now = now.replace(' ', '_')
        now = now.replace(':', '_')
        now = now.replace('-', '_')
        try:
            os.mkdir("./data/models/" + now)
            self.model.save("./data/models" + now)
        except:
            print("Error at saving the model")

