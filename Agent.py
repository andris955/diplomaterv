from MultiTaskA2C import MultitaskA2C
from MultiTaskA2C import myA2CRunner
from MultiTaskPolicy import MultiTaskA2CPolicy
import gym
from stable_baselines.common.vec_env import SubprocVecEnv
import global_config
import datetime
import os


class Agent:
    def __init__(self, algorithm, listOfGames, max_steps, n_cpus, transfer_id):
        self.algorithm = algorithm
        self.listOfGames = listOfGames
        self.max_steps = max_steps
        self.n_cpus = n_cpus
        self.sub_proc_environments = {}
        self.policy = MultiTaskA2CPolicy

        now = str(datetime.datetime.now())[2:16]
        now = now.replace(' ', '_')
        now = now.replace(':', '_')
        now = now.replace('-', '_')
        self.initialize_time = now
        self.transfer_id = transfer_id

        self.__setup_environments()
        self.__setup_model()
        self.__setup_runners()

    def __setup_environments(self):
        for game in self.listOfGames:
            env = SubprocVecEnv([lambda: gym.make(game) for i in range(self.n_cpus)])
            assert isinstance(env.action_space, gym.spaces.Discrete), "Error: all the input games must have Discrete action space"
            self.sub_proc_environments[game] = env

    def __setup_model(self):
        self.model = MultitaskA2C(self.policy, self.sub_proc_environments, verbose=1, tensorboard_log="./data/logs", full_tensorboard_log=True, n_steps=global_config.n_steps)
        self.tbw = self.model._setup_multitask_learn(self.algorithm, self.max_steps, self.initialize_time, self.transfer_id)
        self.writer = self.tbw.enter()

    def __setup_runners(self):
        self.runners = {}
        for environment in self.listOfGames:
            self.runners[environment] = myA2CRunner(environment, self.sub_proc_environments[environment], self.model, n_steps=global_config.n_steps, gamma=0.99)

    def train_for_one_episode(self, game):
        runner = self.runners[game]
        score = self.model.multi_task_learn_for_one_episode(game, runner, self.writer)
        return score

    @staticmethod
    def play(model_id, game, max_number_of_games, show_render=False):
        number_of_games = 0
        sum_reward = 0
        model = MultitaskA2C.load(model_id)
        env = gym.make(game)
        obs = env.reset()
        while number_of_games < max_number_of_games:
            action = model.predict(game, obs)
            obs, rewards, done, info = env.step(action)
            if done:
                obs = env.reset()
                number_of_games += 1
                print(sum_reward)
                sum_reward = 0
            sum_reward += rewards
            if show_render is True:
                env.render()

    def save_model(self):
        try:
            if not os.path.exists("./data/models/" + self.algorithm + '_' + self.initialize_time):
                os.mkdir("./data/models/" + self.algorithm + '_' + self.initialize_time)
            self.model.save("./data/models/" + self.algorithm + '_' + self.initialize_time)
        except:
            print("Error at saving the model")

    def exit_tbw(self):
        self.tbw.exit(None, None, None)


