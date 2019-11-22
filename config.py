MTI1 = ['SpaceInvaders-v0', 'CrazyClimber-v0', 'Seaquest-v0', 'DemonAttack-v0', 'StarGunner-v0']  # disc, obs jó
#actionspace disc 6             disc 9              disc 18         disc 6          disc 18
MTI2 = ["Asterix-v0", "Alien-v0", "Assault-v0", "TimePilot-v0", "Gopher-v0", "ChopperCommand-v0"]  # disc, változó obs méret

MTI3 = ['Breakout-v0', 'Centipede-v0', 'Frostbite-v0', 'Qbert-v0', 'KungFuMaster-v0', 'WizardOfWor-v0']  # disc, observation nem jó változik, de csak a méret

MTI4 = ['Atlantis-v0', 'Breakout-v0', 'Bowling-v0', 'CrazyClimber-v0', 'Seaquest-v0', 'SpaceInvaders-v0', 'Pong-v0', 'Enduro-v0']  # disc, observation jo

MTI5 = ['SpaceInvaders-v0', 'Seaquest-v0', 'Asterix-v0', 'Alien-v0', 'Assault-v0', 'BankHeist-v0', 'CrazyClimber-v0', 'DemonAttack-v0', 'Gopher-v0', 'NameThisGame-v0', 'StarGunner-v0', 'Tutankham-v0']

MTI6 = ['Atlantis-v0', 'Amidar-v0', 'Breakout-v0', 'Bowling-v0', 'BeamRider-v0', 'ChopperCommand-v0', 'Centipede-v0', 'Frostbite-v0', 'KungFuMaster-v0', 'Pong-v0', 'RoadRunner-v0', 'Phoenix-v0']

MTI7 = ['SpaceInvaders-v0', 'Seaquest-v0', 'Asterix-v0', 'Alien-v0', 'Assault-v0', 'BankHeist-v0', 'CrazyClimber-v0', 'DemonAttack-v0', 'Gopher-v0', 'NameThisGame-v0', 'StarGunner-v0', 'Tutankham-v0', 'Amidar-v0', 'ChopperCommand-v0', 'Breakout-v0', 'BeamRider-v0', 'Bowling-v0', 'Centipede-v0', 'Krull-v0', 'Kangaroo-v0', 'Phoenix-v0']

MTIC1 = ['SpaceInvaders-v0', 'Seaquest-v0', 'CrazyClimber-v0']  # különböző, de jó

MTIC2 = ['SpaceInvaders-v0', 'Assault-v0', 'DemonAttack-v0']  # hasonló (1D-ben mozog és lő), de rossz

MTIC3 = ['Seaquest-v0', 'StarGunner-v0', 'ChopperCommand-v0']  # hasonló (2D-ben mozog és lő)


target_performances = {
    'SpaceInvaders-v0': 1200,
    'Seaquest-v0': 2700,
    'Asterix-v0': 2400,
    'Alien-v0': 2700,
    'Assault-v0': 1900,
    'TimePilot-v0': 9000,
    'BankHeist-v0': 1700,
    'CrazyClimber-v0': 170000,
    'DemonAttack-v0': 27000,
    'Gropher-v0': 9400,
    'NameThisGame-v0': 12100,
    'StarGunner-v0': 40000,
    'Tutankham-v0': 260,
    'Amidar-v0': 1030,
    'ChopperCommand-v0': 4970,
    'Breakout-v0': 560,
    'BeamRider-v0': 2200,
    'Bowling-v0': 17,
    'Centipede-v0': 3300,
    'Krull-v0': 1025,
    'Kangaroo-v0': 26,
    'Phoenix-v0': 5384,
    'Atlantis-v0': 163660,
    'Frostbite-v0': 300,
    'KungFuMaster-v0': 36000,
    'Pond-v0': 19.5,
    'RoadRunner-v0': 59540,
    'Qbert-v0': 26000,
    'WizardOfWor-v0': 3300,
    'Enduro-v0': 0.77
}

number_of_episodes_for_estimating = 3
uniform_policy_steps = 100_000  # Number of time steps for which a uniformly random policy is executed for task selection. At the end of l training steps, the agent must have learned on ≥ n
max_timesteps = 300_000_000  # k x 50_000_000 time stepig futott az eredeti cikkbe.
tau = 1
n_steps = 10
seed = 3

stdout_logging_frequency_in_train_steps = 100
file_logging_frequency_in_episodes = 10  # number of episodes
verbose = 1

meta_layers = (100, 100, "lstm")
meta_lstm_units = 100
meta_lambda = 0.5

tensorboard_log = "./data/tb_logs/"
model_path = "./data/models/"
log_path = "./data/logs/"

# if __name__ == '__main__':
#     import gym
#     import time
#     for game in MTIC2:
#         print(game)
#         env = gym.make(game)
#         env.reset()
#         for _ in range(150):
#             action = env.action_space.sample()
#             env.step(action)
#             env.render()
#             time.sleep(0.05)
#     print("ok")
