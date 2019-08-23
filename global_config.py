MTI1 = ['SpaceInvaders-v0', 'CrazyClimber-v0', 'Seaquest-v0', 'DemonAttack-v0', 'StarGunner-v0']
#actionspace disc 6             disc 9              disc 18         disc 6          disc 18
MTI2 = ['Breakout-v0', 'Seaquest-v0']
target_performances = {
    'SpaceInvaders-v0': 100,
    'CrazyClimber-v0': 100,
    'Seaquest-v0': 100,
    'DemonAttack-v0': 100,
    'StarGunner-v0': 100,
    'Breakout-v0': 1000 #TODO random
}

l = 100000

MaxSteps = 100000
tau = 10