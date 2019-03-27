class Agent:
    def __init__(self, SetOfEnvironments, model):
        self.environments = SetOfEnvironments
        self.model = model

    def train_for_one_episode(self, T):
        self.model.train()
