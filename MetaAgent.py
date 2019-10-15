


class MetaAgent:
    def __init__(self):
        self.algorithm = None

    def train_and_sample(self, state, reward):
        p = self.algorithm.learn(state=state, reward=reward)
        return p
