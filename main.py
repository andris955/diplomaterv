from MultiTaskAlgorithm import MultiTasking
import global_config
from Agent import Agent
import local_config


if __name__ == '__main__':
    selected_mti = global_config.MTI2
    amta = Agent(selected_mti, global_config.MaxSteps)
    mt = MultiTasking(selected_mti, "A5C", 3, global_config.target_performances, global_config.l, global_config.MaxSteps, amta)
    if local_config.train:
        mt.train()
    else:
        for game in selected_mti:
            amta.play(local_config.model_path, game, 3, True)
