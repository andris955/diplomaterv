from MultiTaskAlgorithm import MultiTasking
import global_config
from Agent import Agent

def main(selected_mti, n_cpus, train=True):
    amta = Agent(selected_mti, global_config.MaxSteps, n_cpus)
    mt = MultiTasking(selected_mti, "A5C", 3, global_config.target_performances, global_config.l, global_config.MaxSteps, amta)
    if train:
        mt.train()
    else:
        for game in selected_mti:
            amta.play(global_config.model_path, game, 3, True)


if __name__ == '__main__':
    main(global_config.MTI2, 2)
