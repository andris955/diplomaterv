from MultiTaskAlgorithm import MultiTasking
import global_config
from Agent import Agent


def main(selected_mti, n_cpus, train=True, transfer_id=None):
    if train:
        mt = MultiTasking(selected_mti, "A5C", 3, global_config.target_performances, global_config.l, global_config.MaxSteps, n_cpus, transfer_id)
        mt.train()
    else:
        for game in selected_mti:
            Agent.play(global_config.model_id, game, max_number_of_games=3, show_render=True)


if __name__ == '__main__':
    main(selected_mti=global_config.MTIC, n_cpus=2, train=False)
