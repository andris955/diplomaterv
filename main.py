from MultiTaskAlgorithm import MultiTasking
import global_config
from Agent import Agent
from utils import dir_check


def main(algorithm, selected_mti, n_cpus, max_steps, uniform_policy_steps, train=True, transfer_id=None, model_id=None):
    if train:
        dir_check()
        mt = MultiTasking(selected_mti, algorithm, 3, global_config.target_performances, uniform_policy_steps, max_steps, n_cpus, transfer_id)
        mt.train()
    else:
        Agent.play(model_id, max_number_of_games=3, show_render=True)


if __name__ == '__main__':
    main(algorithm='A5C', selected_mti=global_config.MTI1, n_cpus=2, max_steps=global_config.MaxSteps, uniform_policy_steps=100, train=True, transfer_id=None, model_id=global_config.model_id)
