from MultiTaskLearning import MultiTaskLearning
import config
from MultiTaskAgent import MultiTaskAgent
from utils import dir_check
import os
from multiprocessing import cpu_count


def main(algorithm: str, selected_mti: list, policy: str, n_cpus: int, selected_gpus: str = "all",
         train: bool = True, tensorboard_logging: bool = False, logging: bool = True, model_id: str = None):
    """
    :param algorithm: (str) 'A5C' or 'EA4C'
    :param selected_mti: (list) List with the names of the games eg.:
     ['SpaceInvaders-v0', 'CrazyClimber-v0', 'Seaquest-v0', 'DemonAttack-v0', 'StarGunner-v0']'
    :param policy: (str) 'lstm' or 'ff' (feed forward)
    :param n_cpus: (int) number of the virtual cpus.
    :param selected_gpus: (str) the selected gpu ids eg.: '0' or '0,1' or 'all'
    :param train: (bool) Train or play
    :param tensorboard_logging: (bool) Whether to create tensorboard log or not. WARNING: IT LEAKS MEMORY IF IT IS TRUE
    :param logging: (bool) Whether or not to save results in csv files
    :param model_id: (str) or None Name of the model's directory which trying to load either for transfer learning or playing.
    """

    if selected_gpus != "all" and not selected_gpus != 'no':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpus
    if train:
        dir_check()
        if tensorboard_logging:
            print("WARNING: TENSORBOARD LOGING ACTIVE -> IT LEAKS MEMORY")
        mt = MultiTaskLearning(selected_mti, algorithm, policy, config.target_performances,
                               n_cpus, logging, model_id, tensorboard_logging=tensorboard_logging)
        mt.train()
    else:
        MultiTaskAgent.play(model_id, max_number_of_games=3, display=True)


if __name__ == '__main__':
    main(algorithm='EA4C', selected_mti=config.MTI1, policy="lstm", n_cpus=cpu_count(),
         selected_gpus="all", train=True, tensorboard_logging=False, logging=False, model_id=None)
