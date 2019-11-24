import git
import argparse
import config
import os

from utils import dir_check
from multiprocessing import cpu_count

from MultiTaskLearning import MultiTaskLearning
from MultiTaskAgent import MultiTaskAgent


def main(algorithm: str, selected_mti: str, policy: str, n_cpus: int, selected_gpus: str = "",
         train: bool = True, tb_log: bool = False, csv_log: bool = True, model_id: str = None):
    """
    :param algorithm: (str) 'A5C' or 'EA4C'
    :param selected_mti: (str) Name of the multi-task instance
     ['SpaceInvaders-v0', 'CrazyClimber-v0', 'Seaquest-v0', 'DemonAttack-v0', 'StarGunner-v0']'
    :param policy: (str) 'lstm' or 'ff' (feed forward)
    :param n_cpus: (int) number of the virtual cpus.
    :param selected_gpus: (str) the selected gpu ids eg.: '' or '0' or '0,1' or 'all'
    :param train: (bool) Whether to train or play
    :param tb_log: (bool) Whether to create tensorboard log or not. WARNING: IT LEAKS MEMORY IF IT IS TRUE
    :param csv_log: (bool) Whether or not to save results in csv files
    :param model_id: (str) or None Name of the model's directory which trying to load either for transfer learning or playing.
    """
    if isinstance(selected_mti, str):
        if selected_mti.lower() == 'mti1':
            selected_mti = config.MTI1
        elif selected_mti.lower() == 'mti2':
            selected_mti = config.MTI2
        elif selected_mti.lower() == 'mti3':
            selected_mti = config.MTI3
        elif selected_mti.lower() == 'mti4':
            selected_mti = config.MTI4
        elif selected_mti.lower() == 'mti5':
            selected_mti = config.MTI5
        elif selected_mti.lower() == 'mti6':
            selected_mti = config.MTI6
        elif selected_mti.lower() == 'mti7':
            selected_mti = config.MTI7
        elif selected_mti.lower() == 'mtic1':
            selected_mti = config.MTIC1
        elif selected_mti.lower() == 'mtic2':
            selected_mti = config.MTIC2
        elif selected_mti.lower() == 'mtic3':
            selected_mti = config.MTIC3
        else:
            selected_mti = config.MTIC1

    if selected_gpus != "all":
        if "CUDA_DEVICE_ORDER" in os.environ.keys() and "CUDA_VISIBLE_DEVICES" in os.environ.keys():
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpus
    if train:
        dir_check()
        if tb_log:
            print("WARNING: TENSORBOARD LOGGING ACTIVE -> IT LEAKS MEMORY")
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        json_params = {
            'git_sha': sha,
        }
        mt = MultiTaskLearning(selected_mti, algorithm, policy, n_cpus, json_params,
                               csv_log, model_id, tensorboard_logging=tb_log)
        mt.train()
    else:
        if model_id == '':
            raise ValueError("If you want to play, you should provide a model")
        MultiTaskAgent.play(model_id, n_games=1, display=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or play a multi-task agent!')
    parser.add_argument('--algorithm', type=str, nargs='?', help='Name of the multi-task algorithm, can only be A5C or EA4C. Default A5C', default='A5C')
    parser.add_argument('--mti', type=str, nargs='?', help='One of the predefined MTI see in config.py. Only used at training. Default mtic1', default='mtic1')
    parser.add_argument('--policy', type=str, nargs='?', help='Name of the desired policy can be ff (feed forward) or lstm. Default lstm', default='lstm')
    parser.add_argument('--gpu', type=str, nargs='?', help="Selected GPUs to train on can be '0' or '0,1' etc... or '' or 'all'. Default ''", default='')
    parser.add_argument('--play', help='Whether to play with the agent or train the agent', action='store_true')
    parser.add_argument('--tb_log', help='Whether you want tensorboard logging or not during training', action='store_true')
    parser.add_argument('--csv_log', help='Whether you want CSV logging or not during training', action='store_true')
    parser.add_argument('--model', type=str, nargs='?', help="ID (name of the directory in data/model) of the model you want to play with or "
                                                             "you want to transfer learn from. Default ''", default='')
    args = parser.parse_args()

    main(algorithm=args.algorithm, selected_mti=args.mti, policy=args.policy, n_cpus=cpu_count(),
         selected_gpus=args.gpu, train=not args.play, tb_log=args.tb_log, csv_log=args.csv_log, model_id=args.model)
