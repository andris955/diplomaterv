from MultiTaskLearning import MultiTaskLearning
import global_config
from MultiTaskAgent import MultiTaskAgent
from utils import dir_check
import os
from multiprocessing import cpu_count


def main(algorithm: str, selected_mti: list, policy: str, n_cpus: int, selected_gpus: str = None,
         train: bool = True, tensorboard_logging: str = None, logging: bool = True, model_id: str = None):

    if selected_gpus is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpus
    if train:
        dir_check()
        mt = MultiTaskLearning(selected_mti, algorithm, policy, global_config.target_performances,
                               n_cpus, logging, model_id, tensorboard_logging=tensorboard_logging)
        mt.train()
    else:
        MultiTaskAgent.play(model_id, max_number_of_games=3, display=True)


if __name__ == '__main__':
    main(algorithm='A5C', selected_mti=global_config.MTI1, policy="lstm", n_cpus=cpu_count(),
         selected_gpus=None, train=True, tensorboard_logging=None, logging=False, model_id=None)
