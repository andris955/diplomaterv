from MultiTaskAlgorithm import MultiTaskLearning
import global_config
from Agent import Agent
from utils import dir_check
import os
from multiprocessing import cpu_count


def main(algorithm: str, selected_mti: list, n_cpus: int, max_train_steps, uniform_policy_steps: int, selected_gpus: str = None, train: bool = True,
         tensorboard_logging: str = None, logging: bool = True, transfer_id: str = None, model_id: str = None):
    if selected_gpus != None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpus
    if train:
        dir_check()
        mt = MultiTaskLearning(selected_mti, algorithm, global_config.number_of_episodes_for_estimating,
                               global_config.target_performances, uniform_policy_steps,
                               max_train_steps, n_cpus, logging, transfer_id, tensorboard_logging=tensorboard_logging)
        mt.train()
    else:
        Agent.play(model_id, max_number_of_games=3, display=True)


if __name__ == '__main__':
    main(algorithm='A5C', selected_mti=global_config.MTI1, n_cpus=cpu_count(), max_train_steps=global_config.MaxTrainSteps,
         uniform_policy_steps=100, selected_gpus=None, train=True, tensorboard_logging=None, logging=False,
         transfer_id=None, model_id=None)
