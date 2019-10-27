import os
import tensorflow as tf
import glob


class TensorboardWriter:
    def __init__(self, graph, tensorboard_log_path, tb_log_name):
        """
        Create a Tensorboard writer for a code segment, and saves it to the log directory as its own run

        :param graph: (Tensorflow Graph) the model graph
        :param tensorboard_log_path: (str) the save path for the log (can be None for no logging)
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param new_tb_log: (bool) whether or not to create a new logging folder for tensorbaord
        """
        self.graph = graph
        self.tensorboard_log_path = tensorboard_log_path
        self.tb_log_name = tb_log_name
        self.writer = None

        self.__enter()

    def __enter(self):
        if self.tensorboard_log_path is not None:
            save_path = os.path.join(self.tensorboard_log_path, "{}".format(self.tb_log_name))
            self.writer = tf.summary.FileWriter(save_path, graph=self.graph)

    def exit(self):
        if self.writer is not None:
            self.writer.add_graph(self.graph)
            self.writer.close()

    def flush(self):
        if self.writer is not None:
            self.writer.flush()
