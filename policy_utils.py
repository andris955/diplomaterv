import tensorflow as tf


def batch_to_seq(tensor_batch, n_steps, flat=False):
    """
    Transform a batch of Tensors, into a sequence of Tensors for recurrent policies

    :param tensor_batch: (TensorFlow Tensor) The input tensor to unroll
    :param n_steps: (int) The number of steps to run for each environment
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) sequence of Tensors for recurrent policies
    """
    if flat:
        tensor_batch2 = tf.reshape(tensor_batch, [None, n_steps])
    else:
        if len(tensor_batch.shape) == 2: # input sequence
            tensor_batch2 = tf.reshape(tensor_batch, [tf.shape(tensor_batch)[0]/n_steps, n_steps, tensor_batch.shape[1].value])
        elif len(tensor_batch.shape) == 1: # mask
            tensor_batch2 = tf.reshape(tensor_batch, [tf.shape(tensor_batch)[0]/n_steps, n_steps, 1])
        else:
            raise ValueError("Invalid shape: {}. Should have 1 or 2 dimension".format(tensor_batch.shape))

    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=n_steps, value=tensor_batch2)]
