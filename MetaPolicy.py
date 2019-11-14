import numpy as np

from keras import backend as K
import tensorflow as tf

from stable_baselines.common.distributions import CategoricalProbabilityDistributionType
from stable_baselines.a2c.utils import linear


class MetaLstmActorCriticPolicy:
    def __init__(self, sess: tf.Session, input_length: int, output_length: int, n_steps: int, layers: list, lstm_units: int):
        self.input_length = input_length
        self.output_length = output_length
        self.n_steps = n_steps
        assert self.n_steps > 0, "n_batch must be a positive integer!"
        self.layers = layers
        self.lstm_units = lstm_units

        self.sess = sess
        self.pdtype = CategoricalProbabilityDistributionType(self.output_length)
        self.policy = None
        self.q_value = None
        self.proba_distribution = None
        self.value_fn = None
        self.initial_state = None

        self.__setup_model()

    def __setup_model(self):
        # Register session
        K.set_session(self.sess)

        lstm_registered = False
        latent_vector = None
        x = None

        # Input
        self.input_ph = tf.placeholder(shape=(self.n_steps, self.input_length), dtype=tf.float32, name="input_ph")

        # Building the shared part of the network
        for i, layer in enumerate(self.layers):
            if isinstance(layer, int):
                if i == 0:
                    x = tf.keras.layers.Dense(layer, activation='relu', name='fc_' + str(i))(self.input_ph)
                elif i < len(self.layers) - 1:
                    x = tf.keras.layers.Dense(layer, activation='relu', name='fc_' + str(i))(x)
            elif layer == "lstm" and lstm_registered is False and i == len(self.layers)-1:
                x = tf.keras.backend.expand_dims(x, axis=0)
                latent_vector = tf.keras.layers.LSTM(self.lstm_units, name="lstm")(x)
                lstm_registered = True
            else:
                raise ValueError("layers are not correct: it has to contain arbitrary number of integers and finally str(lstm)")

        # Building the non-shared part of the network
        self.value_fn = linear(latent_vector, 'mvf', 1)
        self.proba_distribution, self.policy, self.q_value = \
            self.pdtype.proba_distribution_from_latent(latent_vector, latent_vector)

        # Outputs
        assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
        self.flat_param = self.proba_distribution.flatparam()
        self.action = self.proba_distribution.sample()
        self.neglogp = self.proba_distribution.neglogp(self.action)
        self._value = self.value_fn[:, 0]

        self.initial_state = np.zeros((self.n_steps, self.input_length), dtype=np.float32)

    def step(self, input_state: np.ndarray):
        flat_param, value, neglogp = self.sess.run([self.flat_param, self._value, self.neglogp], {self.input_ph: input_state})
        return flat_param, value, neglogp

    def value(self, input_state: np.ndarray):
        value = self.sess.run(self._value, {self.input_ph: input_state})
        return value
