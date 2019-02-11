from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt


# noinspection PyAttributeOutsideInit
class NMT:
    def __init__(self):
        self.Tx = 30
        self.Ty = 10
        self.n_a = 64
        self.n_s = 128
        self.m = 10000

        self.opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)

        self.examples = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007',
                         'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']

        self._get_data()
        self._build_shared_layers()
        self._build_model()

    def _get_data(self):
        dataset, self.human_vocab, self.machine_vocab, self.inv_machine_vocab = load_dataset(self.m)
        X, Y, self.Xoh, self.Yoh = preprocess_data(dataset, self.human_vocab,
                                                   self.machine_vocab, self.Tx, self.Ty)
        self.human_vocab_size = len(self.human_vocab)  # 37
        self.machine_vocab_size = len(self.machine_vocab)  # 11

    def _build_shared_layers(self):
        self.repeator = RepeatVector(self.Tx)
        self.concatenator = Concatenate(axis=-1)
        self.densor = Dense(1, activation="relu")
        self.activator = Activation(softmax, name='attention_weights')
        self.dotor = Dot(axes=1)
        self.post_activation_LSTM_cell = LSTM(self.n_s, return_state=True)
        self.output_layer = Dense(len(self.machine_vocab), activation=softmax)

    def _one_step_attention(self, a, s_prev):
        s_prev = self.repeator(s_prev)
        concat = self.concatenator([a, s_prev])
        e = self.densor(concat)
        alphas = self.activator(e)
        context = self.dotor([alphas, a])
        return context

    def _build_model(self):
        """
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- size of the python dictionary "machine_vocab"

        Returns:
        model -- Keras model instance
        """

        # Define the inputs of your model with a shape (Tx,)
        # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
        X = Input(shape=(self.Tx, self.human_vocab_size))
        s0 = Input(shape=(self.n_s,), name='s0')
        c0 = Input(shape=(self.n_s,), name='c0')
        s = s0
        c = c0

        # Initialize empty list of outputs
        outputs = []

        # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
        a = Bidirectional(LSTM(self.n_a, return_sequences=True))(X)

        # Step 2: Iterate for Ty steps
        for t in range(self.Ty):
            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
            context = self._one_step_attention(a, s)

            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
            # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
            s, _, c = self.post_activation_LSTM_cell(context, initial_state=[s, c])

            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = self.output_layer(s)

            # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
            outputs.append(out)

        # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
        self.model = Model([X, s0, c0], outputs)

        self.model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy'])

    def model_fit(self, epochs=1):
        s0 = np.zeros((self.m, self.n_s))
        c0 = np.zeros((self.m, self.n_s))
        outputs = list(self.Yoh.swapaxes(0, 1))
        self.model.fit([self.Xoh, s0, c0], outputs, epochs=epochs, batch_size=100)

    def model_predict(self):
        s0 = np.zeros((self.m, self.n_s))
        c0 = np.zeros((self.m, self.n_s))
        for example in self.examples:
            source = string_to_int(example, self.Tx, self.human_vocab)
            source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(self.human_vocab)),
                                       source)), ndmin=3)
            prediction = self.model.predict([source, s0, c0])
            prediction = np.argmax(prediction, axis=-1)
            output = [self.inv_machine_vocab[int(i)] for i in prediction]

            print("source:", example)
            print("output:", ''.join(output))

    def model_load_weights(self, file_path):
        self.model.load_weights(filepath=file_path)


if __name__ == '__main__':
    nmt = NMT()
    # nmt.model_fit(epochs=1)
    nmt.model_load_weights(file_path='models/model_kiank.h5')
    nmt.model_predict()


