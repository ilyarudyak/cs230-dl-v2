import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, SpatialDropout1D, SimpleRNN, LSTM
from keras.layers import Embedding # new!
from keras.callbacks import ModelCheckpoint # new!
import os # new!
from sklearn.metrics import roc_auc_score, roc_curve # new!
import pandas as pd
import matplotlib.pyplot as plt # new!


# noinspection PyAttributeOutsideInit
class Classifier:
    def __init__(self):
        self.output_dir = None

        # training:
        self.epochs = 4
        self.batch_size = 128

        # vector-space embedding:
        self.n_dim = 64
        self.n_unique_words = 10000
        self.n_words_to_skip = 50
        self.max_review_length = 100
        self.pad_type = self.trunc_type = 'pre'
        self.drop_embed = 0.2
        self.n_test = 5000

        # dense network architecture:
        self.n_dense = 64
        self.dropout = 0.5

        # RNN layer architecture:
        self.n_rnn = 256
        self.drop_rnn = 0.2

        # LSTM layer architecture:
        self.n_lstm = 256
        self.drop_lstm = 0.2

        self.model = Sequential()
        self.build_model()
        self.compile_model()

        self.get_data()

    def get_data(self):
        (self.x_train, self.y_train), (self.x_valid, self.y_valid) = imdb.load_data(
            num_words=self.n_unique_words, skip_top=self.n_words_to_skip)
        self.x_train = pad_sequences(self.x_train, maxlen=self.max_review_length,
                                     padding=self.pad_type, truncating=self.trunc_type, value=0)
        self.x_valid = pad_sequences(self.x_valid, maxlen=self.max_review_length,
                                     padding=self.pad_type, truncating=self.trunc_type, value=0)

        self.x_test = self.x_valid[:self.n_test]
        self.y_test = self.y_valid[:self.n_test]
        self.x_valid = self.x_valid[self.n_test:]
        self.y_valid = self.y_valid[self.n_test:]

    def build_model(self):
        raise NotImplementedError

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit_model(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        model_checkpoint = ModelCheckpoint(filepath=self.output_dir + "/weights.{epoch:02d}-{val_acc:.2f}.hdf5",
                                           save_best_only=True)
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size,
                       epochs=self.epochs, verbose=1,
                       validation_data=(self.x_valid, self.y_valid), callbacks=[model_checkpoint])

    def evaluate_model(self):
        score, acc = self.model.evaluate(self.x_test, self.y_test,
                                         batch_size=self.batch_size)
        print(f'test loss:{score:.4f} test accuracy:{acc:.4f}')


class DenseClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.output_dir = 'model_output/dense'
        self.epochs = 4

    def build_model(self):
        self.model.add(Embedding(self.n_unique_words, self.n_dim,
                                 input_length=self.max_review_length))
        self.model.add(Flatten())
        self.model.add(Dense(self.n_dense, activation='relu'))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(1, activation='sigmoid'))


class RNNClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.output_dir = 'model_output/rnn'
        self.epochs = 16

    def build_model(self):
        self.model.add(Embedding(self.n_unique_words, self.n_dim,
                                 input_length=self.max_review_length))
        self.model.add(SpatialDropout1D(self.drop_embed))
        self.model.add(SimpleRNN(self.n_rnn, dropout=self.drop_rnn))
        self.model.add(Dense(1, activation='sigmoid'))


class VanillaLSTMClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.output_dir = 'model_output/vanilla_lstm'
        self.epochs = 4

    def build_model(self):
        self.model.add(Embedding(self.n_unique_words, self.n_dim, input_length=self.max_review_length))
        self.model.add(SpatialDropout1D(self.drop_embed))
        self.model.add(LSTM(self.n_lstm, dropout=self.drop_lstm))
        self.model.add(Dense(1, activation='sigmoid'))


if __name__ == '__main__':
    dm = VanillaLSTMClassifier()
    dm.fit_model()
    dm.evaluate_model()

