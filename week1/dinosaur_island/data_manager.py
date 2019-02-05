import numpy as np
from utils import *
import random


def get_data():
    """
    Read file into one long string: low case letters,
    including '\n', no spaces:
    In [5]: data[:50]
    Out[5]: 'aachenosaurus\naardonyx\nabdallahsaurus\nabelisaurus\n'
    :return:
    """
    data = open('dinos.txt', 'r').read()
    data = data.lower()
    chars = list(set(data))  # 27 characters including 26 letters and '\n'
    char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
    ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
    return data, char_to_ix, ix_to_char
