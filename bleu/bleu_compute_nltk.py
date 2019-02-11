from nltk.translate.bleu_score import sentence_bleu
from example1 import Example1
from example2 import Example2


def get_modified_unigram_precision(reference, candidate):
    """
    Compute Modified Unigram Precision using nltk.
    To compute it we have to set weights = (1, 0, 0, 0).
    """
    return sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))


def get_modified_bigram_precision(reference, candidate):
    """
    Compute Modified Unigram Precision using nltk.
    To compute it we have to set weights = (1, 0, 0, 0).
    """
    return sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))


if __name__ == '__main__':
    pass

