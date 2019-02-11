from bleu_compute import *
import math

MPe1n1 = 17 / 18
MPe1n2 = 10 / 17
MPe2n1 = 8 / 14
MPe2n2 = 1 / 13


def test_unigram_precision():
    e1 = Example1()
    score1 = get_modified_unigram_precision(e1.reference_list, e1.candidate1)
    score2 = get_modified_unigram_precision(e1.reference_list, e1.candidate2)
    assert math.isclose(score1, MPe1n1)


def test_bigram_precision():
    e1 = Example1()
    score1 = get_modified_bigram_precision(e1.reference_list, e1.candidate1)
    score2 = get_modified_bigram_precision(e1.reference_list, e1.candidate2)
    assert math.isclose(score1, MPe1n2)


def test_modified_precision_unigrams():
    e1 = Example1()
    score1 = modified_precision(e1.reference_list, e1.candidate1, n=1)
    score2 = modified_precision(e1.reference_list, e1.candidate2, n=1)
    assert math.isclose(score1, MPe1n1)
    assert math.isclose(score2, MPe2n1)


def test_modified_precision_bigrams():
    e1 = Example1()
    score1 = modified_precision(e1.reference_list, e1.candidate1, n=2)
    score2 = modified_precision(e1.reference_list, e1.candidate2, n=2)
    assert math.isclose(score1, MPe1n2)
    assert math.isclose(score2, MPe2n2)


if __name__ == '__main__':
    test_unigram_precision()
    test_bigram_precision()
