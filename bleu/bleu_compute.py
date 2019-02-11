from collections import Counter
from nltk.util import ngrams
from fractions import Fraction
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


def modified_precision(references, hypothesis, n):
    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()

    # Extract a union of references' counts.
    # max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = (
            Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        )
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {
        ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()
    }

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator, _normalize=False)


if __name__ == '__main__':
    e1 = Example1()
    references = e1.reference_list
    hypothesis = e1.candidate2
    n = 2
    md = modified_precision(references, hypothesis, n)
    print(md)

