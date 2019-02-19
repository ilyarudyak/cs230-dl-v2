from data_utils import is_overlapping


def test_is_overlapping():
    overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
    overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])

    assert not overlap1
    assert overlap2


if __name__ == '__main__':
    test_is_overlapping()
