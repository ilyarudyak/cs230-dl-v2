import numpy as np
from data_utils import is_overlapping, is_two_segments_overlapping, insert_ones


def test_is_two_segments_overlapping():
    assert is_two_segments_overlapping((100, 199), (199, 200))
    assert is_two_segments_overlapping((100, 199), (0, 100))
    assert is_two_segments_overlapping((100, 199), (75, 225))
    assert is_two_segments_overlapping((100, 199), (125, 175))

    assert not is_two_segments_overlapping((100, 199), (200, 300))
    assert not is_two_segments_overlapping((100, 199), (0, 99))


def test_is_overlapping():
    assert not is_overlapping((100, 199), [(200, 300), (0, 99)])
    assert is_overlapping((100, 199), [(199, 200), (0, 100), (75, 225), (125, 175)])


def test_insert_ones():
    Ty = 1375
    arr1 = insert_ones(np.zeros((1, Ty)), segment_end_ms=0)
    assert arr1[0, 0] == 0
    assert arr1[0, 1] == 1
    assert arr1[0, 50] == 1
    assert arr1[0, 51] == 0

    arr1 = insert_ones(np.zeros((1, Ty)), segment_end_ms=10**6/Ty)
    assert arr1[0, 100] == 0
    assert arr1[0, 101] == 1
    assert arr1[0, 150] == 1
    assert arr1[0, 151] == 0


if __name__ == '__main__':
    test_is_overlapping()
