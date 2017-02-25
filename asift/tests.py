from .common import get_by_sorted_indices


class TestGetBySortedIndices:
    """assume here that sort is stable"""
    def test_empty(self):
        assert list(get_by_sorted_indices((0, 0), [[]])) == []

    def test_simple(self):
        arr = [[1, 2, 3, 4, 5]]

        srt = list(get_by_sorted_indices((0, 2), arr))

        assert srt == [3, 2, 4, 1, 5]

    def test_complex(self):
        arr = [[10, 11, 12, 13, 14],
               [20, 21, 22, 23, 24],
               [30, 31, 32, 33, 34]]

        srt = list(get_by_sorted_indices((1, 2), arr))

        assert srt == [22, 12, 21, 23, 32, 11, 13, 20, 24, 31, 33, 10, 14, 30, 34]
