from random import uniform as ran
from typing import List, Tuple

from matplotlib.colors import cnames
from numpy import array_split, concatenate, dtype, ndarray

colors = list(cnames.keys())


def RandomList(start: float, end: float, N: int, classe: str) -> list[Tuple]:
    return [
        (ran(start, end), ran(start, end), ran(start, end), classe) for _ in range(N)
    ]


def RandomColorList(N: int) -> list[str]:
    return colors[7 : 7 + N]


def PointsGenerator(k: int, N: int, start: int, end: int) -> ndarray[List[str], dtype]:
    return concatenate(
        [
            RandomList(s[0], s[-1], N // k, c)
            for s, c in zip(array_split(list(range(start, end)), k), RandomColorList(k))
        ]
    )


if __name__ == "__main__":
    import numpy as np

    target = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(arr[np.where((arr == 7) | (arr == 8))[0]])
