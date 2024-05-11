from random import choices
from random import uniform as ran
from typing import Tuple

from matplotlib.colors import cnames
from numpy import array, dtype, ndarray

colors = list(cnames.keys())


def RandomList(start: float, end: float, N: int) -> list[Tuple[float, float]]:
    return [(ran(start, end), ran(start, end)) for _ in range(N)]


def RandomColorList(N: int) -> list[str]:
    return colors[:N]


def PointsGenerator(
    k: int, N: int, start: float, end: float
) -> ndarray[Tuple[str, Tuple[float, float]], dtype]:
    return array(
        [
            list((i[0], *i[1]))
            for i in zip(choices(RandomColorList(k), k=N), RandomList(start, end, N))
        ]
    )


if __name__ == "__main__":
    ...
