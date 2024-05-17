import timeit
from itertools import chain
from turtle import distance, shape
from typing import Any, List, Tuple

import numpy as np
from tqdm import tqdm

dtype = np.dtype
ndarray = np.ndarray


#  KNN_Paralel:
def TopKMin(
    points_distance: ndarray[float, dtype], label: ndarray[int, dtype], k: int
) -> Tuple[ndarray[float, dtype], ndarray[int, dtype]]:
    if len(points_distance) != len(label):
        print(f"{points_distance,label = }")
        raise ValueError("points_distance and label must have the same length")
    if len(points_distance) == k:
        return points_distance, label
    Kmin = np.argpartition(points_distance, k)[:k]
    return points_distance[Kmin], label[Kmin]


def get_distances(
    points: ndarray[Tuple, dtype], new_point: ndarray[Tuple, dtype]
) -> ndarray[float, dtype]:
    return (
        np.sum(
            (points - new_point) ** 2,
            axis=1,
        )
        ** 0.5
    )


def predict(label: ndarray[Any, dtype]) -> str:
    return max(set(label), key=label.tolist().count)


def SlaveWork(
    X_data: ndarray[tuple, dtype],
    Y_data: ndarray[Any, dtype],
    new_points: ndarray[tuple, dtype],
    k: int,
    rank: int,
) -> tuple[list[tuple[ndarray[float, dtype], ndarray[Any, dtype]]], float]:
    start = timeit.default_timer()
    result: List[Tuple[ndarray[float, dtype], ndarray[Any, dtype]]] = []
    for tple in tqdm(new_points, desc=f"Slave {rank}"):
        result.append(TopKMin(get_distances(X_data, tple), Y_data, k))
    end = timeit.default_timer()
    return (result, (end - start))
