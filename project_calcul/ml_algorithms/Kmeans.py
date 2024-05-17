from typing import Dict, List, Tuple

import numpy as np
from numpy import dtype, ndarray

from project_calcul.ml_algorithms.tools import random_arrays_choice


def ArrayCentroids(centroids: dict) -> ndarray[Tuple, dtype]:
    return np.array(list(centroids.keys()))


def ArrayNormalizer(Array: ndarray):
    return np.array(Array, dtype="f")


def GenDictForClusters(centroids: ndarray, shape: int) -> dict[int, ndarray]:
    return {i: np.array([]).reshape(0, shape) for i in range(len(centroids))}


def GetRandomCentroids(
    points: ndarray[List[float], dtype], k: int
) -> ndarray[List[float], dtype]:
    return np.array(random_arrays_choice(points, k))


def PreCentroids(
    points: dict[int, ndarray[List[float], dtype]]
) -> Tuple[ndarray[List[Tuple], dtype], ndarray[List[int], dtype]]:
    return np.array([np.sum(p, axis=0) for p in points.values()]), np.array(
        [len(p) for p in points.values()]
    )


def GetCentroids(pre_new_centroids) -> ndarray[List[float], dtype]:
    return np.array(
        [
            s / l
            for s, l in zip(
                np.sum([np.array(i[0]) for i in pre_new_centroids], axis=0),
                np.sum([np.array(i[1]) for i in pre_new_centroids], axis=0),
            )
            if l > 0
        ]
    )


def clustering(
    points: ndarray,
    centroids: ndarray,
) -> Tuple[Tuple[List[ndarray], List[int]], ndarray[int, dtype]]:
    labels = np.array([])
    sum_per_cluster = [np.zeros(points.shape[1]) for _ in range(len(centroids))]
    nb_per_cluster = [0 for _ in range(len(centroids))]
    for point in points:
        distances = np.linalg.norm(centroids - point, axis=1)
        index = int(np.argmin(distances))
        labels = np.append(labels, index)
        sum_per_cluster[index] += point
        nb_per_cluster[index] += 1
    return (sum_per_cluster, nb_per_cluster), labels


def has_converged(
    old_centroids: ndarray, new_centroids: ndarray, threshold: float = 0.000001
) -> bool:
    """
    Check if the centroids have moved less than a certain threshold between iterations.

    Parameters:
        old_centroids (ndarray): The centroids from the previous iteration.
        new_centroids (ndarray): The centroids from the current iteration.
        threshold (float): The threshold for considering convergence.

    Returns:
        bool: True if the centroids have moved less than the threshold, False otherwise.
    """
    try:
        centroid_distances = np.linalg.norm(new_centroids - old_centroids, axis=1)
    except:
        print(f"{new_centroids , old_centroids = }")
        return False
    return all(distance < threshold for distance in centroid_distances)
