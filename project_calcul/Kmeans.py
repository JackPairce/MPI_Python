from typing import Any, Dict, Tuple

import numpy as np
from numpy import dtype, ndarray


def ArrayCentroids(centroids: dict) -> ndarray[Tuple[float, float], dtype]:
    return np.array(list(centroids.keys()))


def GenDictForClusters(k: int, shape: int) -> dict[str, ndarray]:
    return {f"c{i}": np.array([]).reshape(0, shape) for i in range(k)}


def GetRandomCentroids(
    points: ndarray[Tuple[float, float], dtype], k: int
) -> dict[Any, str]:
    return {
        tuple(points[p]): f"c{i}"
        for i, p in enumerate(np.random.choice(len(points), k))
    }


def GetCentroids(Output: dict[str, ndarray]) -> dict[Tuple, str]:
    return {
        tuple(np.sum(value, axis=0) / len(value)): key
        for key, value in Output.items()
        if len(value) > 0
    }


def clustering(
    points: ndarray,
    centroids: ndarray,
) -> Tuple[Dict[str, ndarray], ndarray]:
    cluster_output = GenDictForClusters(len(centroids), points.shape[1])
    labels = np.zeros(len(points))
    for i, point in enumerate(points):
        distances = np.linalg.norm(centroids - point, axis=1)
        closest_centroid_idx = np.argmin(distances)
        labels[i] = closest_centroid_idx
        closest_centroid_key = f"c{closest_centroid_idx}"
        cluster_output[closest_centroid_key] = np.append(
            cluster_output[closest_centroid_key], np.array([point]), axis=0
        )
    return cluster_output, labels


def has_converged(
    old_centroids: ndarray, new_centroids: ndarray, threshold: float
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
    centroid_distances = np.linalg.norm(new_centroids - old_centroids, axis=1)
    return all(distance < threshold for distance in centroid_distances)


def SlaveWork(
    points: ndarray,
    centroids: Dict[Tuple, str],
    threshold: float,
    max_iterations: int,
) -> Tuple[Dict[str, ndarray], ndarray]:
    for _ in range(max_iterations):
        old_centroids = centroids
        cluster_output, labels = clustering(points, ArrayCentroids(centroids))
        centroids = GetCentroids(cluster_output)
        if has_converged(
            ArrayCentroids(old_centroids), ArrayCentroids(centroids), threshold
        ):
            break
    return cluster_output, labels
