import timeit
from typing import Dict, List, Tuple

import numpy as np

dtype = np.dtype
ndarray = np.ndarray

TypeCoordinate = Tuple[float, float]
TypePoint = Tuple[str, TypeCoordinate]
TypePoints = List[TypePoint]
TypePointsWithDistances = List[Tuple[TypePoint, float]]


def Normalizer(
    data: ndarray[Tuple[str, Tuple[float, float]], dtype]
) -> List[TypePoint]:
    return [
        (i[0], (float(i[1][0]), float(i[1][1]))) for i in zip(data[:, 0], data[:, 1:])
    ]


#  KNN_Paralel:
def TopKMin(
    points_distance: TypePointsWithDistances, k: int
) -> TypePointsWithDistances:
    return sorted(points_distance, key=lambda x: x[-1])[:k]


def get_points_with_distances(
    points: TypePoints, new_point: TypeCoordinate
) -> TypePointsWithDistances:
    if points == []:
        return []
    return [
        p
        for p in zip(
            points,
            (
                # Calculate Euclidean distance
                np.sum(
                    (np.array([i[1] for i in points]) - np.array(new_point)) ** 2,
                    axis=1,
                )
                ** 0.5
            ),
        )
    ]


def predict(nearest_points: TypePointsWithDistances) -> str:
    label = [point[0][0] for point in nearest_points]
    return max(set(label), key=label.count)


def SlaveWork(
    points: TypePoints,
    new_points: List[TypeCoordinate],
    k: int,
) -> Tuple[Dict[TypeCoordinate, TypePointsWithDistances], float]:
    start = timeit.default_timer()
    result: Dict[TypeCoordinate, TypePointsWithDistances] = {}
    for np in new_points:
        result[np] = TopKMin(get_points_with_distances(points, np), k)
    end = timeit.default_timer()
    return (result, (end - start))
