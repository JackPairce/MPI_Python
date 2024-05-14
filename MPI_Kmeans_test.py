import timeit
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.pylab import normal
from mpi4py import MPI

from project_calcul.Kmeans import ArrayCentroids  # type: ignore
from project_calcul.Kmeans import (
    ArrayNormalizer,
    GenDictForClusters,
    GetCentroids,
    GetRandomCentroids,
    PreCentroids,
    clustering,
    has_converged,
)
from project_calcul.tools import WordsEncoder
from tests.Generators import PointsGenerator

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

k = 3
centroids: np.ndarray[List[float], np.dtype] = np.array([])
LoopList = [False for _ in range(size)]
if rank == 0:
    sequential_time = 0
    sequential_time_block = timeit.default_timer()
    X: List[float] = []
    banchmark: List[float] = []
    Speedup: List[float] = []
    # generate points
    Number = 10**3
    test_percent = 0.2
    points = PointsGenerator(3, Number, 0, 10)
    colors = set(points[:, -1])

    points[:, -1] = WordsEncoder(points[:, -1]).EncodingWords()
    points = np.array(points, dtype="f")
    centroids = GetRandomCentroids(points, k)  # type: ignore
    y_centroids = centroids[:, -1]
    centroids = centroids[:, :-1]
    x = points[:, :-1]
    y = points[:, -1]
    train_test_split = getattr(
        __import__("sklearn.model_selection", fromlist=["train_test_split"]),
        "train_test_split",
    )
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_percent)
    for i in range(size):
        comm.send(centroids, i)
    # print(f"{points = }")
else:
    centroids = comm.recv(source=0)
for desired_size in range(1, size + 1):
    comm.barrier()
    # result: Dict[int, np.ndarray] = {}
    label: np.ndarray[int, np.dtype] = np.array([])
    for ill in range(300):
        comm.barrier()
        sum_points, Nbpoints = np.array([[0, 0] for _ in range(k)]), np.zeros(k)
        if rank < desired_size:
            if rank == 0:
                Splited_X_train = np.array_split(X_train, desired_size)
                data = Splited_X_train[0]
                for i in range(1, desired_size):
                    comm.send(Splited_X_train[i], i)
            else:
                data = comm.recv(source=0)
            (sum_points, Nbpoints), label = clustering(data, centroids)
            # print(f"[{desired_size}][{rank}]{result = }\n")
        pre_new_centroids: List[Tuple[List[np.ndarray], List[int]]] | None = (
            comm.gather((sum_points, Nbpoints), root=0)
        )
        List_label = comm.gather(label, root=0)
        if pre_new_centroids is not None:
            new_centroids = GetCentroids(pre_new_centroids)
            convergence = has_converged(centroids, new_centroids, 10**-7)
            LoopList = [convergence for _ in range(size)]
            centroids = new_centroids
        if comm.scatter(LoopList):
            break
    if List_label is not None:
        label = np.concatenate(List_label)
        Dict_label = {}
        for l, yt in zip(label, y_train):
            Dict_label[yt] = l
            if len(Dict_label) == k:
                break
        (sum_points, Nbpoints), predict_label = clustering(X_test, new_centroids)
        new_centroids = GetCentroids([(sum_points, Nbpoints)])
        from sklearn.metrics import accuracy_score

        print(
            f"accuracy_score with {desired_size}: {accuracy_score([Dict_label[i] for i in y_test], predict_label)*100}%"
        )

        # import matplotlib.pyplot as plt

        # plt.cla()
        # for dict_result in Final_result[:desired_size]:
        #     for i, v in dict_result.items():
        #         plt.scatter(v[:, 0], v[:, 1], color=list(colors)[i])

        # plt.show()

        # print(f"{accuracy_score([0,1,2], [1,0,2],normalize=False) = }")
