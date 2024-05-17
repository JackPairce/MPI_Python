import timeit
from typing import List, Tuple

import numpy as np

from project_calcul.ml_algorithms.Kmeans import *

accuracy_score = getattr(
    __import__("sklearn.metrics", fromlist=["accuracy_score"]),
    "accuracy_score",
)

train_test_split = getattr(
    __import__("sklearn.model_selection", fromlist=["train_test_split"]),
    "train_test_split",
)


def Kmeans_MPI(
    comm, rank: int, size: int, Data: np.ndarray, test_percent: float, k=3
) -> None:
    centroids: np.ndarray[List[float], np.dtype] = np.array([])
    if rank == 0:
        # inisitialize banchmark and speedup
        X: List[float] = []
        banchmark: List[float] = []
        Speedup: List[float] = []
        # inisitialize Sequential time
        sequential_time = 0

        # calculate Sequential time of this block
        sequential_time_block = timeit.default_timer()
        points = np.array(Data, dtype="f")

        centroids = GetRandomCentroids(points, k)  # type: ignore
        centroids = centroids[:, :-1]
        x = points[:, :-1]
        y = points[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=test_percent
        )

        # Send centroids to all processes
        for i in range(size):
            comm.send(centroids, i)
        sequential_time0 = timeit.default_timer() - sequential_time_block

    else:
        centroids = comm.recv(source=0)

    for desired_size in range(1, size + 1):
        comm.barrier()

        # start Kmeans algorithm
        label: np.ndarray[int, np.dtype] = np.array([])
        LoopList = [False for _ in range(size)]

        # Spliting data
        if rank < desired_size:
            if rank == 0:
                sequential_time = sequential_time0  # type: ignore
                Splited_X_train = np.array_split(X_train, desired_size)
                data = Splited_X_train[0]
                for i in range(1, desired_size):
                    comm.send(Splited_X_train[i], i)
                sequential_time = sequential_time0  # type: ignore

            else:
                data = comm.recv(source=0)

        parallel_time = timeit.default_timer()
        for _ in range(300):
            comm.barrier()
            sum_points, Nbpoints = np.array(
                [np.zeros(centroids.shape[1]) for _ in range(k)]
            ), np.zeros(k)

            if rank < desired_size:
                # Clustering step
                (sum_points, Nbpoints), label = clustering(data, centroids)  # type: ignore

            # Gathering Centroids and Nbpoints
            pre_new_centroids: List[Tuple[List[np.ndarray], List[int]]] | None = (
                comm.gather((sum_points, Nbpoints), root=0)
            )
            List_label = comm.gather(label, root=0)

            # Calculate new centroids
            if pre_new_centroids is not None:
                new_centroids = GetCentroids(pre_new_centroids)
                convergence = has_converged(centroids, new_centroids, 10**-7)
                LoopList = [convergence for _ in range(size)]
                centroids = new_centroids

            # Stop condition
            if comm.scatter(LoopList):
                break
        parallel_time = timeit.default_timer() - parallel_time

        # Final step
        if List_label is not None:
            # Append time to banchmark
            X.append(desired_size)
            banchmark.append(parallel_time)
            real_sequential_time = sequential_time + parallel_time
            Speedup.append(
                1 if desired_size == 1 else real_sequential_time / parallel_time
            )

            # Calculate accuracy
            label = np.concatenate(List_label)

            # Convert label to train label
            Dict_label = {}
            for l, yt in zip(label, y_train):
                Dict_label[yt] = l
                if len(Dict_label) == k:
                    break
            (sum_points, Nbpoints), predict_label = clustering(X_test, new_centroids)  # type: ignore
            new_centroids = GetCentroids([(sum_points, Nbpoints)])

            print(
                f"accuracy_score with {desired_size} processor: {"{:.2f}".format(accuracy_score([Dict_label[i] for i in y_test], predict_label)*100)}%"
            )

        # Calculate speedup
        if rank == 0 and desired_size == size:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 6))  # Set figure size

            # Subplot 1: Execution Time
            plt.subplot(2, 1, 1)
            plt.plot(X, banchmark, marker="o", c="orange", label="Execution Time")
            plt.xlabel("number of processes")
            plt.ylabel("time ratio [s]")
            plt.legend()

            # Subplot 2: Speedup
            plt.subplot(2, 1, 2)
            plt.plot(X, Speedup, marker="o", c="black", label="Speedup")
            plt.xlabel("number of processes")
            plt.ylabel("speedup")
            plt.legend()

            # Title
            plt.suptitle(f"MPI with Kmeans and {size} processes")

            # Show the plot
            plt.show()
