from typing import Final, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from project_calcul.ml_algorithms.KNN import *

accuracy_score = getattr(
    __import__("sklearn.metrics", fromlist=["accuracy_score"]), "accuracy_score"
)
LabelEncoder = getattr(
    __import__("sklearn.preprocessing", fromlist=["LabelEncoder"]),
    "LabelEncoder",
)

train_test_split = getattr(
    __import__("sklearn.model_selection", fromlist=["train_test_split"]),
    "train_test_split",
)


# Prepare data
def Knn_MPI(
    comm, rank: int, size: int, Data: np.ndarray, test_percent: float, k=3
) -> None:
    if rank == 0:
        sequential_time = 0
        sequential_time_block = timeit.default_timer()
        X: List[float] = []
        banchmark: List[float] = []
        Speedup: List[float] = []

        points = np.array(Data, dtype="f")

        x = points[:, :-1]
        y = points[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=test_percent
        )
        for i in range(1, size):
            comm.send(X_test, dest=i)
        sequential_time0 = timeit.default_timer() - sequential_time_block
    else:
        X_test = comm.recv(source=0)

    # Global variables
    k = 3
    for desired_size in range(1, size + 1):
        comm.barrier()

        # Start Knn algorithm
        result: Tuple[list[tuple[ndarray, ndarray]], float] = ([], 0.0)
        if rank < desired_size:
            # Spliting data
            if rank == 0:
                sequential_time = sequential_time0  # type: ignore
                sequential_time_block = timeit.default_timer()
                X_parts = np.array_split(X_train, desired_size)
                y_parts = np.array_split(y_train, desired_size)

                data = X_parts[0], y_parts[0]
                for i in range(1, desired_size):
                    comm.send((X_parts[i], y_parts[i]), dest=i)
                sequential_time += timeit.default_timer() - sequential_time_block  # type: ignore
            else:
                data = comm.recv(source=0)
                
            # Start Knn algorithm
            result = SlaveWork(
                data[0],
                data[1],
                X_test,
                k,
                rank
            )

        output: List[Tuple[List[Tuple[ndarray, ndarray]], float]] | None = comm.gather(
            result, root=0
        )
        if output is not None:
            sequential_time_block = timeit.default_timer()
            # Gathering results
            Final_Results: list[tuple[list[float], list[int]]] = [
                ([], []) for _ in range(len(y_test))
            ]
            time_slaves = []
            for result, time_slave in output:  # type: ignore
                time_slaves.append(time_slave)
                for i, tple in enumerate(result):
                    Final_Results[i][0].extend(tple[0])  # type: ignore
                    Final_Results[i][1].extend(tple[1])  # type: ignore
            label = [
                predict(TopKMin(np.array(d), np.array(l), k)[1])
                for d, l in Final_Results
            ]
            time_perfomance = max(time_slaves)
            print(
                f"accuracy score with {desired_size} processor: {"{:.2f}".format(accuracy_score(label,y_test)*100)}%"
            )
            sequential_time += timeit.default_timer() - sequential_time_block  # type: ignore

            # Append time to banchmark
            X.append(desired_size)
            banchmark.append(time_perfomance)
            real_sequential_time = sequential_time + time_perfomance
            Speedup.append(
                1 if desired_size == 1 else real_sequential_time / time_perfomance
            )
            if desired_size == size:
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
                plt.suptitle(f"MPI with KNN and {size} processes")

                # Show the plot
                plt.show()
