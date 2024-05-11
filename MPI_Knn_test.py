from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from project_calcul.KNN import *
from tests.Generators import PointsGenerator

TypeResult = Tuple[Dict[TypeCoordinate, TypePointsWithDistances], float]

# init mpi
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Prepare data
if rank == 0:
    sequential_time = 0
    sequential_time_block = timeit.default_timer()
    X: List[float] = []
    banchmark: List[float] = []
    Speedup: List[float] = []
    # generate points
    Number = 10**4
    test_percent = 0.2
    points = PointsGenerator(3, Number, 0, 10)
    target = {
        c: v for v, c in Normalizer(points[int(np.round((-Number) * test_percent)) :])
    }
    print(f"test points: {len(target)}")
    new_points = list(target.keys())
    for i in range(1, size):
        comm.send(new_points, dest=i)
    sequential_time0 = timeit.default_timer() - sequential_time_block
else:
    new_points = comm.recv(source=0)

# Global variables
k = 3
for desired_size in range(1, size + 1):
    comm.barrier()
    result: TypeResult = ({}, 0)
    if rank < desired_size:
        if rank == 0:
            sequential_time = sequential_time0  # type: ignore
            sequential_time_block = timeit.default_timer()
            points_parts = np.array_split(points, desired_size)

            data = points_parts[0]
            for i in range(1, desired_size):
                comm.send(points_parts[i], dest=i)
            sequential_time += timeit.default_timer() - sequential_time_block  # type: ignore
        else:
            data = comm.recv(source=0)
        result = SlaveWork(
            Normalizer(data),
            new_points,
            k,
        )

    output: List[TypeResult] | None = comm.gather((result), root=0)
    if rank < desired_size and rank == 0 and output is not None:
        sequential_time_block = timeit.default_timer()
        ToPredict: Dict[TypeCoordinate, TypePointsWithDistances] = {
            c: [] for c in new_points
        }
        time_slaves: list[float] = []
        for dct, timing in output:
            time_slaves.append(timing)
            for c, lst in dct.items():
                ToPredict[c] += lst
                ToPredict[c] = TopKMin(ToPredict[c], k)

        prediction: Dict[TypeCoordinate, str] = {
            c: predict(lst) for c, lst in ToPredict.items()
        }
        time_perfomance = max(time_slaves)
        print(f"with {desired_size}  {time_perfomance}")

        accuracy_score = getattr(
            __import__("sklearn.metrics", fromlist=["accuracy_score"]), "accuracy_score"
        )
        LabelEncoder = getattr(
            __import__("sklearn.preprocessing", fromlist=["LabelEncoder"]),
            "LabelEncoder",
        )
        # from sklearn.metrics import accuracy_score
        # from sklearn.preprocessing import LabelEncoder

        classes = list(set([i[0] for i in points]))
        encoded_classes = LabelEncoder().fit_transform(classes)
        doc = {c: e for c, e in zip(classes, encoded_classes)}  # type: ignore
        print(
            f"accuracy score: {accuracy_score(list(doc[n] for n in target.values()),list(doc[n] for n in prediction.values()),)*100}%"
        )
        sequential_time += timeit.default_timer() - sequential_time_block  # type: ignore
        X.append(desired_size)
        banchmark.append(time_perfomance)
        real_sequential_time = sequential_time + time_perfomance
        if desired_size == 1:
            Speedup.append(real_sequential_time / real_sequential_time)
        else:
            Speedup.append(real_sequential_time / time_perfomance)
        print(
            f"{sequential_time = },{time_perfomance = },speedup: {sequential_time / time_perfomance}"
        )
        if desired_size == size:
            MPI.Finalize()
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
