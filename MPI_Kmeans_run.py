import numpy as np
import pandas as pd
from mpi4py import MPI

from project_calcul.ml_algorithms.tools import WordsEncoder
from project_calcul.mpi_ml.Kmeans import Kmeans_MPI
from tests.Generators import PointsGenerator

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Data = np.array([])
k = 2
test_percent = 0.5
if rank == 0:
    # Using Points Generator
    # Number = 10**4
    # Data = PointsGenerator(3, 10**6, 0, 10)
    # Data[:, -1] = WordsEncoder(Data[:, -1]).EncodingWords()

    # Using Datasets
    df = pd.read_csv("assets/KDDTrains.csv")
    cols = df.columns.to_numpy()[[1, 2, 3, -1]]
    for c in cols:
        df[c] = WordsEncoder(df[c]).EncodingWords()
    Data = df.to_numpy()

    print("Starting...")

Kmeans_MPI(comm, rank, size, Data, test_percent, k)
MPI.Finalize()
