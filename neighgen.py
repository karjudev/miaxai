from math import ceil

from pandas import read_csv

from lime.lime_tabular import LimeTabularExplainer
import dask
import dask.array as da
from numpy import delete

HOMEDIR = "../dati/gmariani/miaxai"


dask.config.set({"temporary_directory": "~/dati/gmariani/tmp"})

print("Reading data")
df = read_csv("./data/geotarget_30.csv", index_col=0)
X = df.drop(columns=["profile", "UserID"]).values
print("Creating the explainer")
exp = LimeTabularExplainer(X)

chunk_size = 50
# Number of chunks
n_chunks: int = ceil(X.shape[0] / chunk_size)
print("Computing neighborhood")
for i in range(n_chunks):
    begin: int = i * chunk_size
    end: int = min(begin + chunk_size, X.shape[0])
    arr = []
    for j in range(begin, end):
        _, row = exp.data_inverse(X[j], num_samples=5000, sampling_method="gaussian")
        # Deletes the first row
        row = delete(row, (0), axis=0)
        arr.append(row)
    neigh = da.concatenate(arr)
    da.to_zarr(
        neigh, f"{HOMEDIR}/neighborhood/chunk_{i}.zarr", overwrite=True,
    )
    print(i)

print("End")
