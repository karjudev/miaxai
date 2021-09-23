import bz2
import pickle
import dask
import dask.array as da
from dask_ml.wrappers import ParallelPostFit

HOMEDIR = "../dati/gmariani/miaxai"

dask.config.set({"temporary_directory": "~/dati/gmariani/tmp"})


def load_chunk(i: int):
    # Filename of the chunk
    filename: str = f"{HOMEDIR}/neighborhood/chunk_{i}.zarr"
    # Loads the chunk
    chunk = da.from_zarr(filename)
    return chunk


def main():
    # Black box model
    print("Loading black box")
    bb_dt = None
    with bz2.open(f"{HOMEDIR}/black_box/dt.pkl.bz2", "rb") as f:
        bb_dt = ParallelPostFit(pickle.load(f), scoring="f1")
    bb_rf = None
    with bz2.open(f"{HOMEDIR}/black_box/rf.pkl.bz2", "rb") as f:
        bb_rf = ParallelPostFit(pickle.load(f), scoring="f1")
    # Loads input data
    print("Loading chunks")
    X_chunks = []
    for i in range(62):
        X_chunk = load_chunk(i)
        X_chunks.append(X_chunk)
        print(i)
    # Stacks the array to create a matrix
    X = da.concatenate(X_chunks)
    print("Chunks concatenated")
    # Generates the predictions
    print("Decision tree")
    y_pred_dt = bb_dt.predict(X).rechunk({0: "auto"})
    da.to_zarr(
        y_pred_dt, f"{HOMEDIR}/black_box/predictions/y_pred_dt.zarr", overwrite=True
    )
    y_prob_dt = bb_dt.predict_proba(X).rechunk({0: "auto", 1: "auto"})
    da.to_zarr(
        y_prob_dt, f"{HOMEDIR}/black_box/predictions/y_prob_dt.zarr", overwrite=True
    )
    print("Random Forest")
    y_pred_rf = bb_rf.predict(X).rechunk({0: "auto"})
    da.to_zarr(
        y_pred_rf, f"{HOMEDIR}/black_box/predictions/y_pred_rf.zarr", overwrite=True
    )
    y_prob_rf = bb_rf.predict_proba(X).rechunk({0: "auto", 1: "auto"})
    da.to_zarr(
        y_prob_rf, f"{HOMEDIR}/black_box/predictions/y_prob_rf.zarr", overwrite=True
    )
    print("End")


if __name__ == "__main__":
    main()
