import dask.array as da
import numpy as np

HOMEDIR = "../dati/gmariani/miaxai"


def load_chunk(i: int):
    # Reads file
    filename = f"{HOMEDIR}/neighborhood/chunk_{i}.npz"
    loaded = np.load(filename)
    array = loaded["inverse"]
    return da.from_array(array)


def main():
    for i in range(62):
        print(i)
        # Reads file
        array = load_chunk(i)
        # Creates one Dask array for the chunk
        da.to_zarr(array, f"{HOMEDIR}/neighborhood/chunk_{i}.zarr", overwrite=True)
        print("Saved")


if __name__ == "__main__":
    main()
