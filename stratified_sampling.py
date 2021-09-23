import dask.array as da
from numpy import savez_compressed

from sklearn.model_selection import train_test_split

from generate_predictions import load_chunk, HOMEDIR


def main():
    print("Loading data")
    X = da.concatenate([load_chunk(i) for i in range(62)])
    print("Sampling decision tree predictions")
    y_pred_dt = da.from_zarr(f"{HOMEDIR}/black_box/predictions/y_pred_dt.zarr")
    y_prob_dt = da.from_zarr(f"{HOMEDIR}/black_box/predictions/y_prob_dt.zarr")
    X_sample, _, y_sample, _, y_prob, _ = train_test_split(
        X, y_pred_dt, y_prob_dt, train_size=15000, stratify=y_pred_dt, random_state=42
    )
    savez_compressed(f"{HOMEDIR}/shadow/rf_data", X=X_sample, y=y_sample, y_prob=y_prob)
    print("Sampling random forest predictions")
    y_pred_rf = da.from_zarr(f"{HOMEDIR}/black_box/predictions/y_pred_rf.zarr")
    y_prob_rf = da.from_zarr(f"{HOMEDIR}/black_box/predictions/y_prob_rf.zarr")
    X_sample, _, y_sample, _, y_prob, _ = train_test_split(
        X, y_pred_rf, y_prob_rf, train_size=15000, stratify=y_pred_rf, random_state=42
    )
    savez_compressed(f"{HOMEDIR}/shadow/dt_data", X=X_sample, y=y_sample, y_prob=y_prob)
    print("Completed")


if __name__ == "__main__":
    main()
