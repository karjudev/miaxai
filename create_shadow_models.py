import bz2
import pickle
import os
from math import ceil
from multiprocessing import Process
from pathlib import Path

from numpy import load, savez_compressed, ndarray
from pandas import DataFrame, concat

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

HOMEDIR = "../dati/gmariani/miaxai"
# HOMEDIR = "./data"

N_MODELS = 16

N_WORKERS = 16

TEST_SIZE = 0.2

BLACK_BOX_TYPE = "nn"

MODEL_TYPE = "rf"


def create_random_forest(X_train: ndarray, y_train: ndarray):
    rf = RandomForestClassifier()
    hyperparameters = {
        "bootstrap": [True, False],
        "max_depth": [100, 350, 500],
        "max_features": [5, "auto", "sqrt"],
        "min_samples_leaf": [10, 20, 50],
        "min_samples_split": [5, 10, 50],
        "n_estimators": [100, 350, 500],
        "criterion": ["gini", "entropy"],
    }
    clf = GridSearchCV(rf, hyperparameters, refit=True)
    clf.fit(X_train, y_train)
    return clf.best_estimator_


def worker(id: int, X: ndarray, y: ndarray, begin: int, end: int):
    print(f"Starting worker {id}. from {begin} to {end}")
    for i in range(begin, end):
        # Creates the working data
        Path(f"{HOMEDIR}/shadow/{BLACK_BOX_TYPE}_{MODEL_TYPE}_{i}").mkdir(
            exist_ok=True, parents=True
        )
        # Splits train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y
        )
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        # Saves data on disk
        savez_compressed(
            f"{HOMEDIR}/shadow/{BLACK_BOX_TYPE}_{MODEL_TYPE}_{i}/data",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        # Random forest model
        rf = create_random_forest(X_train, y_train)
        print(f"Process {id} - {i} - Model created")
        # Saves model to disk
        with bz2.open(
            f"{HOMEDIR}/shadow/{BLACK_BOX_TYPE}_{MODEL_TYPE}_{i}/model.pkl.bz2", "wb"
        ) as f:
            pickle.dump(rf, f)
        # Prediction probability on the training set
        y_prob_train = rf.predict_proba(X_train)
        # Creates a classification report
        train_report = classification_report(y_train, rf.predict(X_train))
        with open(
            f"{HOMEDIR}/shadow/{BLACK_BOX_TYPE}_{MODEL_TYPE}_{i}/train_report.txt", "w"
        ) as f:
            f.write(train_report)
        # "IN" dataset
        df_in = DataFrame(y_prob_train)
        df_in["class_label"] = y_train
        df_in["target_label"] = "in"
        # Prediction probability on the test set
        y_prob_test = rf.predict_proba(X_test)
        test_report = classification_report(y_test, rf.predict(X_test))
        with open(
            f"{HOMEDIR}/shadow/{BLACK_BOX_TYPE}_{MODEL_TYPE}_{i}/test_report.txt", "w"
        ) as f:
            f.write(test_report)
        df_out = DataFrame(y_prob_test)
        df_out["class_label"] = y_test
        df_out["target_label"] = "out"
        # Concats the two dataframes
        df = concat([df_in, df_out])
        df["target_label"] = df["target_label"].astype("category")
        df.to_pickle(
            f"{HOMEDIR}/shadow/{BLACK_BOX_TYPE}_{MODEL_TYPE}_{i}/prediction_set.pkl.bz2"
        )

    print(f"Ending worker {id}")


def main():
    # Loads dataset
    loaded = load(f"{HOMEDIR}/shadow/{BLACK_BOX_TYPE}_data.npz")
    # Input data
    X = loaded["X"]
    # Label
    y = loaded["y"]
    print("Data correctly read")
    # Size of the chunk of models for each worker
    chunk_size: int = ceil(N_MODELS / N_WORKERS)
    # List of worker processes
    processes = []
    # Parallelizes shadow model training for each worker
    for i in range(N_WORKERS):
        begin: int = i * chunk_size
        end: int = min(begin + chunk_size, N_WORKERS)
        process = Process(target=worker, args=(i, X, y, begin, end))
        processes.append(process)
        process.start()
    # Joins the parallel processes
    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
