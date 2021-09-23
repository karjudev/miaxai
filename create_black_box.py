import bz2
import pickle
from typing import Dict
from numpy import ndarray, savez_compressed
from pandas import read_csv
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def create_model_grid_search(
    model, X_train: ndarray, y_train: ndarray, hyperparameters: Dict,
):
    """Creates a Random Forest using a grid search.

    Args:
        model: SciKit-Learn machine learning model
        X_train (ndarray): Training set
        y_train (ndarray): Training label
        hyperparameters (Dict): Hyperparameter options
    """
    # Grid search
    clf = GridSearchCV(model, hyperparameters, n_jobs=-1, refit=True)
    clf.fit(X_train, y_train)
    # Model trained with the best found combination of parameters
    best_model = clf.best_estimator_
    return best_model


def create_report(model, X_test: ndarray, y_test: ndarray) -> str:
    """Creates a classification report for the model.

    Args:
        model: SciKit-Learn machine learning model
        X_test (ndarray): Test set
        y_test (ndarray): Test label

    Returns:
        str: Classification report
    """
    y_pred: ndarray = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report


HOMEDIR = "../dati/gmariani/miaxai"

# Loads the DataFrame
print("Reading DataFrame")
df = read_csv("./data/geotarget_30.csv", index_col=0)
df["profile"] = df["profile"].astype("category")
# Splits data and label
print("Getting data")
X = df.drop(columns=["profile", "UserID"])
y = df["profile"]
# Splits data in stratified train and test
print("Splitting data")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Saving data")
savez_compressed(
    "./data/bb_data", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)
# Hyperparameter combinations to use for the decision tree
dt_hyperparameters = {
    "criterion": ["gini", "entropy"],
    "max_depth": [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],
    "random_state": [42],
}
# Creates a random forest using given values from previous experiments
print("Performing grid search to create the best black boxes")
bb_rf = RandomForestClassifier(
    n_estimators=320,
    min_samples_split=5,
    max_depth=64,
    max_features="sqrt",
    class_weight="balanced",
    bootstrap=True,
)
bb_rf.fit(X_train, y_train)
print("Random forest created")
bb_rf_report = create_report(bb_rf, X_test, y_test)
with bz2.open(f"{HOMEDIR}/black_box/rf.pkl.bz2", "wb") as f:
    pickle.dump(bb_rf, f)
with open(f"{HOMEDIR}/black_box/rf_report.txt", "w") as f:
    f.write(bb_rf_report)
bb_dt = DecisionTreeClassifier()
bb_dt = create_model_grid_search(bb_dt, X_train, y_train, dt_hyperparameters)
print("Decision tree created")
bb_dt_report = create_report(bb_dt, X_test, y_test)
with bz2.open(f"{HOMEDIR}/black_box/dt.pkl.bz2", "wb") as f:
    pickle.dump(bb_dt, f)
with open(f"{HOMEDIR}/black_box/dt_report.txt", "w") as f:
    f.write(bb_dt_report)
