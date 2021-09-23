from typing import Iterable, List
from pandas import read_pickle
from numpy import concatenate, load
from pandas import concat, DataFrame
from sklearn.metrics import classification_report
from create_shadow_models import HOMEDIR
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from classifier_wrapper import pytorch_classifier_wrapper

BLACK_BOX_TYPE = "rf"
SHADOW_TYPE = "rf"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ### FOR XAVIER INITIALIZATION
        self.fc1 = nn.Linear(236, 128)  # fc stays for 'fully connected'
        nn.init.xavier_normal_(self.fc1.weight)
        self.drop = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, 30)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = torch.tanh(self.fc1(x))
        x = self.fc4(self.drop(x))
        return nn.functional.softmax(x, dim=1)


def create_nn():
    print("Loading nn")
    model = Net()
    model.load_state_dict(torch.load(f"{HOMEDIR}/black_box/net.pt"))
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(torch.load(f"{HOMEDIR}/black_box/optimizer.pt"))
    loss_function = nn.CrossEntropyLoss()
    loss_function.load_state_dict(torch.load(f"{HOMEDIR}/black_box/loss_function.pt"))
    wrapped = pytorch_classifier_wrapper(
        classifier=model,
        optimizer=optimizer,
        loss_function=loss_function,
        classes=list(range(30)),
    )
    return wrapped


def create_rf():
    print("Loading rf")
    model = read_pickle(f"{HOMEDIR}/black_box/rf.pkl.bz2")
    return model


def create_attack_dataframe(X_prob, y, target_label):
    df = DataFrame(X_prob)
    df["class_label"] = y
    df["target_label"] = target_label
    return df


def create_inout_dataframe(X_train_prob, y_train, X_test_prob, y_test):
    df_in = create_attack_dataframe(X_train_prob, y_train, "in")
    df_out = create_attack_dataframe(X_test_prob, y_test, "out")
    df = concat([df_in, df_out])
    df["target_label"] = df["target_label"].astype("category")
    return df


def create_attack_dataset_artificial():
    df_list = []
    for label in range(30):
        loaded = load(
            f"{HOMEDIR}/attack/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_label_{label}/data.npz",
            allow_pickle=True,
        )
        df = DataFrame(loaded["X_test"])
        df["class_label"] = label
        df["target_label"] = loaded["y_test"]
        df_list.append(df)
    df = concat(df_list)
    return df


def create_attack_dataset_from_bb(model):
    loaded = load(f"{HOMEDIR}/black_box/bb_data.npz")
    X_train = loaded["X_train"]
    X_test = loaded["X_test"]
    y_train = loaded["y_train"]
    y_test = loaded["y_test"]
    X_train_prob = model.predict_proba(X_train)
    X_test_prob = model.predict_proba(X_test)
    df = create_inout_dataframe(X_train_prob, y_train, X_test_prob, y_test)
    return df


def test_all(data: DataFrame, attack_models: List) -> str:
    X_test = data.drop(columns=["class_label", "target_label"])
    y_test = data["target_label"]
    predictions = [model.predict(X_test) for model in attack_models]
    max_predictions = [max(p, key=p.count) for p in zip(*predictions)]
    report = classification_report(y_test, max_predictions)
    return report


def test_single(bb_data: DataFrame) -> Iterable:
    for label, label_df in bb_data.groupby("class_label", sort=False):
        print(f"Label = {label}")
        X = label_df.drop(columns=["class_label", "target_label"])
        y = label_df["target_label"]
        attack_model = read_pickle(
            f"{HOMEDIR}/attack/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_label_{label}/model.pkl.bz2"
        )
        y_pred = attack_model.predict(X)
        yield label, y, y_pred


def test_attack_model(dataset, name):
    print("Single")
    y_true_list = []
    y_pred_list = []
    for label, y_true_label, y_pred_label in test_single(dataset):
        report = classification_report(y_true_label, y_pred_label)
        with open(
            f"{HOMEDIR}/attack/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_label_{label}/test_single_{name}.txt",
            "w",
        ) as f:
            f.write(report)
        y_true_list.append(y_true_label)
        y_pred_list.append(y_pred_label)
    y_true = concatenate(y_true_list)
    y_pred = concatenate(y_pred_list)
    report = classification_report(y_true, y_pred)
    with open(
        f"{HOMEDIR}/attack/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_test_single_complete_{name}.txt",
        "w",
    ) as f:
        f.write(report)
    print("All")
    print("Loading attack models")
    attack_models = [
        read_pickle(
            f"{HOMEDIR}/attack/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_label_{label}/model.pkl.bz2"
        )
        for label in range(30)
    ]
    report = test_all(dataset, attack_models)
    with open(
        f"{HOMEDIR}/attack/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_test_all_{name}.txt", "w"
    ) as f:
        f.write(report)


def main():
    if BLACK_BOX_TYPE == "nn":
        model = create_nn()
    else:
        model = create_rf()
    print("Reconstructing artificial dataset")
    d_artificial = create_attack_dataset_artificial()
    print("Create attack model from black box")
    d_bb = create_attack_dataset_from_bb(model)
    print("Testing attack models on artificial data")
    test_attack_model(d_artificial, "lime")
    print("Testing attack models on black box data")
    test_attack_model(d_bb, "bb")


if __name__ == "__main__":
    main()
