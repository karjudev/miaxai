import dask.array as da
from dask_ml.wrappers import ParallelPostFit
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
from torch import optim
from classifier_wrapper import pytorch_classifier_wrapper

HOMEDIR = "../dati/gmariani/miaxai"


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


def main():
    print("Loading model")
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
    parallel_model = ParallelPostFit(estimator=wrapped, scoring="f1")
    print("Loading data")
    X = da.concatenate(
        [da.from_zarr(f"{HOMEDIR}/neighborhood/chunk_{i}.zarr") for i in range(62)]
    )
    print("Generating predictions")
    y_pred = parallel_model.predict(X).rechunk({0: "auto"})
    da.to_zarr(
        y_pred, f"{HOMEDIR}/black_box/predictions/y_pred_nn.zarr", overwrite=True
    )
    y_prob = parallel_model.predict_proba(X).rechunk({0: "auto", 1: "auto"})
    da.to_zarr(
        y_prob, f"{HOMEDIR}/black_box/predictions/y_prob_nn.zarr", overwrite=True
    )
    print("Stratified sampling")
    X_sample, _, y_sample, _, y_prob_sample, _ = train_test_split(
        X, y_pred, y_prob, train_size=15000, stratify=y_pred, random_state=42
    )
    print("Saving sample")
    np.savez_compressed(
        f"{HOMEDIR}/shadow/nn_data", X=X_sample, y=y_sample, y_prob=y_prob_sample
    )


if __name__ == "__main__":
    main()
