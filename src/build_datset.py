import os, sys
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

import torch
import torch.nn as nn
from torch.utils.data import Dataset


def load_data(data_path):
    name = ["train", "test"]
    columns = [f"V{i}" for i in range(1, 31)]
    val_columns = columns + ["Class"]
    data_dict = {
        key: pd.read_csv(os.path.join(data_path, f"{key}.csv"))[columns] for key in name
    }
    data_dict["valid"] = pd.read_csv(os.path.join(data_path, "valid.csv"))[val_columns]

    # scale 해주면 array 형태가 됨
    data_dict["train"] = scale(data_dict["train"])
    x_train = data_dict["train"]
    x_valid = data_dict["valid"][columns].values
    y_valid = data_dict["valid"].Class.values

    test_data = pd.read_csv("data/test.csv")
    x_test = test_data.iloc[:, 1:]

    return x_train, x_valid, y_valid, x_test


class TabularDataset(Dataset):
    def __init__(
        self,
        inputs: np.array = None,
        normalize=True,
        mean=None,
        std=None,
    ):

        self.inputs = inputs

        if mean is not None:
            self.mean = mean

        if std is not None:
            self.std = std

        self.normalize = normalize

        if self.normalize:
            self.scaled_inputs = self.get_normalize(self.inputs, self.mean, self.std)

    def get_normalize(self, x: np.array, mean, std):
        scaled_x = (x - mean) / std
        return scaled_x

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.normalize:
            x_torch = torch.tensor(self.scaled_inputs, dtype=torch.float)[idx]
        else:
            x_torch = torch.tensor(self.inputs, dtype=torch.float)[idx]

        return {"inputs": x_torch}
