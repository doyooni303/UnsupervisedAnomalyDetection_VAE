import json
import logging
import os
import shutil
import datetime
import random
import argparse

import numpy as np
import torch
from torch.optim import AdamW, SGD, Adam
from transformers import T5ForConditionalGeneration


class RunningAverage:
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def save_json(path, file_name, dictionary):
    """Saves dict of floats in json file

    Args:
        path: Folder name you wish to save in
        file_name: The name of file that will be saved as .json
        dictionary: Dictionary you want to save
    """

    PATH = os.path.join(path, file_name)
    if not os.path.exists(path):
        print("Directory does not exist! Making directory {}".format(path))
        os.mkdir(path)
    else:
        print("Directory exists! ")

    with open(PATH, "w", encoding="utf-8") as make_file:
        json.dump(dictionary, make_file, ensure_ascii=False, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, "last.pth.tar")
    if not os.path.exists(checkpoint):
        print(
            "Checkpoint Directory does not exist! Making directory {}".format(
                checkpoint
            )
        )
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer


def set_device(device):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    return device


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))

    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))


def save_models(model_state_dict, optimizer_state_dict, parameter_dir):
    model_path = os.path.join(parameter_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        },
        model_path,
    )
    print(f"Saved Path :{model_path}")


def set_main_parser():
    parser = argparse.ArgumentParser(description="VAE for Anomaly Detection")
    # Training args
    parser.add_argument(
        "--parameter_dir",
        type=str,
        default="experiment1",
        help="Directory containing params.json",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="Dacon Unsupervised Anomaly Detection",
        help="Name of Project in wandb",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Main device",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="../Data",
        help="Main device",
    )

    parser.add_argument(
        "--note",
        type=str,
        default="Simple notes for the experiment.",
        help="Simple notes for the experiment.",
    )

    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train (default: 1)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for anomaly score",
    )

    parser.add_argument(
        "--enc_hidden_dim",
        type=int,
        default=32,
        help="Random seed",
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=16,
        help="Random seed",
    )

    parser.add_argument(
        "--dec_hidden_dim",
        type=int,
        default=32,
        help="Random seed",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="Select optimizer. Options:[sgd,adamw] / Default: adamw",
    )

    parser.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="accumulation step size to enlarge the batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="learning rate (default: 3e-04)"
    )

    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum when Optimizer is SGD"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )

    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )

    parser.add_argument(
        "--save_best",
        type=str,
        default="True",
        help="Whether to save the best model or not",
    )

    return parser


def set_test_parser():
    parser = argparse.ArgumentParser(description="Test T5 model")
    # Training args
    parser.add_argument(
        "--parameter_dir",
        type=str,
        default="experiment1",
        help="Directory containing params.json",
    )

    parser.add_argument(
        "--note",
        type=str,
        default="Simple notes for the experiment.",
        help="Simple notes for the experiment.",
    )

    parser.add_argument(
        "--bertscore_device",
        type=str,
        default="cuda:1",
        help="Device for computing the Bertscore",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )

    return parser


def set_optimizer(name: str, model: torch.nn.Module, lr: float, momentum: float):
    if name == "sgd":
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    elif name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)

    elif name == "adam":
        optimizer = Adam(model.parameters(), lr=lr)

    return optimizer
