import os
import logging
import time

from tabulate import tabulate

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from model import VAE
from build_datset import load_data, TabularDataset
from train_evaluate import fit, predict
from utils import (
    set_optimizer,
    set_main_parser,
    makedir,
    set_logger,
    save_json,
    set_device,
)


def main():
    parser = set_main_parser()
    args = parser.parse_args()
    args.save_best = args.save_best == True
    makedir(args.parameter_dir)
    params = {}
    for key, value in args._get_kwargs():
        params[key] = value

    set_logger(os.path.join(args.parameter_dir, "train.log"))
    start = time.time()
    logging.info(f"Saving Directory:{args.parameter_dir}")
    save_json(args.parameter_dir, "basic_arguments.json", params)

    # Use GPU if it is available
    device = set_device(args.device)
    logging.info(f"Device is on {device}")

    # build_dataset & data loader
    x_train, x_valid, y_valid, x_test = load_data(args.data_path)
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    input_dim = x_train.shape[1]

    trainset = TabularDataset(x_train, normalize=True, mean=mean, std=std)
    validset = TabularDataset(x_valid, normalize=True, mean=mean, std=std)
    testset = TabularDataset(x_test, normalize=True, mean=mean, std=std)

    train_loader = DataLoader(
        trainset,
        args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        validset,
        args.val_batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        testset,
        args.val_batch_size,
        shuffle=False,
    )

    # set model, optimizer, criterion
    model = VAE(input_dim, args.enc_hidden_dim, args.latent_dim, args.dec_hidden_dim)
    model.to(args.device)

    optimizer = set_optimizer(args.optimizer, model, args.lr, args.momentum)

    # Start fitting the model
    fit(args, model, optimizer, train_loader, valid_loader, y_valid)

    # predictions
    predictions = predict(args, model, test_loader)
    test_data = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    ID = test_data["ID"].values

    pd.DataFrame({"ID": ID, "Class": predictions}).to_csv(
        f"submission/{args.parameter_dir}.csv"
    )
    logging.info(f"All Finished")


if __name__ == "__main__":
    main()
