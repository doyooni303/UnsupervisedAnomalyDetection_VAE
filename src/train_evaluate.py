import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
from collections import defaultdict
from tqdm import tqdm
import warnings
import pdb

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from utils import RunningAverage, set_logger, save_models


def VAE_loss(x, x_hat, mean, log_var):
    mse = nn.MSELoss(reduction="mean")
    reproduction_loss = mse(x_hat, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def train_epoch(args, model, device, dataloader, optimizer, accumulation_steps=1):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    train_loss = RunningAverage()
    model.train()

    for idx, batch in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Train Epoch"
    ):
        batch["inputs"] = batch["inputs"].to(device)

        # optimizer condition
        opt_cond = (idx + 1) % accumulation_steps == 0

        optimizer.zero_grad()

        x_hat, mean, log_var = model(batch["inputs"])
        loss = VAE_loss(batch["inputs"], x_hat, mean, log_var)

        loss.backward()

        if opt_cond:
            # loss update
            optimizer.step()
            optimizer.zero_grad()

            train_loss.update(loss.item() * accumulation_steps)

    result = {
        "epoch_loss": round(train_loss(), 4),
    }
    return result


def validate_epoch(args, model, device, dataloader, labels, threshold=0.5):
    valid_loss = RunningAverage()
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Valid Epoch"):

            x_hat, mean, log_var = model(batch["inputs"].to(device))
            loss = VAE_loss(batch["inputs"].to(device), x_hat, mean, log_var)

            x = batch["inputs"].cpu().numpy()
            x_hat = x_hat.cpu().numpy()
            cos_list = [
                cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))[0][0]
                for i, j in zip(x, x_hat)
            ]

            preds = [1 if l < threshold else 0 for l in cos_list]
            predictions.extend(preds)
            valid_loss.update(loss.item())

    valid_f1 = f1_score(labels, predictions)
    torch.cuda.empty_cache()

    result = {
        "epoch_f1": round(valid_f1, 4),
        "epoch_loss": round(valid_loss(), 4),
        "predictions": predictions,
    }
    return result


def predict(args, model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Make a prediction"):
            x_hat, mean, log_var = model(batch["inputs"].to(args.device))
            x = batch["inputs"].cpu().numpy()
            x_hat = x_hat.cpu().numpy()
            cos_list = [
                cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))[0][0]
                for i, j in zip(x, x_hat)
            ]
            preds = [1 if l < args.threshold else 0 for l in cos_list]
            predictions.extend(preds)
    torch.cuda.empty_cache()

    return predictions


def fit(args, model, optimizer, train_loader, valid_loader, labels):
    set_logger(args.parameter_dir)
    best_f1 = 0

    TRAIN_RESULT, VALID_RESULT = defaultdict(list), defaultdict(list)

    logging.info("<< Training / Validate Start >>")
    for epoch in tqdm(range(args.epochs), desc="Ftitting the Model"):
        train_result = train_epoch(
            args, model, args.device, train_loader, optimizer, args.accum_steps
        )
        valid_result = validate_epoch(
            args, model, args.device, valid_loader, labels, args.threshold
        )

        TRAIN_RESULT["epoch_loss"].append(train_result["epoch_loss"])
        for key, value in valid_result.items():
            VALID_RESULT[key].append(value)

        if valid_result["epoch_f1"] >= best_f1:
            best_f1 = valid_result["epoch_f1"]
            best_loss = valid_result["epoch_loss"]
            best_predictions = valid_result["predictions"]

            if args.save_best:
                save_models(
                    model.state_dict(), optimizer.state_dict(), args.parameter_dir
                )

        logging.info(
            f"Epoch: {epoch+1} || Train loss:{train_result['epoch_loss']} || Valid F1:{valid_result['epoch_f1']} || Valid loss:{valid_result['epoch_loss']}"
        )
