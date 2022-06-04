import argparse
import json
import logging
import os
import pickle
from typing import Optional

import torch
import wandb
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from utils.model import ConvNet
from wandb.sdk.wandb_run import Run

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args: argparse.Namespace) -> None:

    run = wandb.init(job_type="train", save_code=True)
    run.config.update(args)

    logger.info(f"Reading train data: {args.train_data}")
    local_path = wandb.use_artifact(args.train_data).file()
    with open(local_path, "rb") as fh:
        train_loader, test_loader = pickle.load(fh)

    logger.info(f"Reading model config from {args.model_config}")
    with open(args.model_config) as fp:
        model_config = json.load(fp)
    run.config.update(model_config)

    learning_rate = model_config["learning_rate"]
    n_epochs = model_config["epochs"]

    model = ConvNet(model_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    run.watch(model, criterion, log="all")
    logger.info("Training model")
    training_loop(n_epochs, optimizer, model, criterion, train_loader, run)

    model_path = os.path.join("model", args.model_name)
    logger.info(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Uploading {args.model_name} to artifact store")
    save_artifact(args.model_name, "model", "DL model", model_path, run)

    logger.info("Calculating test accuracy")
    acc = get_accuracy(model, test_loader)
    logger.info(f"Accuracy on test data: {acc:.2f}")
    run.summary["test_accuracy"] = acc


def training_loop(
    n_epochs: int,
    optimizer: optim.Optimizer,
    model: nn.Module,
    criterion: _Loss,
    train_loader: DataLoader,
    run: Optional[Run] = None,
) -> None:
    for epoch in range(n_epochs):
        losses = []

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"Epoch: {epoch}; loss: {sum(losses) / len(losses)}")
        acc = get_accuracy(model, train_loader)
        logger.info(f"Accuracy: {acc:.2f}")
        if run:
            run.log(
                {"epoch": epoch, "loss": sum(losses) / len(losses), "accuracy": acc}
            )


def save_artifact(name: str, type_: str, description: str, path: str, run: Run) -> None:
    artifact = wandb.Artifact(
        name,
        type=type_,
        description=description,
    )
    artifact.add_file(path)
    run.log_artifact(artifact)
    artifact.wait()


def get_accuracy(model, data_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for xs, ys in data_loader:
            scores = model(xs)
            _, predictions = torch.max(scores, dim=1)
            correct += (predictions == ys).sum()
            total += ys.shape[0]

        acc = float(correct) / float(total) * 100
    return acc
