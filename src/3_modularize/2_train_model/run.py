"""
Train DL model
"""
import argparse
import json
import logging
import os
import pickle

import torch
import wandb
from torch import nn, optim

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
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

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(
                model_config["conv1"]["in_channels"],
                model_config["conv1"]["out_channels"],
                kernel_size=model_config["conv1"]["kernel_size"],
                padding=model_config["conv1"]["padding"],
            )
            self.act1 = nn.Tanh()
            self.pool1 = nn.MaxPool2d(model_config["max_pool"])
            self.conv2 = nn.Conv2d(
                model_config["conv2"]["in_channels"],
                model_config["conv2"]["out_channels"],
                kernel_size=model_config["conv2"]["kernel_size"],
                padding=model_config["conv2"]["padding"],
            )
            self.act2 = nn.Tanh()
            self.pool2 = nn.MaxPool2d(model_config["max_pool"])
            self.fc = nn.Linear(
                model_config["fc"]["in_features"], model_config["fc"]["out_features"]
            )

        def forward(self, x):
            out = self.pool1(self.act1(self.conv1(x)))
            out = self.pool2(self.act2(self.conv2(out)))
            out = out.reshape(-1, model_config["fc"]["in_features"])
            out = self.fc(out)
            return out

    learning_rate = model_config["learning_rate"]
    n_epochs = model_config["epochs"]

    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    run.watch(model, criterion, log="all")
    logger.info("Training model")
    training_loop(n_epochs, optimizer, model, criterion, train_loader, run)

    model_path = os.path.join("model", args.model_name)
    logger.info(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Uploading {args.model_name} to artifact store")
    artifact = wandb.Artifact(
        args.model_name,
        type="model",
        description="DL model",
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    artifact.wait()

    logger.info("Calculating test accuracy")
    acc = get_accuracy(model, test_loader)
    logger.info(f"Accuracy on test data: {acc:.2f}")
    run.summary["test_accuracy"] = acc


def training_loop(n_epochs, optimizer, model, criterion, train_loader, run=None):
    for epoch in range(n_epochs):
        losses = []

        for imgs, labels in train_loader:
            outputs = model(imgs)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train DL model")
    parser.add_argument("train_data", type=str, help="Dataset artifact")
    parser.add_argument(
        "model_config",
        type=str,
        help="Path to json with model configuration",
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of model artifact",
    )
    args = parser.parse_args()

    go(args)
