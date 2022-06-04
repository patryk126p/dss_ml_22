import argparse
import logging
import os
import pickle
from typing import Tuple

import torchvision
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms as T
from wandb.sdk.wandb_run import Run

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args: argparse.Namespace) -> None:

    run = wandb.init(job_type="download", save_code=True)
    run.config.update(args)

    logger.info("Loading data")
    data = load_data(args.dataset, args.batch_size)

    file_path = os.path.join("data", args.artifact_name)
    with open(file_path, "wb") as fh:
        pickle.dump(data, fh)

    logger.info(f"Uploading {args.artifact_name} to artifact store")
    save_artifact(
        args.artifact_name,
        "dataset",
        "pickled dataset",
        file_path,
        run,
    )
    logger.info("Get data finished")


def load_data(dataset: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    logger.info(f"Loading {dataset}")
    dataset_class = getattr(torchvision.datasets, dataset)
    transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_set = dataset_class(
        root="data", train=True, download=True, transform=transforms
    )
    test_set = dataset_class(
        root="data", train=False, download=True, transform=transforms
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def save_artifact(name: str, type_: str, description: str, path: str, run: Run) -> None:
    artifact = wandb.Artifact(
        name,
        type=type_,
        description=description,
    )
    artifact.add_file(path)
    run.log_artifact(artifact)
    artifact.wait()
