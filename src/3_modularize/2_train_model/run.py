"""
Train DL model
"""
import argparse
import logging

from utils.functions import go

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


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
