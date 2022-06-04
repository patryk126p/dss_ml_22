"""
Download dataset
"""
import argparse

from utils import go

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download torchvision dataset")
    parser.add_argument(
        "dataset", type=str, help="Name of torchvision class for downloading dataset"
    )
    parser.add_argument("batch_size", type=int, help="Batch size for DataLoader")
    parser.add_argument("artifact_name", type=str, help="Dataset name")
    args = parser.parse_args()
    go(args)
