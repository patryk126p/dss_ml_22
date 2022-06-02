import argparse
import os

from utils.functions import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("batch_size", type=int, help="Batch size for DataLoader")
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
    parser.add_argument("project_name", type=str, help="W&B project name")
    parser.add_argument("group_name", type=str, help="W&B group name")
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ["WANDB_RUN_GROUP"] = args.group_name

    run_experiment(args)
