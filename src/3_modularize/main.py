import argparse
import os

import yaml
from utils import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "steps", type=str, help="Comma separated names of steps to run or all"
    )
    parser.add_argument("project_name", type=str, help="W&B project name")
    parser.add_argument("group_name", type=str, help="W&B group name")
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ["WANDB_RUN_GROUP"] = args.group_name

    with open("config.yaml", "r") as fh:
        conf = yaml.safe_load(fh)

    run_experiment(conf, args.steps)
