import mlflow

_steps = [
    "download",
    "train",
]


def run_experiment(config: dict, steps: str) -> None:

    # Steps to execute
    active_steps = steps.split(",") if steps != "all" else _steps

    if "download" in active_steps:
        _ = mlflow.run(
            "1_get_data",
            "main",
            parameters={
                "dataset": config["download"]["dataset"],
                "batch_size": config["download"]["batch_size"],
                "artifact_name": config["download"]["artifact_name"],
                "artifact_type": config["download"]["artifact_type"],
                "artifact_description": config["download"]["artifact_description"],
            },
        )

    if "train" in active_steps:
        _ = mlflow.run(
            "2_train_model",
            "main",
            parameters={
                "train_data": config["train"]["train_data"],
                "model_config": config["train"]["model_config"],
                "model_name": config["train"]["model_name"],
            },
        )
