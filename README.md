# Data Science Summit ML Edition 2022

## Easy ML experiment tracking with Weights & Biases and MLflow

### Setup

1. Create account at [Weights & Biases](https://wandb.ai/site)
2. Install requirements `pip install -r requirements.txt`
3. Login to Weights & Biases from terminal `wandb login`

### How to use this repo

To run experiments with the default settings from the repo root execute:
- 1_start_simple: `mlflow run -e run_1 --env-manager local .`
- 2_add_tracking: `mlflow run -e run_2 --env-manager local .`
- 3_modularize: `mlflow run -e run_3 --env-manager local .`
