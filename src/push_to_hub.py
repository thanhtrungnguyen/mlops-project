import json
import os
import sys
from pathlib import Path

import typer
from huggingface_hub import HfApi
from loguru import logger

from src.config import MODELS_DIR, PROJ_ROOT

app = typer.Typer()


def push_to_hub(model_path: str, repo_id: str, token: str = None):
    """
    Push a trained model to Hugging Face Hub.

    Args:
        model_path: Path to the saved MLflow model
        repo_id: Hugging Face Hub repository ID (username/repo-name)
        readme_path: Path to README.md file
        token: Hugging Face token (optional)
    """
    try:
        # Create a minimal config.json to let the Hugging Face Hub recognize your model as a transformer model.
        # Adjust these settings as needed for your transformer.
        config = {
            "architectures": ["MyDummyTransformer"],
            "model_type": "bert",  # Even if your model isn’t BERT, this helps signal transformer compatibility.
            "tokenizer_class": "BertTokenizer",
            "version": "1.0",
            "name_or_path": "trungngnthanh/mlops-project"
        }
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Written config.json to {config_path}")

        # Create a simple model card (README.md) as a model description.
        readme_content = """---
library: scikit-learn
tags:
  - classification
  - logistic-regression
  - mlflow
  - sklearn
license: mit
datasets:
  - synthetic
metrics:
  - accuracy
---

# My Dummy Transformer Model

This is a dummy transformer model for demonstration purposes. 
It is intended to show how including YAML metadata in your model card can resolve the warning
and trigger the "Use this transformer" label on the Hugging Face Hub.
"""
        readme_path = os.path.join(model_path, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)
        logger.info(f"Written README.md to {readme_path}")

        # ---- Hugging Face Upload Section ----

        # Read the Hugging Face token from the environment variable 'HF_TOKEN'
        if not token:
            raise EnvironmentError("Ensure the environment variable HF_TOKEN is set.")

        # Use HfApi to upload the folder containing the model and metadata files
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
        )
        logger.success(f"Model successfully pushed to: https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to push to Hugging Face Hub: {str(e)}")
        return False


@app.command()
def main(
        model_path: Path = MODELS_DIR / "best_model",
        repo_id: str = "trungngnthanh/mlops-project",
        token: str = None,
):
    """
    Push a trained model to Hugging Face Hub.
    """
    if not repo_id:
        logger.error("Please provide a repository ID (--repo-id username/model-name)")
        sys.exit(1)

    success = push_to_hub(
        model_path=str(model_path),
        repo_id=repo_id,
        token=token
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    app()
