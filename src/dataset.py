from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
from sklearn.datasets import make_classification
import pandas as pd
import os

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def generate_data(
        n_samples: int = 10000,
        n_features: int = 5,
        n_informative: int = 2,
        n_redundant: int = 2,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    x, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_redundant=n_redundant,
                               random_state=random_state)
    return pd.DataFrame(x), pd.Series(y)


def save_data(x: pd.DataFrame, y: pd.Series, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = x.copy()
    data["target"] = y
    data.to_csv(path, index=False)
    logger.info(f"Data saved to {path}")


@app.command()
def main(
        output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
) -> None:
    logger.info("Processing dataset...")
    x, y = generate_data()
    save_data(x, y, str(output_path))
    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
