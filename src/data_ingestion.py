import os
import logging
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_DATA_PATH = os.path.join("data", "raw", "heart_disease_raw.csv")


def download_and_save_data(output_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Fetches the Heart Disease dataset from the UCI ML Repository,
    applies initial target binarization, and saves the raw CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        logging.info("Fetching Heart Disease UCI dataset (ID: 45) via ucimlrepo...")
        heart_disease = fetch_ucirepo(id=45)
        X = heart_disease.data.features
        y = heart_disease.data.targets

        df = pd.concat([X, y], axis=1)

        # The raw target column is 'num' (0 = no disease, 1-4 = presence of disease)
        # Convert to strict binary target as required by the problem statement
        if "num" in df.columns:
            df["target"] = (df["num"] > 0).astype(int)
            df.drop(columns=["num"], inplace=True)

    except Exception as e:
        logging.warning(f"ucimlrepo fetch failed ({e}). Falling back to direct UCI URL...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
        ]
        df = pd.read_csv(url, names=columns, na_values="?")
        df["target"] = (df["num"] > 0).astype(int)
        df.drop(columns=["num"], inplace=True)

    df.to_csv(output_path, index=False)
    logging.info(f"Dataset successfully saved to {output_path} with shape {df.shape}")
    logging.info(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


if __name__ == "__main__":
    download_and_save_data()
