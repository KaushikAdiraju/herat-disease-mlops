import os
import logging
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DATA_PATH = os.path.join("data", "raw", "heart_disease_raw.csv")
PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"

# Feature groupings
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "rpp_index", "chol_age_ratio"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that imputes base columns if needed and creates domain-specific
    cardiovascular features:
      - rpp_index: (trestbps * thalach) / 100.0 (Rate Pressure Product proxy)
      - chol_age_ratio: chol / age
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # Ensure numeric types in case raw JSON/CSV had strings or NaNs
        for col in ["age", "trestbps", "chol", "thalach"]:
            X_out[col] = pd.to_numeric(X_out[col], errors="coerce")

        X_out["rpp_index"] = (X_out["trestbps"] * X_out["thalach"]) / 100.0
        X_out["chol_age_ratio"] = X_out["chol"] / (X_out["age"] + 1e-5)
        return X_out


def build_preprocessor() -> Pipeline:
    """Builds a complete end-to-end preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, CONTINUOUS_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES)
        ]
    )

    full_preprocessing_pipeline = Pipeline(steps=[
        ("feature_engineer", ClinicalFeatureEngineer()),
        ("column_transformer", column_transformer)
    ])

    return full_preprocessing_pipeline


def preprocess_and_save(
    raw_path: str = RAW_DATA_PATH,
    processed_dir: str = PROCESSED_DIR,
    model_dir: str = MODEL_DIR,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Loads raw data, splits into train/test, fits the preprocessing pipeline on train,
    saves cleaned datasets and the fitted preprocessor artifact.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(raw_path)

    # Save a cleaned version of the full dataset (missing values imputed) as required by deliverables
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    cleaned_csv_path = os.path.join(processed_dir, "heart_disease_cleaned.csv")
    df_cleaned.to_csv(cleaned_csv_path, index=False)
    logging.info(f"Saved cleaned full dataset to {cleaned_csv_path}")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save raw splits (for end-to-end pipeline training) and processed splits
    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    # Save fitted preprocessor artifact for reproducibility
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logging.info(f"Preprocessor saved to {preprocessor_path}")
    logging.info(f"Processed Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    preprocess_and_save()
