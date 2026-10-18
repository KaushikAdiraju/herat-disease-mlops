from src.preprocess import ClinicalFeatureEngineer, build_preprocessor  # noqa: E402
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "age": 63.0, "sex": 1, "cp": 1, "trestbps": 145.0, "chol": 233.0,
            "fbs": 1, "restecg": 2, "thalach": 150.0, "exang": 0, "oldpeak": 2.3,
            "slope": 3, "ca": 0.0, "thal": 6.0
        },
        {
            "age": 52.0, "sex": 0, "cp": 3, "trestbps": 130.0, "chol": 210.0,
            "fbs": 0, "restecg": 0, "thalach": 165.0, "exang": 0, "oldpeak": 0.5,
            "slope": 1, "ca": np.nan, "thal": np.nan  # Include missing values to test imputation
        }
    ])


def test_clinical_feature_engineer():
    """Verifies that custom domain features rpp_index and chol_age_ratio are computed properly."""
    df = get_sample_dataframe()
    engineer = ClinicalFeatureEngineer()
    transformed = engineer.fit_transform(df)

    assert "rpp_index" in transformed.columns
    assert "chol_age_ratio" in transformed.columns
    expected_rpp = (145.0 * 150.0) / 100.0
    assert np.isclose(transformed.loc[0, "rpp_index"], expected_rpp)


def test_preprocessing_pipeline_handles_nans():
    """Verifies that the full preprocessor imputes missing values and produces a valid matrix."""
    df = get_sample_dataframe()
    preprocessor = build_preprocessor()
    processed_matrix = preprocessor.fit_transform(df)

    assert not np.isnan(processed_matrix).any(), "Processed feature matrix should contain zero NaNs"
    assert processed_matrix.shape[0] == 2
