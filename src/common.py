import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
TARGETS = ["NOX", "CO"]


def load_data():
    """Load the five yearly files, remove duplicates and missing rows."""
    frames = []
    for f in sorted(glob.glob("data/gt_*.csv")):
        d = pd.read_csv(f)
        d["year"] = int(f[-8:-4])
        frames.append(d)
    return pd.concat(frames, ignore_index=True).drop_duplicates().dropna()


def split_data(df, seed=42):
    """Random 80% train / 20% test split (same split for every model)."""
    return train_test_split(df, test_size=0.2, random_state=seed)


def score(y_true, y_pred):
    """The three metrics used to compare all models."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }