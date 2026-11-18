import os

# Match to your physical CPU cores before importing numpy/torch
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np
import torch
from tabicl import TabICLRegressor


class FoundationTabularModel:
    def __init__(
        self,
        n_estimators: int = 2,
        batch_size: int = 4,
        random_state: int = 42,
    ):
        self.model = None
        self.n_estimators = n_estimators
        self.batch_size = batch_size
        self.random_state = random_state

    def train(self, X_train, y_train):
        X_fit = X_train.astype(np.float32)
        y_fit = np.asarray(y_train, dtype=np.float32).ravel()

        self.model = TabICLRegressor(
            n_estimators=self.n_estimators,
            batch_size=self.batch_size,
            device="cpu",
            use_amp=False,
            kv_cache=False,
        )
        self.model.fit(X_fit, y_fit)
        return self

    def infer(self, X_test):
        X_ordered = X_test[self.model.feature_names_in_].astype(np.float32)
        n_samples = len(X_ordered)
        preds = np.empty(n_samples, dtype=np.float32)

        # bfloat16 autocast cuts RAM in half and speeds up CPU matrix multiplications
        with torch.inference_mode(), torch.amp.autocast("cpu", dtype=torch.bfloat16):
            preds = self.model.predict(X_ordered)

        return preds
