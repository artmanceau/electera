# TODO (spatial boosting model) with a function .fit and .predict

import numpy as np
import pandas as pd


class ComposeDistanceMatrix:
    def __init__(self):
        self.distance_array = None
        self.i = None
        self.j = None
        self.N = None

    def _load_files(self):
        distance_npz = np.load(
            "../../data/raw/geo_data/distance_cities/distance_matrix.npz"
        )
        self.distance_array = distance_npz["array"]

        m = pd.read_parquet("../../data/raw/geo_data/distance_cities/mapping.parquet")
        self.mapping = {value: idx for idx, value in m.iloc[:, 0].items()}

        self.N = len(m)

    def _get_k(i, j, N):
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    def w(self, alpha, beta):
        i, j = self.mapping[alpha], self.mapping[beta]
        if i > j:
            i, j = j, i
        return self.distance_matrix[self._get_k(i, j, self.N)]

    def W_i(self, alpha):
        i = self.mapping[alpha]
        indices = np.array(list(self.mapping.values()))
        j_vals = indices
        i_vals = np.full_like(indices, i)
        swap_mask = i_vals > j_vals
        i_corrected = np.where(swap_mask, j_vals, i_vals)
        j_corrected = np.where(swap_mask, i_vals, j_vals)
        k_indices = self._get_k(i_corrected, j_corrected, self.N)
        return self.distance_matrix[k_indices]


class BoostingSpatialLoss:
    @staticmethod
    def LISE(y):
        """Computes Local Moran's I"""
        cdm = ComposeDistanceMatrix()._load_files()
        N = len(y)
        y_c = y - y.mean()
        m2 = (y_c**2).sum() / N
        z = (y_c / m2) @ (cdm.W() @ y_c).T
        return z

    @staticmethod
    def spatial_loss(lambd, L):
        """Following: Geerts, M., vanden Broucke, S., & De Weerdt, J. (2024).
        A Spatial Loss Function for Gradient Boosted Trees.
        In CEUR Workshop Proceedings.
        R. Piskac c/o Redaktion Sun SITE Informatik V RWTH Aachen."""
        S = BoostingSpatialLoss.LISE

        def loss(y_pred, y_true):
            return L(y_pred, y_true) + lambd * L(S(y_pred), S(y_true))

        return loss
