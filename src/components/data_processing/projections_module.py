import re

import numpy as np
import pandas as pd


class ProjectionUtils:
    @staticmethod
    def extract_year_from_column(x):
        return re.search(r"\d{4}$", str(x))[0]

    @staticmethod
    def extract_feature_type_from_feature(x):
        return re.sub(r"\d{4}$", "", str(x))

    @staticmethod
    def find_all_years_of(feature, dfc):
        return [
            int(ProjectionUtils.extract_year_from_column(c))
            for c in dfc.columns
            if (feature in c and len(c.split("/")[1]) == len(feature.split("/")[1]) + 4)
        ]

    @staticmethod
    def find_all_cols_of(feature, dfc):
        return [
            c
            for c in dfc.columns
            if (feature in c and len(c.split("/")[1]) == len(feature.split("/")[1]) + 4)
        ]

    @staticmethod
    def find_last_year_of(feature, dfc):
        return ProjectionUtils.find_all_years_of(feature, dfc)[-1]

    @staticmethod
    def find_freq_of(feature, dfc):
        all_years = np.sort(ProjectionUtils.find_all_years_of(feature, dfc))
        if len(all_years) < 4:
            return "N/A"
        return int(
            np.mean(
                [
                    (all_years[i] - all_years[i - 1])
                    for i in range(len(all_years) - 4, len(all_years))
                ]
            )
        )


class ProjectionModel:
    def __init__(self, p=5, alpha=None):
        self.alpha = alpha
        self.p = p
        self.data = None
        self.feature = None
        self.reference_year = None
        self.new_columns = {}

    def fit(self, data):
        self.data = data

    def _find_col(self, i, feature, reference_year):
        if i <= reference_year:
            return self.data[f"{feature}{i}"]
        else:
            col_name = f"{feature}{i}"
            if col_name in self.new_columns:
                return self.new_columns[col_name]
            else:
                raise Exception("Column not created yet.")

    def predict_smoothing(self, start, end, feature, reference_year):
        # Predict. Start from n and predict to k iteratively.
        for i in range(start, end + 1):  # i = (n=2023), 2024, ..., (k=2027)
            s = 0
            for m in range(1, self.p + 1):
                s += self.alpha ** (m) * self._find_col(
                    i - m, feature, reference_year
                )  # i - m = (2022=2023-1=i-1), ... (2018=2022-5=i-p)

            self.new_columns[f"{feature}{i}"] = (
                self.alpha * self._find_col(i - 1, feature, reference_year)
                + (1 - self.alpha) * s
            )

        return self.new_columns

    def predict_linear(self, start, end, feature, reference_year):
        # Predict with a growth rate computed on the p previous years. From n to k.
        for i in range(start, end + 1):
            data_series = [
                self._find_col(i - m, feature, reference_year)
                for m in range(1, self.p + 1)
            ][::-1]
            growth_rates = (
                (pd.concat(data_series, axis=1))
                .pct_change(axis=1, fill_method=None)
                .mean(skipna=True, axis=1)
            )
            self.new_columns[f"{feature}{i}"] = (1 + growth_rates) * self._find_col(
                i - 1, feature, reference_year
            )
        return self.new_columns
