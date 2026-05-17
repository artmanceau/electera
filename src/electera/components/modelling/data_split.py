# Functionalities to prepare data for model training / inference
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


class Splitter:
    def __init__(self, var):
        self.var = var
        self.vote_variable = f"pvote{self.var}"

    @staticmethod
    def _find_correlated_in(X, threshold=0.95):
        corr = X.corr(method="pearson")
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        return set(
            [column for column in upper.columns if any(upper[column] > threshold)]
        )

    @staticmethod
    def is_stationnary_feature(feature):
        return ("pct_change" in feature) | ("rank" in feature)

    def clean_features_list(
        self,
        X_train,
        X_val,
        X_test,
        nan_threshold=0.9,
        keep_stationnary=True,
        remove_correlated=True,
        features_to_save=[],
    ):
        """
        Models are faster to adjust with less features.
        We want to avoid training with features that don't exist in both X_train, X_val, X_test.
        Features under threshold in at least on of the three dataset is removed.
        """
        nan_cols = set()
        for X in [X_train, X_test, X_val]:
            nan_X = X.isnull().mean()
            nan_cols_X = set(nan_X[nan_X > nan_threshold].index.tolist())
            nan_cols = nan_cols.union(nan_cols_X)

        nan_cols_list = list(nan_cols)
        n_cols = len(nan_cols_list)

        if n_cols > 0:
            logger.warning(
                f"The following columns ({n_cols}) are removed because they are not populated (enough, i.e. more NaNs than {nan_threshold * 100}%) in both three datasets (train, val, test)."
            )
            X_train = X_train.drop(columns=nan_cols)
            X_val = X_val.drop(columns=nan_cols)
            X_test = X_test.drop(columns=nan_cols)

        assert set(X_train.columns) == set(X_test.columns) == set(X_val.columns)
        columns = X_train.columns.to_list()

        socio_eco_features = [col for col in columns if col.startswith("F_")]
        non_socio_eco_features = list(set(columns) - set(socio_eco_features))
        assert len(non_socio_eco_features) + len(socio_eco_features) == len(columns)

        if keep_stationnary:
            # Stationnary features are pct_change, rank and delta
            logger.info("Selecting only stationnary features")
            socio_eco_features_stationnary = [
                col for col in socio_eco_features if self.is_stationnary_feature(col)
            ]
            socio_eco_features = socio_eco_features_stationnary

        if remove_correlated:
            # Pearson correlation
            logger.info("Removing most correlated features")
            to_drop_correlated = set()
            for X in [X_train, X_test, X_val]:
                to_drop_correlated = to_drop_correlated.union(
                    self._find_correlated_in(X[socio_eco_features])
                )

            socio_eco_features = list(set(socio_eco_features) - to_drop_correlated)

        features = non_socio_eco_features + socio_eco_features
        features = list(set(features).union(set(features_to_save)))

        # Percentage of missing values
        nb_nans_train = X_train[features].isnull().sum().sum()
        logger.debug(
            f"Percentage of missing values in the training dataset: {nb_nans_train}"
        )
        if nb_nans_train > 0:
            # Handle nan in dep_num
            logger.warning(
                f"The following columns in the training dataset contains missing values: {X_train.columns[X_train.isna().any()]}."
            )

        return X_train[features], X_val[features], X_test[features]

    def get_Xy(self, data, predict_delta=False, selected_features=None):
        feature_cols = data.columns[data.columns.str.startswith("F_")].tolist()

        previous_vote_cols = [
            f"previous{self.vote_variable}",
            f"previousprevious{self.vote_variable}",
        ]
        # We may have missing values in the previous vote statitics
        # Fill NaN with mean within each dep_num group
        data[previous_vote_cols] = (
            data.groupby(data["dep"])[previous_vote_cols]
            .transform(lambda col: col.fillna(col.mean()))
            .fillna(data[previous_vote_cols].mean())
        )

        X = data[
            feature_cols
            + ["inscrits", "type", "annee", "lat", "long", "dep_num"]
            + previous_vote_cols
        ].astype(float)

        if predict_delta:
            y = data[self.vote_variable] - data[f"previouspvote{self.var}"]
            y_split = pd.concat([y, data[["annee", "type"]]], axis=1)
            y_split.columns = [self.vote_variable, "annee", "type"]
            logger.debug(
                f"Prediction of the difference in vote {self.var} over two elections"
            )
        else:
            y = data[self.vote_variable]
            y_split = data[[self.vote_variable, "annee", "type"]]
            logger.debug(f"Prediction of vote statistics {self.var}")

        # Replace inf with nan
        # (handled by models - all proposed models should handle missing values)
        inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        assert inf_count == 0

        # Make sure there is no nan
        indices_to_drop = y[y.isna()].index
        if len(indices_to_drop) > 0:
            X.drop(index=indices_to_drop, errors="ignore", inplace=True)
            y.drop(index=indices_to_drop, errors="ignore", inplace=True)
            y_split.drop(index=indices_to_drop, errors="ignore", inplace=True)
            logger.warning(
                f"{len(indices_to_drop)} rows were dropped because the target was missing!"
            )

        # Make sure columns in X are all float
        non_float_columns = X.select_dtypes(exclude=["float"]).columns
        assert len(non_float_columns) == 0

        if selected_features is None:
            selected_features = list(X.columns)
        else:
            selected_features = list(set(selected_features) & set(list(X.columns)))

        return X[selected_features], y, y_split

    def _split_shuffle(self, X, y_split, y, t, test_size, val_size, random_seed):
        # First split: Train + Validation and Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y_split,
            test_size=test_size,
            random_state=random_seed,
            shuffle=True,
        )

        # Second split: Train and Validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size,
            random_state=random_seed,
            shuffle=True,
        )
        return (
            X_train,
            X_val,
            X_test,
            y_train[self.vote_variable],
            y_val[self.vote_variable],
            y_test[self.vote_variable],
        )

    def _split_years(self, X, y_split, y, t, test_size, val_size, random_seed):
        years = X["annee"].unique()
        # First split: Train + Validation and Test
        train_val_years, test_years = train_test_split(
            years,
            test_size=test_size,
            random_state=random_seed,
            shuffle=True,
        )

        X_test = X[X["annee"].isin(test_years)]
        y_test = y_split[y_split["annee"].isin(test_years)][self.vote_variable]

        train_years, val_years = train_test_split(
            train_val_years,
            test_size=val_size,
            random_state=random_seed,
            shuffle=True,
        )
        X_train = X[X["annee"].isin(train_years)]
        y_train = y_split[y_split["annee"].isin(train_years)][self.vote_variable]
        X_val = X[X["annee"].isin(val_years)]
        y_val = y_split[y_split["annee"].isin(val_years)][self.vote_variable]
        return X_train, X_val, X_test, y_train, y_val, y_test

    def split(
        self,
        X,
        y_split,
        split_method="shuffle",
        test_size=0.2,
        val_size=0.2,
        random_seed=42,
    ):
        logger.debug(f"Splitting data with method: {split_method}")
        if split_method not in ["shuffle", "year"]:
            split_m = "custom"
            y, t = split_method.split("_")
            logger.debug(
                f"The test election will be {t} {y}. The train election will be the previous one of the same type."
            )
        else:
            split_m = split_method
            y, t = "2022", "0"

        split_methods = {
            "shuffle": self._split_shuffle,
            "year": self._split_years,
            "custom": self._custom_split,
        }
        return split_methods[split_m](
            X=X,
            y_split=y_split,
            y=y,
            t=t,
            test_size=test_size,
            val_size=val_size,
            random_seed=random_seed,
        )

    def _custom_split(self, X, y_split, y, t, test_size, val_size, random_seed):
        available_years = np.sort(
            X[X["type"].astype(int) == int(t)]["annee"].astype(int).unique()
        ).tolist()
        x = available_years.index(int(y))
        if x < 2:
            logger.warning(
                "Not possible because we don't have enough past elections. Choosing random elections years instead"
            )

        t_year = available_years[x - 1]
        v_year = available_years[x - 2]

        # mask
        base_mask = X["type"].astype(int) == int(t)
        mask_train = base_mask & (X["annee"].astype(int) == int(t_year))
        mask_val = base_mask & (X["annee"].astype(int) == int(v_year))
        mask_test = base_mask & (X["annee"].astype(int) == int(y))

        # X
        X_train = X.loc[mask_train].copy()
        X_val = X.loc[mask_val].copy()
        X_test = X.loc[mask_test].copy()

        # y
        y_train = y_split.loc[mask_train, self.vote_variable].copy()
        y_val = y_split.loc[mask_val, self.vote_variable].copy()
        y_test = y_split.loc[mask_test, self.vote_variable].copy()

        logger.debug(
            f"Test election: {y}, train election: {t_year}, validation election: {v_year}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def remove_previous_features(self, X):
        # Remove previous election features from training data
        cols_to_drop = []
        if f"pvoteprevious{self.var}" in X.columns:
            cols_to_drop.append(f"pvoteprevious{self.var}")
        if f"pvotepreviousprevious{self.var}" in X.columns:
            cols_to_drop.append(f"pvotepreviousprevious{self.var}")

        if cols_to_drop:
            # Drop columns directly from DataFrames
            X = X.drop(columns=cols_to_drop)
        return X
