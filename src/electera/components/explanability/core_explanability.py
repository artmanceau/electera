from sklearn.model_selection import train_test_split

from electera.components.data_processing.data_loader import DataLoader
from electera.components.modelling.data_split_pl import get_Xy_pl
from electera.components.modelling.meta_booster import MetaBooster
from electera.components.modelling.boosting.boosting import BoostingModel


class ExplainCore:
    """Core Explanability Module"""

    def __init__(self, var, year, t):
        self.var = var
        self.year = year
        self.t = t

    @staticmethod
    def _load_model(data_path, var, year, type_, vars_, model_version, fs):
        vars_.sort()
        vars_str = "_".join(vars_)
        model_path = f"{data_path}output/models/model_{year}_{type_}_{vars_str}_{model_version}.pkl"
        model = DataLoader.load_pickle(file_path=model_path, fs=fs)
        n_models = (
            len(model.models[var].best_models)
            if isinstance(model.models[var], MetaBooster)
            else 1
        )

        # Adapt boosting to metaboosting structure
        if isinstance(model.models[var], BoostingModel):
            setattr(model.models[var], "features", model.models[var].features_selected)
            setattr(model.models[var], "best_models", [model.models[var].model])

        return model, n_models

    def _data_processing(self, data):
        _, _, X_test, _, _, y_test, _, _, _, meta_test = get_Xy_pl(
            data=data,
            vote_variable=f"pvote{self.var}",
            year=self.year,
            election_type="presidentiel" if self.t == 1 else "legislative",
            predict_delta=False,
            predict_perc=False,
            split_method_way="time-serie-cv",
            remove_nulls=False,
        )
        meta_test["y"] = y_test
        return X_test, meta_test

    def stratify_sample(self, X, y, frac=None, random_state=42):
        if frac is None:
            return X, y
        else:
            X_sample, _, y_sample, _ = train_test_split(
                X, y, train_size=frac, random_state=random_state
            )  # No stratification
            return X_sample, y_sample

    def run(self, data, frac=None):
        X, meta = self._data_processing(data)
        X_sampled, meta_sampled = self.stratify_sample(X, meta, frac=frac)
        return X_sampled, meta_sampled["y"], meta_sampled["codecommune"]
