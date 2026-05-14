from loguru import logger
from sklearn.model_selection import train_test_split

from src.components.data_processing.data_loader import DataLoader
from src.components.modelling.data_split import Splitter
from src.components.modelling.meta_booster import MetaBooster


class ExplainCore:
    """Core Explanability Module"""

    def __init__(self, var, year, t):
        self.var = var
        self.year = year
        self.t = t

    @staticmethod
    def _load_model(data_path, var, year, type_, vars_, model_version, fs):
        model_path = f"{data_path}output/models/model_{year}_{type_}_{str(vars_)}_{model_version}.pkl"
        model = DataLoader.load_pickle(file_path=model_path, fs=fs)
        if not isinstance(model.models[var], MetaBooster):
            logger.error(
                "This pipeline is not configured for this type of model. Only metaboosting models. Raising an error"
            )
            raise ValueError(
                "This pipeline is not configured for this type of model. Only metaboosting models."
            )
        n_models = len(model.models[var].best_models)
        return model, n_models

    def _data_processing(self, data):
        st = Splitter(self.var)
        split_method = f"{self.year}_{self.t}"
        X, y, y_split = st.get_Xy(data)
        _, X_test, _, _, y_test, _ = st.split(
            X,
            y_split,
            split_method=split_method,
        )
        return X_test, y_test

    def stratify_sample(self, X, y, frac=None, random_state=42):
        if frac is None:
            return X, y
        else:
            X_sample, _, y_sample, _ = train_test_split(
                X, y, train_size=frac, random_state=random_state
            )  # No stratification
            return X_sample, y_sample

    def run(self, data, frac=None):
        X, y = self._data_processing(data)
        X_sampled, y_sampled = self.stratify_sample(X, y, frac=frac)
        commune_sampled = data.loc[X_sampled.index, "codecommune"]
        return X_sampled, y_sampled, commune_sampled
