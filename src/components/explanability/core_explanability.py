from sklearn.model_selection import train_test_split

from src.components.data_processing.data_loader import DataLoader
from src.components.modelling.data_split import Splitter


class ExplainCore:
    """Core Explanability Module"""

    def __init__(self, model, var, year, t):
        self.model = model
        self.var = var
        self.year = year
        self.t = t

    def _data_processing(self, data):
        st = Splitter("p" + self.var)
        split_method = f"{self.year}_{self.t}"
        X, y, y_split = st.get_Xy(data)
        _, X_test, _, _, y_test, _ = st.split(
            X,
            y_split,
            split_method=split_method,
        )
        return X_test, y_test

    def stratify_sample(self, X, y, frac=0.30, random_state=42):
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=frac, random_state=random_state
        )  # No stratification
        return X_sample, y_sample

    def run(self, data):
        X, y = self._data_processing(data)
        X_sampled, y_sampled = self.stratify_sample(X, y)
        commune_sampled = data.loc[X_sampled.index, "codecommune"]
        return X_sampled, y_sampled, commune_sampled
