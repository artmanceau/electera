import os
import pickle

from src.components.explanability.core_explanability import ExplainCore
from src.components.utils.config import CounterfactualConfig
from src.components.utils.read_config import ConfigReader
from src.components.modelling.meta_booster import MetaBooster
from src.components.data_processing.data_loader import DataLoader, DataUtils


class CounterfactualPipeline:

    def __init__(self):
        """Initialize the explainability pipeline with a configuration."""
        self.config = ConfigReader._read_config(
            "config/counterfactual_config.json", CounterfactualConfig
        )
        self.model_version = self.config.model_version
        self.var = self.config.var
        self.year = self.config.year
        self.t = 0 if self.config.type_ == "pres" else 1

        self.output_dir = "results/cfs/"
        os.makedirs(self.output_dir, exist_ok=True)

        self.n_models = len(self.model.models[self.var].best_models)

    def _load_model(self, var, year, type_, vars_):
        model_path = f"{self.data_path}output/models/model_{year}_{type_}_{str(vars_)}_{self.model_version}.pkl"
        self.model = DataLoader.load_pickle(file_path=model_path)
        if not isinstance(self.model.models[var], MetaBooster):
            logger.error(
                "This pipeline is not configured for this type of model. Only metaboosting models. Raising an error"
            )
            raise ValueError(
                "This pipeline is not configured for this type of model. Only metaboosting models."
            )
        self.n_models = len(self.model.models[var].best_models)

    def run(self):
        """Generate counterfactual explanation using the infer method of a model"""
         # 0. Get model
        self._load_model(self.var, self.year, self.type_, self.vars_)
        if data is None:
            data = self._load_data(self.model.data_paths[self.var])

        
        # 1. Get sample data from model
        X, y = ExplainCore(self.model, self.var, self.year, self.t).run()

        # 2. Generate counterfacutals
        breakpoint()
        # cfs = CFS.generate_counterfactuals(X, y)
        # TODO + Import code


if __name__ == "__main__":
    pipeline = CounterfactualPipeline()
    pipeline.run()
