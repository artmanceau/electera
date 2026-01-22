import os
import pickle

from src.components.explanability.core_explanability import ExplainCore
from src.components.utils.config import CounterfactualConfig
from src.components.utils.read_config import ConfigReader


class CounterfactualPipeline:

    def __init__(self):
        """Initialize the explainability pipeline with a configuration."""
        self.config = ConfigReader._read_config(
            "config/counterfactual_config.json", CounterfactualConfig
        )
        self.model_path = self.config.model_path
        self.var = self.config.var
        self.year = self.config.year
        self.t = 0 if self.config.type_ == "pres" else 1

        # read model
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        self.output_dir = "results/cfs/"
        os.makedirs(self.output_dir, exist_ok=True)

        self.n_models = len(self.model.models[self.var].best_models)

    def run(self):
        """Generate counterfactual explanation using the infer method of a model"""
        # 1. Get sample data from model
        X, y = ExplainCore(self.model, self.var, self.year, self.t).run()

        # 2. Generate counterfacutals
        # cfs = CFS.generate_counterfactuals(X, y)
        # TODO + Import code


if __name__ == "__main__":
    pipeline = CounterfactualPipeline()
    pipeline.run()
