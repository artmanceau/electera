import numpy as np
from loguru import logger
from src.components.explanability.core_explanability import ExplainCore
from src.components.utils.config import CFConfig
from src.components.utils.read_config import ConfigReader
from src.components.data_processing.data_loader import DataLoader
from src.components.explanability.counterfactuals.cf_data_processing import CounterfactualDataProcessing
from src.components.explanability.counterfactuals.cf_generator.generator import CounterfactualGenerator


class CounterfactualPipeline:
    """Build as an interface between electera and counterfactuals generation process"""

    def __init__(self, var, year, type_, vars_, version, data_path, fs=None):
        """Initialize the explainability pipeline with a configuration."""
        self.cf_config = ConfigReader._read_config(
                "config/counterfactual_config.json", CFConfig
            )
        self.model_version = version
        self.var = var
        self.year = year
        self.vars_ = vars_
        self.type_ = type_
        self.t = 0 if self.type_ == "pres" else 1
        self.data_path = data_path
        self.fs = fs

    def run(self, codecommune, variation=0.2):
        """Generate counterfactual explanation using the infer method of a model"""
        ec = ExplainCore(self.var, self.year, self.t)

        # 0. Get model
        self.model, self.n_models = ec._load_model(data_path=self.data_path, var=self.var, year=self.year, type_=self.type_, vars_=self.vars_, model_version=self.model_version, fs=self.fs)
        
        ## Assumes a choice for the data distribution:
        ##  1. Evolution : same commune over time. Use diff (over codecommune)
        ##  2. Neighborood : same election (same codecommune) but don't use diff.
        data = DataLoader.load_dataset(self.model.data_paths[self.var],  fs=self.fs, formate='parquet', columns=None, filters=[("annee", "==", float(self.year)), ("type", "==", self.t)])
        
        # 1. Counterfactual generation preparation
        stat_computer = CounterfactualDataProcessing(self.model.models[self.var].features, self.cf_config)
        (
                metadata,
                mean_values,
                _,
                mad_values,
                metadata_gen,
                mean_values_gen,
                _,
                mad_values_gen,
                correlation_matrix,
                features_distributions,
            ) = stat_computer.compute_statistics(data[['codecommune']+self.model.models[self.var].features])
        transformer_object = stat_computer.get_transformer()

        data_dict = {
                "data": data[self.model.models[self.var].features + ['pvotep'+self.var]],
                "metadata": metadata_gen,
                "features_distributions": features_distributions,
                "correlation_matrix": correlation_matrix,
                "mad_values": mad_values_gen,
                "mean_values": mean_values_gen,
                "transformer": transformer_object,
            }

        # Generated for the first model
        model_object = self.model.models[self.var].best_models[self.cf_config.model_id]
        model_object.get_booster().feature_names = self.model.models[self.var].features
        
        # 2. Generate counterfacutals
        generator = CounterfactualGenerator(
                model=model_object,
                config=self.cf_config,
                data_dict=data_dict
            )
        data_instance = data[(data['codecommune'] == codecommune) & (data['type'] == self.t) & (data['annee'] == float(self.year))]

        abs_variation = np.abs(variation)
        assert abs_variation != 0

        if variation < 0:
            desired_ranges_scenario = ([-abs_variation, 0], f"Variation in the vote of -{abs_variation}%")
        else:
            desired_ranges_scenario = ([0, abs_variation], f"Variation in the vote of +{abs_variation}%")

        assert len(data_instance) == 1
        interval, scenario_name = desired_ranges_scenario
        assert interval[0] < interval[1]
        
        counterfactuals_list = generator.generate_counterfactuals(
                    query_instances=data_instance[self.model.models[self.var].features].copy(deep=True),
                    desired_ranges_scenario=desired_ranges_scenario,
                    etiquette=None
                )
        # For later :
        #   - Etiquetting is deactivated, but provided data is from the same election and same type
        #   - No feasibility enforcing 
        #   - Only one boosting model

        return counterfactuals_list


if __name__ == "__main__":
    pipeline = CounterfactualPipeline(var='voteG', year=2022, type_='pres', vars_="['voteG', 'voteD', 'voteCG', 'voteCD', 'voteC', 'par']", version='0.0.1.', data_path="s3://arthurmanceau/election_modeling_uhcp/data/")
    counterfactuals_list = pipeline.run(codecommune='01001', variation=0.05)
    logger.success(counterfactuals_list)