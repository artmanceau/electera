import json
import sys
from electera.components.utils.read_config import ConfigReader
from electera.components.utils.config import BackTesterConfig, ExplanabilityConfig


def generate_backtest_params():
    config = ConfigReader._read_config("../config/backtester.json", BackTesterConfig)
    params = []
    for model in config.models:
        for trends in config.political_trends:
            for type_ in config.k_type:
                for year in config.k_year.get(type_, []):
                    # We output the exact keys needed for config overrides
                    params.append({
                        "models": [model],
                        "political_trends": [trends],
                        "k_type": [type_],
                        "k_year": {type_: [year]}
                    })
    return params


def generate_explain_params():
    config = ConfigReader._read_config("../config/explainability.json", ExplanabilityConfig)
    params = []
    for type_ in config.types:
        for year in config.years.get(type_, []):
            for vs in config.vars_:
                for var in vs:
                    # We output the exact keys needed for config overrides
                    params.append({
                        "types": [type_],
                        "years": {type_: [year]},
                        "vars_": [vs]
                    })
    return params


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if mode == "backtest":
        print(json.dumps(generate_backtest_params()))
    elif mode == "explain":
        print(json.dumps(generate_explain_params()))
