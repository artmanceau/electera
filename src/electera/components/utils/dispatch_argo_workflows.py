import argparse
import json
import sys
from typing import Any, Dict, List, Optional


def generate_backtest_params(
    models: List[str],
    political_trends: List[List[str]],
    k_type: List[str],
    k_year: Dict[str, List[int]],
    experiment_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params = []
    for model in models:
        for trends in political_trends:
            for type_ in k_type:
                for year in k_year.get(type_, []):
                    # We output the exact keys needed for config overrides
                    param_set = {
                        "models": [model],
                        "political_trends": [trends],
                        "k_type": [type_],
                        "k_year": {type_: [year]},
                    }
                    if experiment_name:
                        param_set["mlflow_experiment"] = experiment_name
                    params.append(param_set)
    return params


def generate_explain_params(
    types: List[str],
    years: Dict[str, List[int]],
    vars_: List[List[str]],
) -> List[Dict[str, Any]]:
    params = []
    for type_ in types:
        for year in years.get(type_, []):
            for vs in vars_:
                for var in vs:
                    # We output the exact keys needed for config overrides
                    params.append(
                        {"types": [type_], "years": {type_: [year]}, "vars_": [vs]}
                    )
    return params


def parse_json_arg(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate parameters for Argo Workflows"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Backtest subcommand
    bt_parser = subparsers.add_parser("backtest", help="Generate backtest parameters")
    bt_parser.add_argument(
        "--models", type=str, required=True, help="JSON list of models"
    )
    bt_parser.add_argument(
        "--political-trends",
        type=str,
        required=True,
        help="JSON list of lists of trends",
    )
    bt_parser.add_argument(
        "--k-type", type=str, required=True, help="JSON list of election types"
    )
    bt_parser.add_argument(
        "--k-year", type=str, required=True, help="JSON dict of years per type"
    )
    bt_parser.add_argument("--mlflow-experiment", type=str, help="Experiment name")

    # Explain subcommand
    ex_parser = subparsers.add_parser("explain", help="Generate explain parameters")
    ex_parser.add_argument(
        "--types", type=str, required=True, help="JSON list of types"
    )
    ex_parser.add_argument(
        "--years", type=str, required=True, help="JSON dict of years per type"
    )
    ex_parser.add_argument(
        "--vars",
        dest="vars_",
        type=str,
        required=True,
        help="JSON list of lists of variables",
    )

    args = parser.parse_args()

    if args.command == "backtest":
        params = generate_backtest_params(
            models=parse_json_arg(args.models),
            political_trends=parse_json_arg(args.political_trends),
            k_type=parse_json_arg(args.k_type),
            k_year=parse_json_arg(args.k_year),
            experiment_name=args.mlflow_experiment,
        )
        print(json.dumps(params))
    elif args.command == "explain":
        params = generate_explain_params(
            types=parse_json_arg(args.types),
            years=parse_json_arg(args.years),
            vars_=parse_json_arg(args.vars_),
        )
        print(json.dumps(params))
    else:
        parser.print_help()
        sys.exit(1)
