import argparse
import json
import sys

from loguru import logger
from electera.pipeline.election_backtester import BackTester
from electera.pipeline.explain_model import Explainer


def parse_json_arg(value):
    """Attempts to parse a string as JSON; returns the original value if it fails."""
    if value is None:
        return None
    if isinstance(value, str) and (value.startswith("[") or value.startswith("{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def main():
    parser = argparse.ArgumentParser(description="Electera Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Backtester Subcommand
    bt_parser = subparsers.add_parser("backtest", help="Run election backtester")
    bt_parser.add_argument("--data-path", type=str, help="Override data_path")
    bt_parser.add_argument("--dataset-path", type=str, help="Override dataset_path")
    bt_parser.add_argument("--models", type=str, help="Override models (JSON list)")
    bt_parser.add_argument("--k-year", type=str, help="Override k_year (JSON dict)")
    bt_parser.add_argument("--k-type", type=str, help="Override k_type (JSON list)")
    bt_parser.add_argument(
        "--political-trends",
        type=str,
        help="Override political_trends (JSON list of lists)",
    )
    bt_parser.add_argument(
        "--predict-delta", type=str, help="Override predict_delta (true/false)"
    )
    bt_parser.add_argument(
        "--predict-percentile",
        type=str,
        help="Override predict_percentile (true/false)",
    )
    bt_parser.add_argument(
        "--organize-vote", type=str, help="Override organize_vote (true/false)"
    )
    bt_parser.add_argument(
        "--use-mlflow", type=str, help="Override use_mlflow (true/false)"
    )
    bt_parser.add_argument(
        "--mlflow-experiment", type=str, help="Override mlflow_experiment"
    )
    bt_parser.add_argument("--version", type=str, help="Override version")

    # Explainer Subcommand
    ex_parser = subparsers.add_parser("explain", help="Run model explainer")
    ex_parser.add_argument("--model-version", type=str, help="Override model_version")
    ex_parser.add_argument("--years", type=str, help="Override years (JSON dict)")
    ex_parser.add_argument("--types", type=str, help="Override types (JSON list)")
    ex_parser.add_argument(
        "--vars", dest="vars_", type=str, help="Override vars_ (JSON list of lists)"
    )
    ex_parser.add_argument("--data-path", type=str, help="Override data_path")
    ex_parser.add_argument("--steps", type=str, help="Override steps (JSON list)")

    args = parser.parse_args()

    if args.command == "backtest":
        overrides = {}
        for key, value in vars(args).items():
            if key == "command":
                continue
            if value is not None:
                parsed_val = parse_json_arg(value)
                if isinstance(parsed_val, str) and parsed_val.lower() in [
                    "true",
                    "false",
                ]:
                    parsed_val = parsed_val.lower() == "true"
                overrides[key] = parsed_val

        logger.info(f"Launching BackTester with overrides: {overrides}")
        BackTester(**overrides).run()

    elif args.command == "explain":
        overrides = {}
        for key, value in vars(args).items():
            if key == "command":
                continue
            if value is not None:
                overrides[key] = parse_json_arg(value)

        logger.info(f"Launching Explainer with overrides: {overrides}")
        Explainer(**overrides).run()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
