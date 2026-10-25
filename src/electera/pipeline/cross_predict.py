import argparse
import io
import re
from functools import lru_cache
from pathlib import Path

import boto3
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from botocore import UNSIGNED
from botocore.config import Config

DATA_PATH = "s3://arthurmanceau/election_modeling_uhcp/data/"
SOURCE = "derived/processed/data_processed_presidentiel_legislative_from1950_to2027_20261006_185710.parquet/"
OUTPUT_DIR = Path("cross_predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_S3_BUCKET = "arthurmanceau"
OUTPUT_S3_PREFIX = "election_modeling_uhcp/data/analysis/cross_predictions_error/"


def _load_election_results(year, election_type, political_trends):
    election_type_long = "presidentiel" if election_type == "pres" else "legislative"
    ground_truth_data_path = (
        DATA_PATH
        + f"raw/elections/{election_type_long}/{year}/{election_type}{year}_csv/{election_type}{year}comm.parquet"
    )
    if "CGCCD" in political_trends or "tauCGCCD" in political_trends:
        political_trends_ = ["par", "CG", "C", "G", "D", "CD"]
        X_true = pl.read_parquet(
            ground_truth_data_path,
            storage_options={"aws_skip_signature": "true"},
        ).to_pandas()[
            ["codecommune", "nomcommune", "inscrits", "votants", "exprimes"]
            + [
                f"vote{t.replace('tau', '')}"
                for t in political_trends_
                if t.replace("tau", "") != "par"
            ]
            + [
                f"pvote{t.replace('tau', '')}"
                for t in political_trends_
                if t.replace("tau", "") != "par"
            ]
            + ["ppar"]
        ]
        X_true = X_true.copy()
        X_true["voteCGCCD"] = X_true["voteCG"] + X_true["voteC"] + X_true["voteCD"]
        X_true["pvoteCGCCD"] = X_true["pvoteCG"] + X_true["pvoteC"] + X_true["pvoteCD"]
    else:
        X_true = pl.read_parquet(
            ground_truth_data_path,
            storage_options={"aws_skip_signature": "true"},
        ).to_pandas()[
            ["codecommune", "nomcommune", "inscrits", "votants", "exprimes"]
            + [
                f"vote{t.replace('tau', '')}"
                for t in political_trends
                if t.replace("tau", "") != "par"
            ]
            + [
                f"pvote{t.replace('tau', '')}"
                for t in political_trends
                if t.replace("tau", "") != "par"
            ]
            + ["ppar"]
        ]
    X_true = X_true.dropna()
    str_cols = ["codecommune", "nomcommune"]
    float_cols = [
        f"pvote{t.replace('tau', '')}"
        for t in political_trends
        if t.replace("tau", "") != "par"
    ] + ["ppar"]
    int_cols = [
        f"vote{t.replace('tau', '')}"
        for t in political_trends
        if t.replace("tau", "") != "par"
    ] + ["inscrits", "votants", "exprimes"]
    X_true[str_cols] = X_true[str_cols].astype(str)
    X_true[int_cols] = X_true[int_cols].astype(int)
    X_true[float_cols] = X_true[float_cols].astype(float)
    # exprimes_ = X_true[
    #     [
    #         f"vote{t.replace('tau', '')}"
    #         for t in political_trends
    #         if t.replace("tau", "") != "par"
    #     ]
    # ].sum(axis=1)

    X_true = X_true.loc[
        ~(
            X_true["codecommune"].isin(
                ["75056"]
                + [f"1320{i}" for i in range(1, 9 + 1)]
                + [f"132{i}" for i in range(10, 16 + 1)]
                + [f"69328{i}" for i in range(1, 9 + 1)]
            )
        ),
        :,
    ]
    return X_true


@lru_cache(maxsize=None)
def load_dataset(year, election_type):
    return (
        pl.scan_parquet(
            DATA_PATH + SOURCE,
            storage_options={"aws_skip_signature": "true"},
        )
        .filter(pl.col("annee") == year)
        .filter(pl.col("election_type") == election_type)
        .collect()
        .to_pandas()
    )


@lru_cache(maxsize=None)
def load_model(year, election_type, trends, version="6.0.0"):
    s3 = boto3.client(
        "s3",
        endpoint_url="https://minio.lab.sspcloud.fr",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )

    bucket = "arthurmanceau"
    key = f"election_modeling_uhcp/data/output/models/model_{year}_{election_type}_{trends}_{version}.joblib"

    response = s3.get_object(Bucket=bucket, Key=key)
    return joblib.load(io.BytesIO(response["Body"].read()))


@lru_cache(maxsize=None)
def load_election_results(year, election_type, political_trends):
    return _load_election_results(year, election_type, list(political_trends))


PRESIDENTIAL_ELECTIONS = sorted(
    [1969, 2022, 1995, 1974, 1981, 1988, 2002, 2007, 2012, 2017]
)
LEGISLATIVE_ELECTIONS = sorted(
    [1968, 2022, 1973, 1978, 1981, 1986, 1988, 1993, 1997, 2002, 2007, 2012, 2017, 2024]
)
ELECTION_LISTS = {
    "leg": LEGISLATIVE_ELECTIONS,
    "pres": PRESIDENTIAL_ELECTIONS,
}
POLITICAL_TRENDS = [
    ("taupar", "tauDCD", "tauGCG", "tauC"),
    ("taupar", "tauTD", "tauTG"),
    ("taupar", "tauD", "tauG", "tauCGCCD"),
    ("taupar", "tauD", "tauCD", "tauC", "tauCG", "tauG"),
]


def compute_metrics():
    metric_names = [
        "bias",
        "absbias",
        "rel_bias",
        "rel_absbias",
        "mae",
        "rel_mae",
        "weighted_mae",
        "rel_weighted_mae",
        "rmse",
        "rel_rmse",
        "weighted_rmse",
        "rel_weighted_rmse",
    ]
    metrics = {name: {} for name in metric_names}

    for election_type, election_list in ELECTION_LISTS.items():
        election_type_long = (
            "presidentiel" if election_type == "pres" else "legislative"
        )

        for political_trends in POLITICAL_TRENDS:
            trends_key = "_".join(political_trends)
            model_trends = "_".join(sorted(political_trends))

            for index, _ in enumerate(election_list):
                year = election_list[index]
                model_year = (
                    election_list[index + 1]
                    if index < len(election_list) - 1
                    else (2027 if election_type == "pres" else 2024)
                )
                model = load_model(model_year, election_type, model_trends)

                for eval_year in election_list:
                    results = load_election_results(
                        eval_year, election_type, political_trends
                    )
                    input_data = load_dataset(eval_year, election_type_long)

                    predictions = model.predict_votes(input_data)
                    merged = model.evaluate_predictions(predictions, results)

                    diff_cols = [c for c in merged.columns if c.endswith("_diff")]

                    # Poids communaux : inscrits réels.
                    weights = merged["inscrits_true"].astype(float)

                    # Moyennes pondérées globales, comme dans le notebook.
                    # valid_weights = (
                    #     weights.notna() & np.isfinite(weights) & (weights >= 0)
                    # )
                    # total_weight = weights[valid_weights].sum()

                    key = (year, eval_year)

                    for col in diff_cols:
                        trend = col.removesuffix("_diff")
                        true_col = f"{trend}_true"
                        pred_col = f"{trend}_pred"

                        if (
                            true_col not in merged.columns
                            or pred_col not in merged.columns
                        ):
                            continue

                        diff = merged[col].astype(float)
                        actual = merged[true_col].astype(float)
                        predicted = merged[pred_col].astype(float)

                        valid = (
                            diff.notna()
                            & actual.notna()
                            & predicted.notna()
                            & np.isfinite(diff)
                            & np.isfinite(actual)
                            & np.isfinite(predicted)
                        )

                        d = diff[valid]
                        y_true = actual[valid]
                        y_pred = predicted[valid]
                        w = weights[valid]

                        if d.empty:
                            continue

                        # Erreurs communales.
                        mae = d.abs().mean()
                        rmse = np.sqrt((d**2).mean())

                        # Relative metrics (compared to avg true value)
                        avg_true = y_true.mean()
                        rel_mae = (mae / avg_true * 100) if avg_true != 0 else np.nan
                        rel_rmse = (rmse / avg_true * 100) if avg_true != 0 else np.nan

                        # Pondération par les inscrits.
                        valid_w = w.notna() & np.isfinite(w) & (w >= 0)
                        d_w = d[valid_w]
                        true_w = y_true[valid_w]
                        pred_w = y_pred[valid_w]
                        w = w[valid_w]

                        if w.sum() > 0:
                            weighted_mae = (d_w.abs() * w).sum() / w.sum()
                            weighted_rmse = np.sqrt(((d_w**2) * w).sum() / w.sum())

                            avg_pred_weighted = (pred_w * w).sum() / w.sum()
                            avg_true_weighted = (true_w * w).sum() / w.sum()

                            # Relative weighted metrics
                            rel_weighted_mae = (
                                (weighted_mae / avg_true_weighted * 100)
                                if avg_true_weighted != 0
                                else np.nan
                            )
                            rel_weighted_rmse = (
                                (weighted_rmse / avg_true_weighted * 100)
                                if avg_true_weighted != 0
                                else np.nan
                            )

                            bias = avg_pred_weighted - avg_true_weighted
                            absbias = abs(bias)

                            if avg_true_weighted != 0:
                                rel_bias = avg_pred_weighted / avg_true_weighted - 1
                                rel_absbias = abs(rel_bias)
                            else:
                                rel_bias = np.nan
                                rel_absbias = np.nan
                        else:
                            weighted_mae = np.nan
                            weighted_rmse = np.nan
                            rel_weighted_mae = np.nan
                            rel_weighted_rmse = np.nan
                            bias = np.nan
                            absbias = np.nan
                            rel_bias = np.nan
                            rel_absbias = np.nan

                        values = {
                            "bias": float(bias),
                            "absbias": float(absbias),
                            "rel_bias": float(rel_bias),
                            "rel_absbias": float(rel_absbias),
                            "mae": float(mae),
                            "rel_mae": float(rel_mae),
                            "weighted_mae": float(weighted_mae),
                            "rel_weighted_mae": float(rel_weighted_mae),
                            "rmse": float(rmse),
                            "rel_rmse": float(rel_rmse),
                            "weighted_rmse": float(weighted_rmse),
                            "rel_weighted_rmse": float(rel_weighted_rmse),
                        }

                        for metric_name, value in values.items():
                            metrics[metric_name].setdefault(
                                election_type, {}
                            ).setdefault(trends_key, {}).setdefault(trend, {})[
                                key
                            ] = value

                    print(
                        f"Computed metrics for {election_type}, "
                        f"trends={political_trends}, "
                        f"model_year={year}, "
                        f"eval_year={eval_year}"
                    )

    return metrics


def to_matrix(values, years):
    matrix = pd.DataFrame(np.nan, index=years, columns=years)
    if values:
        pairs = pd.Series(values, dtype=float)
        for (model_year, eval_year), value in pairs.items():
            matrix.loc[eval_year, model_year] = value * 100
    return matrix


def save_metric_figures(metrics):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    for metric_name, per_election in metrics.items():
        for election_type, per_trends_key in per_election.items():
            for trends_key, per_trend in per_trends_key.items():
                # trends is the list of variables (e.g., 'taupar', 'tauD')
                trends = sorted(per_trend)
                years = sorted(
                    {
                        year
                        for values in per_trend.values()
                        for pair in values
                        for year in pair
                    }
                )

                for trend in trends:
                    matrix = to_matrix(per_trend[trend], years)

                    # Scaling logic for large values
                    scale_factor = 1.0
                    unit = ""
                    max_val = matrix.max().max()

                    if not np.isnan(max_val):
                        if max_val >= 10000:
                            scale_factor = 10000.0
                            unit = " (tens of thousands)"
                        elif max_val >= 1000:
                            scale_factor = 1000.0
                            unit = " (thousands)"

                    matrix_scaled = matrix / scale_factor

                    plt.figure(figsize=(8, 6))
                    sns.heatmap(
                        matrix_scaled,
                        annot=True,
                        fmt=".1f",
                        cmap="GnBu",
                        square=True,
                        annot_kws={"size": 8},
                        cbar_kws={"label": f"{metric_name.upper()}{unit}"},
                    )

                    title = (
                        f"{metric_name.capitalize()} of cross-predictions for variable {trend} "
                        f"(model trained on {election_type} with the following political bloc {trends_key})"
                    )
                    plt.title(title, fontsize=10, pad=18)
                    plt.xlabel(
                        "Main election year used in the training set", fontsize=10
                    )
                    plt.ylabel("Election year predicted", fontsize=10)
                    plt.tick_params(axis="both", labelsize=10)
                    plt.tight_layout()

                    safe_trends = re.sub(r"[^A-Za-z0-9_.-]+", "_", trends_key)
                    safe_trend = re.sub(r"[^A-Za-z0-9_.-]+", "_", trend)
                    output_path = (
                        OUTPUT_DIR
                        / f"{metric_name}_{election_type}_{safe_trends}_{safe_trend}.pdf"
                    )
                    plt.savefig(output_path, format="pdf", bbox_inches="tight")
                    plt.close()
                    print(f"Saved {output_path}")


def save_metrics(metrics):
    metrics_path = OUTPUT_DIR / "metrics.joblib"
    joblib.dump(metrics, metrics_path, compress=3)
    print(f"Saved metrics to {metrics_path}")
    # upload_output_file(metrics_path)


def load_metrics():
    metrics_path = OUTPUT_DIR / "metrics.joblib"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No saved metrics found at {metrics_path}. Run with --mode compute or --mode all first."
        )
    return joblib.load(metrics_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-prediction metrics and/or save heatmaps."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "compute", "figures"),
        default="all",
        help="all: compute metrics and figures; compute: only compute and save metrics; figures: load saved metrics and only generate figures.",
    )
    args = parser.parse_args()

    if args.mode in ("all", "compute"):
        metrics = compute_metrics()
        save_metrics(metrics)
        if args.mode == "all":
            save_metric_figures(metrics)
    else:
        save_metric_figures(load_metrics())


if __name__ == "__main__":
    main()
