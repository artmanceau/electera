from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import polars as pl
import os
import polars_distance as pld
import polars as pl
import polars.selectors as cs
from loguru import logger
from shapely.geometry import shape
from loguru import logger
from electera.components.data_processing.data_loader import DataLoader, DataUtils

@dataclass
class Config:
    election_type: str = "pres"
    exclude: List[str] = None
    num_bins: int = 200
    dist_step: float = 1.5
    p_max: int = 350

vote = ["inscrits", "votants", "exprimes", "abstentions"]
pvote = ["pvotepar", "pvoteabs"]
tendances = ["G", "CG", "C", "CD", "D", "TD", "TG", "GCG", "DCD"]
tendances_column_vote = [f"vote{tendance}" for tendance in tendances]
tendances_column_pvote = ["p" + col for col in tendances_column_vote]

def previous(cols):
        return [f"previous{col}" for col in cols]

def previousprevious(cols):
        return [f"previousprevious{col}" for col in cols]

electoral_schema = {
        "dep": pl.String,
        "nomdep": pl.String,
        "codecommune": pl.String,
        "nomcommune": pl.String,
        "annee": pl.Int64,
        "type": pl.Int8,
        "election_type": pl.String,
        "election_code": pl.String,
        "inscrits": pl.Float64,
        "votants": pl.Float64,
        "exprimes": pl.Float64,
        # "inscritsT2": pl.Float64,
        # "votantsT2": pl.Float64,
        # "exprimesT2": pl.Float64,
        "voteG": pl.Float64,
        "voteCG": pl.Float64,
        "voteC": pl.Float64,
        "voteCD": pl.Float64,
        "voteD": pl.Float64,
        "voteTG": pl.Float64,
        "voteTD": pl.Float64,
        "voteGCG": pl.Float64,
        "voteDCD": pl.Float64,
        # "voixOUI": pl.Float64,
        # "voixNON": pl.Float64,
    }
socio_economic_schema = {
        'dep': pl.String,  # Key
        "codecommune": pl.String,  # key
        "raw": pl.Float64,
        "variable": pl.String,
        "annee": pl.Int64,  # key
        "feature_name": pl.String,  # key
        "lag": pl.Float64,
        "rank": pl.Float64,
        "delta": pl.Float64,
        "pct_change": pl.Float64,
    }

type_encoding = {
        "presidentiel": 0,
        "legislative": 1,
        "referundum": 2,
        "municipales": 3,
    }


def run_pipeline(dfs_by_file, df_pairs, centroids, communes_code_df, config=Config()):

    # 1. filter elections
    dfs = {k: v for k, v in dfs_by_file.items() if config.election_type in k}
    for k in (config.exclude or []):
        dfs.pop(k, None)

    # 2. compute tau
    for k, df in dfs.items():
        p = np.where(df["inscrits"] == 0, -1, df["votants"] / df["inscrits"])
        dfs[k]["tau"] = np.where(
            p == -1, 0,
            np.where(p == 0, 0.5 / df["inscrits"],
            np.where(p == 1, 1 - 0.5 / df["inscrits"],
            np.log(np.where((p > 0) & (p < 1), p / (1 - p), 1)))))

    # 3. distance bins
    distance_bins = np.arange(0, config.p_max + config.dist_step, config.dist_step)

    results = {}

    # 4. loop elections
    for e, df in dfs.items():

        size_bins = np.percentile(df["inscrits"], np.linspace(0, 100, config.num_bins + 1))

        tau_df = pl.DataFrame({"codecommune": df["codecommune"], "tau": df["tau"]})

        df_join = (
            df_pairs
            .join(tau_df, left_on="commune_i", right_on="codecommune")
            .rename({"tau": "tau_i"})
            .join(tau_df, left_on="commune_j", right_on="codecommune")
            .rename({"tau": "tau_j"})
        )

        pdf = df_join.to_pandas()
        pdf["bin"] = pd.cut(pdf["distance"], bins=distance_bins, labels=False)

        results[e] = pdf.groupby("bin").apply(
            lambda g: (g["tau_i"] - g["tau_i"].mean()).corr(g["tau_j"] - g["tau_j"].mean())
        )

    return results


def interpret_election_path(relative_path):
        path_element = relative_path.split(os.sep)[
            -3:
        ]  # ['legislative', '1848', 'leg1848_csv']
        election_type = path_element[0]  # 'legislative'
        year = path_element[1]  # '1848'
        election_code = path_element[2].split("_")[0]  # leg1848
        return election_type, year, election_code


def load_electoral_data():
        """Function that loads all electoral dataset. A schema is defined for all elections."""
        folder_path = os.path.join("s3://arthurmanceau/election_modeling_uhcp/data/" + "raw/", "elections/")
        combined = None
        catalog = {
            "legislative": [],
            "referundum": [],
            "municipales": [],
            "presidentiel": [],
        }
        election_code_mapping = {}
        xs = (
            DataUtils._create_fs()
            if DataUtils._detect_s3("s3://arthurmanceau/election_modeling_uhcp/data/")
            else os
        )
        for root, dirs, files in xs.walk(folder_path):
            for dir in dirs:
                for root, _, files in xs.walk(os.path.join(folder_path, dir)):
                    for file in files:
                        if (
                            file.endswith(".parquet")
                            and (not file.startswith("."))
                        ):
                            election_type, year, election_code = (
                                interpret_election_path(
                                    os.path.relpath(root, folder_path)
                                )
                            )
                            if election_type not in ['legislative', 'presidentiel']:
                                continue

                            logger.debug(
                                f"Processing election : {(election_type, year)}"
                            )

                            catalog[election_type] += [year]
                            election_code_mapping[(election_type, year)] = election_code

                            file_path = DataUtils.path_helper(
                                folder_path, os.path.join(root, file)
                            )
                            df = pl.scan_parquet(file_path).collect()
                            df = df.with_columns(
                                annee=pl.lit(year).cast(pl.Int64),
                                election_code=pl.lit(election_code),
                                election_type=pl.lit(election_type),
                                type=pl.lit(type_encoding[election_type]),
                                codecommune=pl.col("codecommune")
                                .cast(pl.String)
                                .str.zfill(5),
                            )
                            df = df.cast(
                                {
                                    key: value
                                    for key, value in electoral_schema.items()
                                    if key in df.columns
                                }
                            )
                            df = df.match_to_schema(
                                electoral_schema,
                                missing_columns="insert",
                                extra_columns="ignore",
                            )
                            combined = df if combined is None else combined.vstack(df)

        electoral_data = combined.rechunk()

        electoral_data = (
            electoral_data.filter(
                (
                    pl.col("codecommune")
                    .str.slice(0, 2)
                    .is_between(pl.lit("01"), pl.lit("95"))
                    | pl.col("codecommune").str.slice(0, 2).is_in(["2A", "2B"])
                )
            )
            .filter(
                ~pl.all_horizontal(
                    pl.col(
                        ["votants", "exprimes", "inscrits"] + tendances_column_vote
                    ).is_null()
                )
            )
            .with_columns(
                # Zero out cols when inscrits == 0
                [
                    pl.when(pl.col("inscrits") == 0)
                    .then(0)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in ["votants", "exprimes"] + tendances_column_vote
                ]
                + [
                    pl.when(pl.col("inscrits") == 0)
                    .then(pl.lit(True))
                    .otherwise(pl.lit(False))
                    .alias("no_inscrits")
                ]
            )
            .with_columns(
                # Zero out cols when votants == 0
                [
                    pl.when(pl.col("votants") == 0)
                    .then(0)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in ["exprimes"] + tendances_column_vote
                ]
                + [
                    pl.when(pl.col("votants") == 0)
                    .then(pl.lit(True))
                    .otherwise(pl.lit(False))
                    .alias("no_votants")
                ]
            )
            .filter(
                pl.any_horizontal(
                    *[
                        pl.col(c) >= 0
                        for c in tendances_column_vote
                        + ["votants", "exprimes", "inscrits"]
                        if c in electoral_data.columns
                    ]
                )
            )
            .with_columns(
                exprimes_=pl.sum_horizontal(
                    "voteG", "voteCG", "voteC", "voteCD", "voteD"
                )
            )
        )
        assert electoral_data.filter(pl.col("inscrits") < 0).height == 0
        assert electoral_data.filter(pl.col("votants") < 0).height == 0
        assert (
            electoral_data.filter(~(pl.col("election_code").is_in(["muni2026"])))
            .filter(pl.col("inscrits").round(0) < pl.col("votants").round(0))
            .height
            == 0
        )
        assert (
            electoral_data.filter(
                ~(
                    pl.col("election_code").is_in(
                        ["muni2020", "muni2026", "muni2014", "leg2024"]
                    )
                )
            )
            .filter(pl.col("exprimes_").round(0) != pl.col("exprimes").round(0))
            .height
            == 0
        )

        # Creating new columns
        electoral_data = (
            electoral_data.with_columns(
                pvotepar=pl.col("votants") / pl.col("inscrits"),
                abstentions=pl.col("inscrits") - pl.col("votants"),
                pvoteG=pl.col("voteG") / pl.col("exprimes_"),
                pvoteCG=pl.col("voteCG") / pl.col("exprimes_"),
                pvoteC=pl.col("voteC") / pl.col("exprimes_"),
                pvoteCD=pl.col("voteCD") / pl.col("exprimes_"),
                pvoteD=pl.col("voteD") / pl.col("exprimes_"),
                pvoteTG=pl.col("voteTG") / pl.col("exprimes_"),
                pvoteTD=pl.col("voteTD") / pl.col("exprimes_"),
                pvoteGCG=pl.col("voteGCG") / pl.col("exprimes_"),
                pvoteDCD=pl.col("voteDCD") / pl.col("exprimes_"),
                # pvoixOUI=pl.col("voixOUI") / pl.col("exprimes_"),
                # pvoixNON=pl.col("voixNON") / pl.col("exprimes_"),
            )
            .with_columns(
                pvoteabs=pl.col("abstentions") / pl.col("inscrits"),
                tau=(pl.col('pvotepar') / (1 - pl.col('pvotepar'))).log()
            )
            .with_columns(
                # Replace NaN (0/0) and inf (n/0) with 0
                pl.col(tendances_column_pvote + pvote)
                .fill_nan(0)
                .clip(lower_bound=0, upper_bound=1.0)
            )
            .with_columns(pl.col(tendances_column_pvote + pvote).round(2))
        )
        assert (
            electoral_data.filter(
                pl.any_horizontal(
                    *[
                        (pl.col(c) < 0) | (pl.col(c) > 1)
                        for c in tendances_column_pvote + pvote
                    ]
                )
            ).height
            == 0
        )

        return electoral_data, (catalog, election_code_mapping)


def load_communes_data():
        logger.debug("Load communes data")

        # Import communes file from 2026
        communes = pl.scan_csv(
            "s3://arthurmanceau/election_modeling_uhcp/data/raw/insee_geo/2026/communes_2026.csv",
            infer_schema_length=40000,
        ).collect()
        passage = (
            pl.scan_csv(
                "s3://arthurmanceau/election_modeling_uhcp/data/raw/insee_geo/2026/passage_2026.csv",
                infer_schema_length=40000,
                separator=";",
            )
            .drop_nulls()
            .collect()
        )

        # Transformation, add older associated codecommune
        communes = communes.with_columns(
            dep_num=pl.col("DEP").replace({"2A": 20, "2B": 20}).cast(pl.Int64),
            codecommune=pl.col("COM").cast(pl.String).str.zfill(5),
            codecommune_parent=pl.col("COMPARENT").cast(pl.String).str.zfill(5),
        ).join(
            passage.group_by("codecommune_2026").agg(
                all_codecommune_that_existed=pl.col("codecommune_init").implode()
            ),
            left_on="codecommune",
            right_on="codecommune_2026",
            how="left",
        )

        # We keep track of the communes deleted
        associes = (
            communes.select("codecommune", "LIBELLE", "TYPECOM", "COM")
            .group_by("codecommune")
            .agg(
                pl.struct("LIBELLE", "TYPECOM", "COM")
                .implode()
                .alias("COMMUNES_ASSOCIES")
            )
        )

        # Remove duplicated communes that share the same code
        communes = communes.sort(
            "TYPECOM", descending=False
        ).unique(  # We prefer to keep 'COM' that is sorted before 'COMD'
            subset="codecommune", keep="first", maintain_order=True
        )

        # Join back and remove the primary LIBELLE from the list
        communes = (communes.join(associes, on="codecommune", how="left")).sort(
            "codecommune"
        )

        assert communes.filter(pl.col("codecommune").is_duplicated()).height == 0

        # Join geo_data
        geo_data = add_geographical_data()
        communes = (
            communes.join(geo_data, on="codecommune", how="left")
            # Step 1: fill from parent commune
            .join(
                geo_data.select(["codecommune", "lat", "long"]).rename(
                    {"lat": "lat_parent", "long": "long_parent"}
                ),
                left_on="codecommune_parent",
                right_on="codecommune",
                how="left",
            )
            .with_columns(
                lat=pl.col("lat").fill_null(pl.col("lat_parent")),
                long=pl.col("long").fill_null(pl.col("long_parent")),
            )
            .drop(["lat_parent", "long_parent"])
            # Step 2: replace NaN with null so fill_null works on them too
            .with_columns(
                pl.col("lat").fill_nan(None),
                pl.col("long").fill_nan(None),
            )
        )
        return communes


def add_geographical_data():
        """Add geographical coordinates from GeoJSON file"""
        logger.debug("Adding geographical data...")

        geo_data_path = "s3://arthurmanceau/election_modeling_uhcp/data/" + "raw/geo_data/communes.geojson"

        fs = DataUtils._create_fs() if DataUtils._detect_s3(geo_data_path) else None
        if not DataUtils._exists(geo_data_path, fs):
            logger.warning(
                f"GeoJSON file not found at {geo_data_path}. Skipping geographical data."
            )
            return None

        # Load the GeoJSON map of French communes
        communes_geojson = DataLoader.load_geojson(geo_data_path)

        geo_data = []
        for feature in communes_geojson["features"]:
            commune_geom = shape(feature["geometry"])
            lat = float(commune_geom.centroid.x)
            long = float(commune_geom.centroid.y)
            codecommune = feature["properties"]["code"]
            geo_data.append({"codecommune": codecommune, "lat": lat, "long": long})

            # Handle PARIS / LYON / MARSEILLE
            if codecommune == "13055":
                for i in range(1, 16 + 1):
                    if i < 10:
                        geo_data.append(
                            {"codecommune": f"1320{i}", "lat": lat, "long": long}
                        )
                    else:
                        geo_data.append(
                            {"codecommune": f"132{i}", "lat": lat, "long": long}
                        )
            if codecommune == "69123":
                for i in range(1, 9 + 1):
                    geo_data.append(
                        {"codecommune": f"6938{i}", "lat": lat, "long": long}
                    )
            if codecommune == "75056":
                for i in range(1, 20 + 1):
                    if i < 10:
                        geo_data.append(
                            {"codecommune": f"7510{i}", "lat": lat, "long": long}
                        )
                    else:
                        geo_data.append(
                            {"codecommune": f"751{i}", "lat": lat, "long": long}
                        )

        geo_data = pl.DataFrame(geo_data).sort("codecommune")
        geo_data = geo_data.with_columns(
            lat=pl.col("lat").round(4), long=pl.col("long").round(4)
        )
        return geo_data


if __name__ == "__main__":
    # Re-use data processing functionalities
    electoral_data, (catalog, election_code_mapping) = load_electoral_data()
    # Re-compute tau?
    commune_data = load_communes_data()

    df = commune_data.select('codecommune', 'lat', 'long').drop_nulls()
    
    (
        df.lazy()
        .join(df.lazy(), how="cross", suffix="_dest")
        .filter(pl.col("codecommune") < pl.col("codecommune_dest"))
        .with_columns([
            pl.struct(latitude=pl.col("lat"), longitude=pl.col("long")).alias("origin"),
            pl.struct(latitude=pl.col("lat_dest"), longitude=pl.col("long_dest")).alias("destination"),
        ])
        .with_columns(
            pld.col("origin").dist.haversine("destination", "km").alias("distance_km")
        )
        .sink_parquet('s3://arthurmanceau/election_modeling_uhcp/data/derived/spatial_correlations/distance.parquet') # Add timestamp
    )
    