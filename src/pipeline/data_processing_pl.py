import os
from datetime import datetime

import polars as pl
import polars.selectors as cs
from loguru import logger
from shapely.geometry import shape

from src.components.data_processing.data_loader import DataLoader, DataUtils
from src.components.utils.config import DataProcessingConfig
from src.components.utils.read_config import ConfigReader

"""
Objective: improve the data quality by re-designing the pipeline
- Reduce errors
- Be flexible for the integration of new features
- Be faster using polars

The main challenge is that we have communes that change through time.
We make the choice to only consider the communes actives in 2026 (this can be re-run to adapt to the new commune).

TABLE1: commune. This table list all the communes active in 2026, it is the ground truth for code commune, commune name and geographic information. Everything will be a left join from this table
TABLE2: election results. 
"""

# Improvements (data quality):
# - PLM
# - Forward fill for socio-economic data (to have them for more years) and to project for future elections
# - All communes that existed (use passage) : if we can recover features from old communes code

# Revoir la config... implement new config

class ElectionDataProcessor:
    """Class to handle election data processing pipeline.

    Three building blocks:
        1. Election data (reference)
        2. Commune information (yet to be done)
        3. Socio-economic variables

    A clean dataset is built for each election, they are then stacked together
    """
    vote = ['inscrits', "votants", "exprimes", "abstentions"]
    pvote = ['pvotepar', 'pvoteabs']
    tendances = ['G', 'CG', 'C', 'CD', 'D', 'TD', 'TG', 'GCG', 'DCD']
    tendances_column_vote = [f'vote{tendance}' for tendance in tendances]
    tendances_column_pvote = ["p" + col for col in tendances_column_vote]

    def previous(self, cols):
        return [f"previous{col}" for col in cols]

    def previousprevious(self, cols):
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

    def __init__(self):
        """
        Initialize the data processor using the configuration file.
        """
        self.config = ConfigReader._read_config(
            "config/data_processing.json", DataProcessingConfig
        )

    @staticmethod
    def interpret_election_path(relative_path):
        path_element = relative_path.split(os.sep)[
            -3:
        ]  # ['legislative', '1848', 'leg1848_csv']
        election_type = path_element[0]  # 'legislative'
        year = path_element[1]  # '1848'
        election_code = path_element[2].split("_")[0]  # leg1848
        return election_type, year, election_code

    def load_communes_data(self):
        logger.debug("Load communes data")

        # Import communes file from 2026
        communes = pl.scan_csv(
            "s3://arthurmanceau/election_modeling_uhcp/data/raw/insee_geo/2026/communes_2026.csv",
            infer_schema_length=40000,
        ).collect()
        passage = pl.scan_csv('s3://arthurmanceau/election_modeling_uhcp/data/raw/insee_geo/2026/passage_2026.csv',
            infer_schema_length=40000,
            separator=';'
        ).drop_nulls().collect()

        # Transformation, add older associated codecommune
        communes = communes.with_columns(
            dep_num=pl.col("DEP").replace({"2A": 20, "2B": 20}).cast(pl.Int64),
            codecommune=pl.col("COM").cast(pl.String).str.zfill(5),
            codecommune_parent=pl.col("COMPARENT").cast(pl.String).str.zfill(5),
        ).join(
            passage.group_by('codecommune_2026').agg(
                all_codecommune_that_existed=pl.col('codecommune_init').implode()
            ),
            left_on='codecommune',
            right_on='codecommune_2026',
            how='left'
        )

        # We keep track of the communes deleted
        associes = (
            communes.select('codecommune', 'LIBELLE', 'TYPECOM', 'COM')
            .group_by('codecommune')
            .agg(pl.struct('LIBELLE', 'TYPECOM', 'COM').implode().alias('COMMUNES_ASSOCIES'))
        )

        # Remove duplicated communes that share the same code
        communes = (
            communes
            .sort('TYPECOM', descending=False)  # We prefer to keep 'COM' that is sorted before 'COMD'
            .unique(subset='codecommune', keep='first', maintain_order=True)
        )

        # Join back and remove the primary LIBELLE from the list
        communes = (
            communes
            .join(associes, on='codecommune', how='left')
        ).sort('codecommune')

        assert communes.filter(pl.col('codecommune').is_duplicated()).height == 0

        # Join geo_data
        geo_data = self.add_geographical_data()
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

    def add_geographical_data(self):
        """Add geographical coordinates from GeoJSON file"""
        logger.debug("Adding geographical data...")

        geo_data_path = self.config.data_path + "raw/geo_data/communes.geojson"

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

    def load_electoral_data(self):
        """Function that loads all electoral dataset. A schema is defined for all elections."""
        folder_path = os.path.join(self.config.data_path + "raw/", "elections/")
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
            if DataUtils._detect_s3(self.config.data_path)
            else os
        )
        for root, dirs, files in xs.walk(folder_path):
            for dir in dirs:
                for root, _, files in xs.walk(os.path.join(folder_path, dir)):
                    for file in files:
                        if (
                            file.endswith(".parquet")
                            and (not file.startswith("."))
                            and file not in self.config.elections_to_exclude
                        ):
                            election_type, year, election_code = (
                                self.interpret_election_path(
                                    os.path.relpath(root, folder_path)
                                )
                            )
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
                                type=pl.lit(self.type_encoding[election_type]),
                                codecommune=pl.col("codecommune")
                                .cast(pl.String)
                                .str.zfill(5),
                            )
                            df = df.cast(
                                {
                                    key: value
                                    for key, value in self.electoral_schema.items()
                                    if key in df.columns
                                }
                            )
                            df = df.match_to_schema(
                                self.electoral_schema,
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
            .filter(~pl.all_horizontal(pl.col(["votants", "exprimes", 'inscrits'] + self.tendances_column_vote).is_null()))
            .with_columns(
                # Zero out cols when inscrits == 0
                [
                    pl.when(pl.col("inscrits") == 0)
                    .then(0)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in ["votants", "exprimes"] + self.tendances_column_vote
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
                    for c in ["exprimes"] + self.tendances_column_vote
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
                    *[pl.col(c) >= 0 for c in self.tendances_column_vote+["votants", "exprimes", 'inscrits'] if c in electoral_data.columns]
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
            )
            .with_columns(
                # Replace NaN (0/0) and inf (n/0) with 0
                pl.col(self.tendances_column_pvote+self.pvote).fill_nan(0).clip(lower_bound=0, upper_bound=1.0)
            )
            .with_columns(pl.col(self.tendances_column_pvote+self.pvote).round(2))
        )
        assert (
            electoral_data.filter(
                pl.any_horizontal(*[(pl.col(c) < 0) | (pl.col(c) > 1) for c in self.tendances_column_pvote + self.pvote])
            ).height
            == 0
        )

        # Adding previous and previousprevious election results
        lag1_exprs = [
            pl.col(c)
            .shift(1)
            .over("codecommune", "type", order_by="annee")
            .alias(f"previous{c}")
            for c in self.tendances_column_pvote + self.pvote + self.vote + self.tendances_column_vote
        ]

        lag2_exprs = [
            pl.col(c)
            .shift(2)
            .over("codecommune", "type", order_by="annee")
            .alias(f"previousprevious{c}")
            for c in self.tendances_column_pvote + self.pvote + self.vote + self.tendances_column_vote
        ]

        electoral_data = electoral_data.with_columns(lag1_exprs + lag2_exprs)

        return electoral_data, (catalog, election_code_mapping)

    def load_socio_economic_data(self):
        folder_path = os.path.join(self.config.data_path, "raw/")
        combined = None
        xs = (
            DataUtils._create_fs()
            if DataUtils._detect_s3(self.config.data_path)
            else os
        )
        for root, dirs, files in xs.walk(folder_path):
            for dir in dirs:
                if dir != "elections":
                    for root, _, files in xs.walk(os.path.join(folder_path, dir)):
                        for file in files:
                            if file.endswith(".parquet") and (not file.startswith(".")):
                                # Only consider the relevant file : commune level
                                # How to integrate departemental data?
                                key = "codecommune"
                                if "communes" not in file:
                                    continue

                                if file[:5] == "codes":
                                    continue

                                file_path = DataUtils.path_helper(
                                    folder_path, os.path.join(root, file)
                                )
                                data_code = file_path.split("/")[-1].split(".")[0]
                                logger.debug(f"Processing file : {data_code}")

                                df = pl.scan_parquet(file_path).collect()

                                # Can insert projection here....

                                # Identify features (time evolution)
                                all_feature_years = df.select(
                                    cs.matches(r".*\d{4}$")
                                ).columns

                                # Single unpivot for all features
                                df_long = (
                                    df.unpivot(
                                        on=all_feature_years,
                                        index=key,
                                        variable_name="variable",
                                        value_name="raw",
                                    )
                                    .with_columns(
                                        pl.col("raw").cast(pl.Float64, strict=False),
                                        annee=pl.col("variable")
                                        .str.tail(4)
                                        .cast(pl.Int32),
                                        feature_name=pl.col("variable").str.head(-4),
                                    )
                                    .sort([key, "feature_name", "annee"])
                                    .with_columns(
                                        lag=pl.col("raw")
                                        .shift(1)
                                        .over(key, "feature_name"),
                                        rank=pl.col("raw")
                                        .rank("dense")
                                        .over(key, "feature_name"),
                                        delta=pl.col("raw")
                                        .diff(1)
                                        .over(key, "feature_name"),
                                        pct_change=pl.col("raw")
                                        .pct_change(1)
                                        .over(key, "feature_name"),
                                    )
                                    .fill_nan(None)
                                )
                                float_cols = [
                                    c
                                    for c, dtype in zip(df_long.columns, df_long.dtypes)
                                    if dtype in (pl.Float32, pl.Float64)
                                ]
                                df_long = df_long.with_columns(
                                    [
                                        pl.when(pl.col(c).is_infinite())
                                        .then(None)
                                        .otherwise(pl.col(c))
                                        .alias(c)
                                        for c in float_cols
                                    ]
                                )
                                df_long = df_long.cast(
                                    {
                                        key: value
                                        for key, value in self.socio_economic_schema.items()
                                        if key in df_long.columns
                                    }
                                )
                                df_long = df_long.match_to_schema(
                                    self.socio_economic_schema,
                                    missing_columns="insert",
                                    extra_columns="ignore",
                                )
                                combined = (
                                    df_long
                                    if combined is None
                                    else combined.vstack(df_long)
                                )

        socio_economic_data = combined.rechunk().fill_nan(None)
        assert (
            socio_economic_data.drop("codecommune", "feature_name", "variable")
            .select(pl.all().is_nan().sum())
            .sum_horizontal()
            .item()
            == 0
        )
        assert (
            socio_economic_data.drop("codecommune", "feature_name", "variable")
            .select(pl.all().is_infinite().sum())
            .sum_horizontal()
            .item()
            == 0
        )

        catalog = socio_economic_data.group_by("feature_name").agg(
            pl.col("annee").unique().sort()
        )

        return socio_economic_data, catalog

    @staticmethod
    def get_features_for_years(
        X,
        year=2022,
        feature_aug=["raw", "lag", "rank", "delta", "pct_change"],
        n_years_back=0,
    ):
        index_cols = ["codecommune"]

        # Base: pivot for the election year
        data = (
            X.filter(pl.col("annee") == year)
            .pivot(
                on="feature_name",
                index=["annee", "codecommune"],
                values=feature_aug,
                aggregate_function="first",
            )
            .sort("codecommune")
        )

        # Iteratively join each year back
        for i in range(1, n_years_back + 1):
            year_back = year - i
            df_back = (
                X.filter(pl.col("annee") == year_back)
                .pivot(
                    on="feature_name",
                    index=["annee", "codecommune"],
                    values=feature_aug,
                    aggregate_function="first",
                )
                .sort("codecommune")
            )

            new_cols = [
                c
                for c in df_back.columns
                if c not in data.columns and c not in index_cols
            ]

            data = data.join(
                df_back.select(index_cols + new_cols).rename(
                    {c: f"{c}|minus_{i}" for c in new_cols}
                ),
                on=index_cols,
                how="left",
            )

        return data

    def find_features(self, dataset, non_features_cols):
        return [col for col in dataset.columns if col not in non_features_cols]

    def create_dataset(
        self, election_code, electoral_data, commune_data, socio_economic_data
    ):
        # We allow missing values for meta_cols
        all_votes = self.vote + self.pvote + self.tendances_column_vote + self.tendances_column_pvote 
        meta_cols = [
            "dep",
            "nomdep",
            "codecommune",
            "nomcommune",
            "annee",
            "election_type",
            "type",
            "election_code",
            "no_inscrits",
            "no_votants",
            "exprimes_",
            "TYPECOM",
            "NCC",
            "NCCENR",
            "LIBELLE",
            'lat',
            'long'
        ] + all_votes + self.previous(all_votes) + self.previousprevious(all_votes)

        election_results = electoral_data.filter(
            pl.col("election_code") == election_code
        )

        # Merge with commune data
        dataset = election_results.join(
            commune_data.select(
                "TYPECOM", "codecommune", "NCC", "NCCENR", "LIBELLE", "lat", "long",
            ),
            on="codecommune",
            how="left",
        )

        year = int(election_code[-4:])
        X = self.get_features_for_years(
            socio_economic_data,
            year=year,
            feature_aug=["raw", "lag", "rank", "delta", "pct_change"],
            n_years_back=0,
        )
        dataset = dataset.join(X.drop("annee"), on="codecommune", how="left").sort(
            "codecommune"
        )

        # Features are all cols not in metacols
        features = self.find_features(dataset, meta_cols)

        # 1. Features with too many missing values are dropped
        threshold = 0.8
        cols_to_keep = [
            s.name
            for s in dataset.select(features)
            if s.null_count() / dataset.height < threshold
        ]
        cols_dropped = [
            s.name
            for s in dataset.select(features)
            if s.null_count() / dataset.height >= threshold
        ]

        print(f"✅ Kept {len(cols_to_keep)} columns.")
        print(
            f"❌ Dropped {len(cols_dropped)} columns (>= {int(threshold * 100)}% nulls):"
        )
        print(cols_dropped)

        dataset = dataset.drop(cols_dropped)

        # Remaining missing values are imputed with the departemental mean, 
        # and if not the average for the whole country
        features = self.find_features(dataset, meta_cols)
        dataset = dataset.with_columns(
            [
                pl.col(c)
                .fill_null(pl.col(c).mean().over("dep"))
                .fill_null(pl.col(c).mean())  # Fallback to all
                .alias(c)
                for c in features + ['lat', 'long']
            ]
        )

        # This may lead to feature with low variance.
        # We remove features with no variation
        features = self.find_features(dataset, meta_cols)
        std_df = dataset.select(features).std()
        cols_with_variation = [
                col
                for col, std_val in zip(std_df.columns, std_df.row(0))
                if std_val is not None and std_val > 0
        ]
        cols_dropped = [
                col
                for col, std_val in zip(std_df.columns, std_df.row(0))
                if std_val is None or std_val == 0
        ]

        print(f"✅ Kept {len(cols_with_variation)} columns with variation.")
        print(
            f"❌ Dropped {len(cols_dropped)} columns with no variation (std == 0 or null):"
        )
        for col in cols_dropped:
            print(f"   - {col}")

        # Apply the filter
        features = self.find_features(dataset, meta_cols)
        dataset = dataset.select(meta_cols + features)

        # The feature dataset should not contain any missing/inf/nan values
        assert (
            dataset.drop(cs.string() | cs.boolean())
            .select(pl.all().is_nan().sum())
            .sum_horizontal()
            .item()
            == 0
        )
        assert (
            dataset.select(features).drop(cs.string() | cs.boolean())
            .select(pl.all().is_null().sum())
            .sum_horizontal()
            .item()
            == 0
        )
        assert (
            dataset.drop(cs.string() | cs.boolean())
            .select(pl.all().is_infinite().sum())
            .sum_horizontal()
            .item()
            == 0
        )

        # Rename the feature columns for easy identification
        dataset = dataset.select(meta_cols + [pl.col(features).name.prefix("F_")])

        # Create feature dep_num
        dataset = dataset.with_columns(
            dep_num=pl.when(pl.col("codecommune").str.slice(0, 2).is_in(["2A", "2B"]))
                .then(pl.lit(20.0))
                .otherwise(pl.col("codecommune").str.slice(0, 2).cast(pl.Float64, strict=False))
            )
        assert dataset.select('dep_num').min().item() == 1.0
        assert dataset.select('dep_num').max().item() == 95.0

        return dataset

    def save_processed_data(self, agg_data):
        if not DataUtils._detect_s3(self.config.data_path):
            os.makedirs("data/derived/processed/", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_{'_'.join(self.config.vote_variables)}_{self.config.include_elections_after}_{'_'.join(self.config.include_elections_of_type)}_{timestamp}.parquet"

        DataLoader.write_dataset(
            agg_data,
            self.config.data_path + "derived/processed/" + filename,
        )

        logger.success(
            f"All processed data saved to {self.config.data_path + 'derived/processed/'}"
        )


def main():
    """Main function to run the data processing pipeline"""
    processor = ElectionDataProcessor()

    logger.info("Step 1: Electoral data")
    electoral_data, election_catalogs = processor.load_electoral_data()
    election_catalog, election_code_mapping = election_catalogs

    logger.info("Step 2: Commune data")
    commune_data = processor.load_communes_data()

    logger.info("Step 3: Socio-economic data")
    socio_economic_data, feature_catalog = processor.load_socio_economic_data()

    logger.info("Building aggregated training dataset")
    agg_dataset = None
    for election_type in processor.config.include_elections_of_type:
        relevant_years = election_catalog[election_type]
        for year in relevant_years:
            if int(year) >= processor.config.include_elections_after:
                if int(year) <= 2023:
                    election_code = election_code_mapping[(election_type, year)]
                    logger.info(f"Processing: {election_code}")
                    dataset = processor.create_dataset(
                        election_code, electoral_data, commune_data, socio_economic_data
                    )
                    agg_dataset = (
                        dataset
                        if agg_dataset is None
                        else pl.concat([agg_dataset, dataset], how="diagonal")
                    )
    
    agg_dataset = agg_dataset.rechunk()

    # Save to S3
    processor.save_processed_data(agg_dataset.to_pandas())

    return None


if __name__ == "__main__":
    main()
