import os
import re
from datetime import datetime

import polars as pl
import polars.selectors as cs
import polars_distance as pld
from loguru import logger
from shapely.geometry import shape

from electera.components.data_processing.data_loader import DataLoader, DataUtils
from electera.components.utils.config import DataProcessingConfigPl
from electera.components.utils.read_config import ConfigReader

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

    vote = ["inscrits", "votants", "exprimes", "abstentions", "blancsnuls"]
    pvote = ["pvotepar", "pvoteabs", "pvoteblancsnuls"]

    # T2
    voteT2 = ["inscritsT2", "votantsT1", "exprimesT2", "abstentionsT2", "blancsnulsT2"]
    pvoteT2 = ["pvoteparT2", "pvoteabsT2", "pvoteblancsnulsT2"]

    tendances = ["G", "CG", "C", "CD", "D", "TD", "TG", "GCG", "DCD"]
    tendances_column_vote = [f"vote{tendance}" for tendance in tendances]
    tendances_column_pvote = ["p" + col for col in tendances_column_vote]

    # Refs
    refs_tendances = ["OUI", "NON"]
    refs_tendances_column_vote = [f"vote{tendance}" for tendance in refs_tendances]
    refs_tendances_column_pvote = ["p" + col for col in refs_tendances_column_vote]

    def previous(self, cols):
        return [f"previous{col}" for col in cols]

    def previousprevious(self, cols):
        return [f"previousprevious{col}" for col in cols]

    def percentile(self, cols):
        return [f"percentile{col}" for col in cols]

    # For T1 (politiques)
    # For ref : include OUI/NON (other dataset because the checks will be different)
    # For T2 : include T2 var (other dataset because the checks will be different)
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
        "dep": pl.String,  # Key
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
            "../config/data_processing_pl.json", DataProcessingConfigPl
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
            self.config.data_path + "raw/insee_geo/2026/communes_2026.csv",
            infer_schema_length=40000,
        ).collect()
        passage = (
            pl.scan_csv(
                self.config.data_path + "raw/insee_geo/2026/passage_2026.csv",
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
        geo_data = self.add_geographical_data()
        PARIS_LAT = 48.8566
        PARIS_LON = 2.3522
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
            .with_columns(
                # Create struct column for your coordinates
                pl.struct(latitude=pl.col("lat"), longitude=pl.col("long")).alias(
                    "coords"
                )
            )
            .with_columns(
                pl.lit({"latitude": PARIS_LAT, "longitude": PARIS_LON}).alias(
                    "paris_coords"
                )
            )
            .with_columns(
                pld.col("coords")
                .dist.haversine("paris_coords", "km")
                .alias("distanceparis")
            )
            .drop("coords", "paris_coords")
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
                        if file.endswith(".parquet") and (not file.startswith(".")):
                            election_type, year, election_code = (
                                self.interpret_election_path(
                                    os.path.relpath(root, folder_path)
                                )
                            )
                            if election_type not in self.config.elections_type:
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
            .filter(
                ~pl.all_horizontal(
                    pl.col(
                        ["votants", "exprimes", "inscrits"] + self.tendances_column_vote
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
                    *[
                        pl.col(c) >= 0
                        for c in self.tendances_column_vote
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
                blancsnuls=pl.col("votants") - pl.col("exprimes_"),
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
                pvoteblancsnuls=pl.col("blancsnuls") / pl.col("inscrits"),
            )
            .with_columns(
                # Replace NaN (0/0) and inf (n/0) with 0
                pl.col(self.tendances_column_pvote + self.pvote)
                .fill_nan(0)
                .clip(lower_bound=0, upper_bound=1.0)
            )
            .with_columns(pl.col(self.tendances_column_pvote + self.pvote).round(2))
        )
        assert (
            electoral_data.filter(
                pl.any_horizontal(
                    *[
                        (pl.col(c) < 0) | (pl.col(c) > 1)
                        for c in self.tendances_column_pvote + self.pvote
                    ]
                )
            ).height
            == 0
        )

        # Adding ranks
        rank_expr = [
            (pl.col(c).rank() / pl.count(c))
            .round(4)
            .over("type", "annee")
            .alias(f"percentile{c}")
            for c in self.tendances_column_pvote
        ]

        # Adding previous and previousprevious election results
        lag1_exprs = [
            pl.col(c)
            .shift(1)
            .over("codecommune", "type", order_by="annee")
            .alias(f"previous{c}")
            for c in self.tendances_column_pvote
            + self.pvote
            + self.vote
            + self.tendances_column_vote
        ]

        lag2_exprs = [
            pl.col(c)
            .shift(2)
            .over("codecommune", "type", order_by="annee")
            .alias(f"previousprevious{c}")
            for c in self.tendances_column_pvote
            + self.pvote
            + self.vote
            + self.tendances_column_vote
        ]

        electoral_data = electoral_data.with_columns(
            rank_expr + lag1_exprs + lag2_exprs
        )

        return electoral_data, (catalog, election_code_mapping)

    def _process_parquet_file(self, file_path: str, key: str) -> pl.LazyFrame | None:
        source_stem = file_path.split("/")[-1].split(".")[0]
        logger.debug(f"Processing file: {source_stem}")

        lf = pl.scan_parquet(file_path)
        all_cols = lf.collect_schema().names()
        year_cols = [c for c in all_cols if re.search(r"\d{4}$", c)]

        if not year_cols:
            logger.warning(
                f"No feature matching pattern [feature][year] in {source_stem}"
            )
            return None

        df = (
            lf.unpivot(
                index=key,
                on=year_cols,
                variable_name="variable_with_year",
                value_name="raw",
            )
            .with_columns(
                # Extract the 4-digit year suffix
                annee=pl.col("variable_with_year")
                .str.extract(r"(\d{4})$")
                .cast(pl.Int32),
                # Feature
                feature=pl.lit(source_stem)
                + "/"
                + pl.col("variable_with_year").str.replace(r"\d{4}$", ""),
                # Cast value to Float64
                raw=pl.col("raw").cast(pl.Float64, strict=False),
            )
            .drop("variable_with_year")
            .sort([key, "feature", "annee"])
        )

        return df

    def _build_year_grids(self, df: pl.LazyFrame, key: str) -> pl.LazyFrame:
        year_grids = (
            df.select("feature", "annee")
            .unique()
            .group_by("feature")
            .agg(
                pl.int_ranges(pl.col("annee").min(), pl.col("annee").max() + 1).alias(
                    "annee"
                )
            )
            .explode("annee")
        )

        all_keys = df.select(key).unique()
        full_grid = all_keys.join(year_grids, how="cross")

        df = full_grid.join(
            df,
            on=[key, "feature", "annee"],
            how="left",
        )

        # Interpolate missing values linearly within each (key, feature) group
        df = df.sort(key, "feature", "annee").with_columns(
            pl.col("raw").interpolate().over(key, "feature"),
        )

        return df

    def _concat_and_check(self, frames, key):
        df = pl.concat(frames, how="vertical")

        assert (
            df.drop(key, "feature")
            .select(pl.all().is_nan().sum())
            .sum_horizontal()
            .item()
            == 0
        )
        assert (
            df.drop(key, "feature")
            .select(pl.all().is_infinite().sum())
            .sum_horizontal()
            .item()
            == 0
        )

        catalog = df.group_by("feature").agg(pl.col("annee").unique().sort())
        df = df.fill_nan(None)
        return df, catalog

    def _augment(self, df, key):
        return df.with_columns(
            lag=pl.col("raw").shift(1).round(4).over(key, "feature"),
            rank=(pl.col("raw").rank() / pl.count("raw")).round(4).over(key, "feature"),
            delta=pl.col("raw").diff(1).round(4).over(key, "feature"),
            pct_change=pl.col("raw").pct_change(1).round(4).over(key, "feature"),
        ).fill_nan(None)

    def load_socio_economic_data(self):
        folder_path = os.path.join(self.config.data_path, "raw/")
        communes_frames: list[pl.LazyFrame] = []
        dep_frames: list[pl.LazyFrame] = []

        xs = (
            DataUtils._create_fs()
            if DataUtils._detect_s3(self.config.data_path)
            else os
        )

        for root, dirs, files in xs.walk(folder_path):
            for dir_name in dirs:
                if dir_name == "elections":
                    continue
                for sub_root, _, sub_files in xs.walk(
                    os.path.join(folder_path, dir_name)
                ):
                    for file in sub_files:
                        # Excluding some files
                        # Should a a parquet file
                        if not file.endswith(".parquet"):
                            continue
                        # Should not be a hidden file
                        if file.startswith("."):
                            continue
                        if file[:5] == "codes":
                            continue

                        # Determine geographic level
                        if "communes" in file:
                            key = "codecommune"
                        elif "departements" in file:
                            key = "dep"
                        else:
                            continue

                        file_path = DataUtils.path_helper(
                            folder_path, os.path.join(sub_root, file)
                        )

                        df = self._process_parquet_file(file_path, key)
                        if df is None:
                            continue

                        # Build year grids and fill gaps (linear interpolation)
                        df = self._build_year_grids(df, key)

                        # Augmentations
                        df = self._augment(df, key)

                        target_schema = {
                            key: pl.String,
                            "feature": pl.String,
                            "annee": pl.Int64,
                            "raw": pl.Float64,
                            "lag": pl.Float64,
                            "rank": pl.Float64,
                            "delta": pl.Float64,
                            "pct_change": pl.Float64,
                        }
                        schema = df.collect_schema()
                        df = df.cast(
                            {
                                k: v
                                for k, v in target_schema.items()
                                if k in schema.names()
                            },
                            strict=True,
                        ).match_to_schema(
                            target_schema,
                            missing_columns="insert",
                            extra_columns="ignore",
                        )
                        float_cols = [
                            c
                            for c, dtype in zip(schema.names(), schema.dtypes())
                            if dtype in (pl.Float32, pl.Float64)
                        ]
                        df = df.with_columns(
                            [
                                pl.when(pl.col(c).is_infinite())
                                .then(None)
                                .otherwise(pl.col(c))
                                .alias(c)
                                for c in float_cols
                            ]
                        )
                        df = df.collect()

                        assert (
                            df.select(cs.float().is_nan().sum()).sum_horizontal().item()
                            == 0
                        )
                        assert (
                            df.select(cs.float().is_infinite().sum())
                            .sum_horizontal()
                            .item()
                            == 0
                        )
                        assert (
                            df.select(key, "feature", "annee")
                            .select(pl.all().is_null().sum())
                            .sum_horizontal()
                            .item()
                            == 0
                        )

                        if key == "codecommune":
                            communes_frames.append(df)
                        else:
                            dep_frames.append(df)

        socio_economic_communes, catalog_communes = self._concat_and_check(
            communes_frames, "codecommune"
        )
        socio_economic_dep, catalog_dep = self._concat_and_check(dep_frames, "dep")

        return (
            socio_economic_communes,
            socio_economic_dep,
            catalog_communes,
            catalog_dep,
        )

    # def load_socio_economic_data_(self):
    #     folder_path = os.path.join(self.config.data_path, "raw/")
    #     communes_frames = []
    #     dep_frames = []
    #     xs = (
    #         DataUtils._create_fs()
    #         if DataUtils._detect_s3(self.config.data_path)
    #         else os
    #     )
    #     for root, dirs, files in xs.walk(folder_path):
    #         for dir in dirs:
    #             if dir != "elections":
    #                 for root, _, files in xs.walk(os.path.join(folder_path, dir)):
    #                     for file in files:
    #                         if file.endswith(".parquet") and (not file.startswith(".")):
    #                             if "communes" in file:
    #                                 key = "codecommune"
    #                             elif "departements" in file:
    #                                 key = "dep"
    #                             else:
    #                                 continue

    #                             if file[:5] == "codes":
    #                                 continue

    #                             file_path = DataUtils.path_helper(
    #                                 folder_path, os.path.join(root, file)
    #                             )
    #                             data_code = file_path.split("/")[-1].split(".")[0]
    #                             logger.debug(f"Processing file : {data_code}")

    #                             df = pl.scan_parquet(file_path).collect()
    #                             HIDDEN_RE_PY = (
    #                                 r"[\x00-\x1F\x7F\u200B\u200C\u200D\uFEFF]"
    #                             )
    #                             df = df.rename(
    #                                 {c: re.sub(HIDDEN_RE_PY, "", c) for c in df.columns}
    #                             )

    #                             # 1. Identify features (time evolution)
    #                             all_feature_years = df.select(
    #                                 cs.matches(r".*\d{4}$")
    #                             ).columns

    #                             if len(all_feature_years) == 0:
    #                                 logger.warning(
    #                                     "No feature matching the pattern [feature][year]"
    #                                 )
    #                                 continue

    #                             feature_years = {}
    #                             for c in all_feature_years:
    #                                 feature = c[:-4]
    #                                 year = int(c[-4:])
    #                                 feature_years.setdefault(feature, []).append(year)

    #                             feature_years = {
    #                                 k: sorted(v) for k, v in feature_years.items()
    #                             }
    #                             logger.debug(f"Features: {feature_years}")

    #                             # 2. Convert to long format
    #                             df_long = (
    #                                 df.unpivot(
    #                                     on=df.select(cs.matches(r".*\d{4}$")).columns,
    #                                     index=key,
    #                                     variable_name="variable",
    #                                     value_name="raw",
    #                                 )
    #                                 .with_columns(
    #                                     raw=pl.col("raw").cast(
    #                                         pl.Float64, strict=False
    #                                     ),
    #                                     annee=pl.col("variable")
    #                                     .str.tail(4)
    #                                     .cast(pl.Int64),
    #                                 )
    #                                 .with_columns(
    #                                     variable=(
    #                                         pl.lit(file).str.replace(r"\.parquet$", "")
    #                                         + "/"
    #                                         + pl.col("variable")
    #                                     ),
    #                                     feature_name=(
    #                                         pl.lit(file).str.replace(r"\.parquet$", "")
    #                                         + "/"
    #                                         + pl.col("variable").str.head(-4)
    #                                     ),
    #                                 )
    #                                 .sort([key, "feature_name", "annee"])
    #                             )
    #                             print(repr(file))

    #                             # 3. Linear interpolation for missing years
    #                             year_grids = pl.concat(
    #                                 [
    #                                     pl.DataFrame(
    #                                         {
    #                                             "feature_name": re.sub(
    #                                                 r"\.parquet$", "", file
    #                                             )
    #                                             + "/"
    #                                             + feature,
    #                                             "annee": list(
    #                                                 range(min(years), max(years) + 1)
    #                                             ),
    #                                         }
    #                                     )
    #                                     for feature, years in feature_years.items()
    #                                 ]
    #                             )

    #                             df_full = (
    #                                 df_long.lazy()
    #                                 # Get unique (codecommune, feature_name) combos
    #                                 .select(key, "feature_name")
    #                                 .unique()
    #                                 # Join with the per-feature year grid
    #                                 .join(
    #                                     year_grids.lazy(), on="feature_name", how="left"
    #                                 )
    #                                 # Join back original data
    #                                 .join(
    #                                     df_long.lazy(),
    #                                     on=[key, "feature_name", "annee"],
    #                                     how="left",
    #                                 )
    #                                 .sort([key, "feature_name", "annee"])
    #                                 .with_columns(
    #                                     pl.col("raw")
    #                                     .interpolate()
    #                                     .over(key, "feature_name")
    #                                 )
    #                                 .with_columns(
    #                                     variable=pl.concat_str(
    #                                         pl.col("feature_name"),
    #                                         pl.col("annee").cast(pl.String),
    #                                         separator="",
    #                                     )
    #                                 )
    #                                 .collect()
    #                             )

    #                             # 4. Augmentations
    #                             df_full = df_full.with_columns(
    #                                 lag=pl.col("raw")
    #                                 .shift(1)
    #                                 .round(4)
    #                                 .over(key, "feature_name"),
    #                                 rank=(pl.col("raw").rank() / pl.count("raw"))
    #                                 .round(4)
    #                                 .over(key, "feature_name"),
    #                                 delta=pl.col("raw")
    #                                 .diff(1)
    #                                 .round(4)
    #                                 .over(key, "feature_name"),
    #                                 pct_change=pl.col("raw")
    #                                 .pct_change(1)
    #                                 .round(4)
    #                                 .over(key, "feature_name"),
    #                             ).fill_nan(None)

    #                             # 5. Sanity checks
    #                             float_cols = [
    #                                 c
    #                                 for c, dtype in zip(df_full.columns, df_full.dtypes)
    #                                 if dtype in (pl.Float32, pl.Float64)
    #                             ]
    #                             df_full = df_full.with_columns(
    #                                 [
    #                                     pl.when(pl.col(c).is_infinite())
    #                                     .then(None)
    #                                     .otherwise(pl.col(c))
    #                                     .alias(c)
    #                                     for c in float_cols
    #                                 ]
    #                             )
    #                             df_full = df_full.cast(
    #                                 {
    #                                     key: value
    #                                     for key, value in self.socio_economic_schema.items()
    #                                     if key in df_full.columns
    #                                 }
    #                             )
    #                             df_full = df_full.match_to_schema(
    #                                 self.socio_economic_schema,
    #                                 missing_columns="insert",
    #                                 extra_columns="ignore",
    #                             )
    #                             # includes NUL + all ASCII control chars + BOM/zero-width
    #                             HIDDEN_RE = r"[\x00-\x1F\x7F\u200B\u200C\u200D\uFEFF]"

    #                             def hidden_counts(df: pl.DataFrame, label: str) -> None:
    #                                 str_cols = [
    #                                     c
    #                                     for c, t in zip(df.columns, df.dtypes)
    #                                     if t == pl.String
    #                                 ]
    #                                 if not str_cols:
    #                                     print(f"{label}: no string cols")
    #                                     return
    #                                 out = df.select(
    #                                     [
    #                                         pl.col(c)
    #                                         .fill_null("")
    #                                         .str.contains(HIDDEN_RE)
    #                                         .sum()
    #                                         .alias(c)
    #                                         for c in str_cols
    #                                     ]
    #                                 )
    #                                 print(f"\n[{label}]")
    #                                 print(out)
    #                                 return out

    #                             # call after each stage
    #                             h1 = hidden_counts(df, "raw parquet")
    #                             if h1.sum_horizontal().item() > 0:
    #                                 breakpoint()
    #                             h2 = hidden_counts(df_long, "after unpivot")
    #                             if h2.sum_horizontal().item() > 0:
    #                                 breakpoint()
    #                             h3 = hidden_counts(df_full, "final before csv")
    #                             if h3.sum_horizontal().item() > 0:
    #                                 breakpoint()
    #                             h4 = hidden_counts(year_grids, "yg")
    #                             if h4.sum_horizontal().item() > 0:
    #                                 breakpoint()

    #                             # 6. Append to results
    #                             if key == "codecommune":
    #                                 communes_frames.append(df_full)
    #                             elif key == "dep":
    #                                 dep_frames.append(df_full)

    #                             # if key == "codecommune":
    #                             #     combined_communes = (
    #                             #         df_full
    #                             #         if combined_communes is None
    #                             #         else combined_communes.vstack(df_full)
    #                             #     )
    #                             # elif key == "dep":
    #                             #     combined_dep = (
    #                             #         df_full
    #                             #         if combined_dep is None
    #                             #         else combined_dep.vstack(df_full)
    #                             #     )

    #     # socio_economic_data_communes = combined_communes.rechunk().fill_nan(None)
    #     # hn = hidden_counts(socio_economic_data_communes, "after all")
    #     socio_economic_data_communes = pl.concat(
    #         communes_frames, how="vertical"
    #     ).fill_nan(None)
    #     socio_economic_data_dep = pl.concat(dep_frames, how="vertical").fill_nan(None)
    #     # socio_economic_data_dep = combined_dep.rechunk().fill_nan(None)
    #     breakpoint()
    #     for socio_economic_data in [
    #         socio_economic_data_communes,
    #         socio_economic_data_dep,
    #     ]:
    #         assert (
    #             socio_economic_data.drop(
    #                 "codecommune", "dep", "feature_name", "variable"
    #             )
    #             .select(pl.all().is_nan().sum())
    #             .sum_horizontal()
    #             .item()
    #             == 0
    #         )
    #         assert (
    #             socio_economic_data.drop(
    #                 "codecommune", "dep", "feature_name", "variable"
    #             )
    #             .select(pl.all().is_infinite().sum())
    #             .sum_horizontal()
    #             .item()
    #             == 0
    #         )

    #     catalog_communes = socio_economic_data.group_by("feature_name").agg(
    #         pl.col("annee").unique().sort()
    #     )
    #     catalog_dep = socio_economic_data_dep.group_by("feature_name").agg(
    #         pl.col("annee").unique().sort()
    #     )

    #     return (
    #         socio_economic_data_communes,
    #         socio_economic_data_dep,
    #         catalog_communes,
    #         catalog_dep,
    #     )

    @staticmethod
    def get_features_for_years(
        X_communes,
        X_dep,
        year=2022,
        feature_aug=["raw", "lag", "rank", "delta", "pct_change"],
    ):
        # index_cols = ["codecommune"]
        # Base: pivot for the election year
        data_communes = (
            X_communes.filter(pl.col("annee") == year)
            .pivot(
                on="feature",
                index=["annee", "codecommune"],
                values=feature_aug,
                aggregate_function="first",
            )
            .sort("codecommune")
            .with_columns(dep=pl.col("codecommune").str.slice(0, 2).alias("dep"))
        )
        data_dep = (
            X_dep.filter(pl.col("annee") == year)
            .pivot(
                on="feature",
                index=["annee", "dep"],
                values=feature_aug,
                aggregate_function="first",
            )
            .sort("dep")
        )
        data = data_communes.join(data_dep, on="dep", validate="m:1", coalesce=True)

        return data.drop("annee_right", "dep")

    def find_features(self, dataset, non_features_cols):
        return [col for col in dataset.columns if col not in non_features_cols]

    def create_dataset(
        self,
        election_code,
        electoral_data,
        commune_data,
        socio_economic_data_communes,
        socio_economic_data_dep,
    ):
        # We allow missing values for meta_cols
        all_votes = (
            self.vote
            + self.pvote
            + self.tendances_column_vote
            + self.tendances_column_pvote
        )
        meta_cols = (
            [
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
                "lat",
                "long",
                "distanceparis",
            ]
            + all_votes
            + self.percentile(self.tendances_column_pvote)
            + self.previous(all_votes)
            + self.previousprevious(all_votes)
        )

        election_results = electoral_data.filter(
            pl.col("election_code") == election_code
        )

        # Merge with commune data
        dataset = election_results.join(
            commune_data.select(
                "TYPECOM",
                "codecommune",
                "NCC",
                "NCCENR",
                "LIBELLE",
                "lat",
                "long",
            ),
            on="codecommune",
            how="left",
        )

        year = int(election_code[-4:])
        X = self.get_features_for_years(
            socio_economic_data_communes,
            socio_economic_data_dep,
            year=year,
            feature_aug=["raw", "lag", "rank", "delta", "pct_change"],
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
                for c in features + ["lat", "long"]
            ]
        )

        # This may lead to feature with low variance.
        # We remove features with no variation
        features = self.find_features(dataset, meta_cols)
        if len(features) > 0:
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
            dataset.select(features)
            .drop(cs.string() | cs.boolean())
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
            .otherwise(
                pl.col("codecommune").str.slice(0, 2).cast(pl.Float64, strict=False)
            )
        )
        assert dataset.select("dep_num").min().item() >= 1.0
        assert dataset.select("dep_num").max().item() <= 95.0

        return dataset

    def save_processed_data(self, agg_data):
        if not DataUtils._detect_s3(self.config.data_path):
            os.makedirs("data/derived/processed/", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_processed_{'_'.join(self.config.elections_type)}_from{self.config.year_inf}_to{self.config.year_sup}_{timestamp}.parquet"

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
    (
        socio_economic_data_communes,
        socio_economic_data_dep,
        catalog_communes,
        catalog_dep,
    ) = processor.load_socio_economic_data()

    logger.info("Building aggregated training dataset")
    agg_dataset = None
    for election_type in processor.config.elections_type:
        relevant_years = election_catalog[election_type]
        for year in relevant_years:
            if processor.config.year_inf <= int(year) <= processor.config.year_sup:
                election_code = election_code_mapping[(election_type, year)]
                logger.info(f"Processing: {election_code}")
                dataset = processor.create_dataset(
                    election_code,
                    electoral_data,
                    commune_data,
                    socio_economic_data_communes,
                    socio_economic_data_dep,
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
