"""
Pipeline 1: Data Processing
============================
This module handles all data loading, preprocessing, feature engineering, and dataset preparation
for the election modeling project.
"""

import gc
import os
import re
from datetime import datetime

import pandas as pd
import polars as pl
from loguru import logger
from shapely.geometry import shape

from src.components.data_processing.data_loader import DataLoader, DataUtils
from src.components.data_processing.projections_module import (
    ProjectionModel,
    ProjectionUtils,
)
from src.components.utils.config import DataProcessingConfig
from src.components.utils.read_config import ConfigReader

# TODO: encoding of dep columns with something related to the density of habitation
# and encoding of the year with the gini index for France at that time
# or political division indicator ?
# Add polling data to capture what the model can't
# Encoding of DEP


class ElectionDataProcessor:
    """Class to handle election data processing pipeline"""

    def __init__(self):
        """
        Initialize the data processor using the configuration file.
        """
        self.config = ConfigReader._read_config(
            "config/data_processing.json", DataProcessingConfig
        )

        # Data containers
        self.dfs = None  # Electoral data
        self.dfc = None  # Socio-economic data
        self.global_dataset = None  # Final merged dataset

    def load_electoral_data(self):
        """Load and process electoral data from parquet files"""

        logger.info("Loading electoral data...")
        logger.debug(
            f"Vote statistics (targets) will be {' '.join(self.config.vote_variables)}"
        )

        folder_path = os.path.join(self.config.data_path + "raw/", "elections")
        self.dfs = pd.DataFrame(columns=["codecommune"])
        self.election_included = []

        # Iterate through all elections to collect vote statistics
        # TODO: replace by a generator
        xs = (
            DataUtils._create_fs()
            if DataUtils._detect_s3(self.config.data_path)
            else os
        )
        for root, dirs, files in xs.walk(folder_path):
            for file in files:
                if (
                    file.endswith(".parquet")
                    and (not file.startswith("."))
                    and file not in self.config.elections_to_exclude
                ):
                    path_element = os.path.relpath(root, folder_path).split(os.sep)[
                        -3:
                    ]  # ['legislative', '1848', 'leg1848_csv']
                    election_type = path_element[0]  # 'legislative'
                    year = path_element[1]  # '1848'
                    self.election_included.append((election_type, year))
                    logger.info(f"Processing election : {(election_type, year)}")

                    file_path = DataUtils.path_helper(
                        folder_path, os.path.join(root, file)
                    )
                    df = DataLoader.load_dataset(file_path=file_path)
                    df["codecommune"] = df["codecommune"].astype(str)

                    X = df[["codecommune", "inscrits"] + self.config.vote_variables]
                    for var in self.config.vote_variables:
                        X = X.rename(columns={var: f"pvote{var}{year}-{election_type}"})
                    X = X.rename(
                        columns={"inscrits": f"inscrits{year}-{election_type}"}
                    )
                    # Each time we fetch the target AND inscrits as they will be features
                    # of the model (previous elections)
                    self.dfs = pd.merge(self.dfs, X, on="codecommune", how="outer")

        # We find some mismatch with the INSEE code communes (file to check available at : )
        # LA RÉPARA-AURIPLES 26383 -> 26020
        # . for PARIS ? | 69380 for LYON but very few inscrits
        # 07350 ANNONAY SUD (ANCIEN NR INSEE)
        # 51700..713 commune 700...712
        # We remove these weird cases...

        # Mayotte is excluded from the data

        codes_to_remove = [
            "26383",
            "69380",
            ".",
            "07350",
            "51700",
            "51701",
            "51702",
            "51703",
            "51704",
            "51705",
            "51706",
            "51707",
            "51708",
            "51709",
            "51710",
            "51711",
            "51712",
            "51713",
        ]
        logger.warning(
            f"Following commune are excluded due to data inconstency: {codes_to_remove}"
        )

        self.dfs = self.dfs[~self.dfs["codecommune"].isin(codes_to_remove)]

        logger.success(f"Electoral data loaded: {self.dfs.shape}")

    def _add_year_to_file(self, df, file_name, year):
        exclude_columns = [
            "dep",
            "nomdep",
            "codecommune",
            "nomcommune",
        ]
        columns_to_rename = [col for col in df.columns if col not in exclude_columns]
        df = df.rename(columns={col: f"{col}_{year}" for col in columns_to_rename})
        logger.warning(f"Add year {year} after {file_name}")
        return df

    def load_socioeconomic_data(
        self, relative_path_cache_file="derived/cache/dfc_cached.parquet"
    ):
        """
        Load and process socio-economic data from parquet files
        """
        logger.info("Loading socio-economic data...")

        # Load cache if available
        cache_file_path = self.config.data_path + relative_path_cache_file
        fs = DataUtils._create_fs() if DataUtils._detect_s3(cache_file_path) else None
        if DataUtils._exists(file_path=cache_file_path, fs=fs):
            logger.debug("Loading cached data")
            self.dfc = DataLoader.load_dataset(cache_file_path)

        else:
            # Initialize empty dataframe
            self.dfc = pd.DataFrame(columns=["codecommune", "dep"])
            files_not_used = []

            # Iterate through all files in the folder
            xs = (
                DataUtils._create_fs()
                if DataUtils._detect_s3(self.config.data_path)
                else os
            )
            for root, dirs, files in xs.walk(self.config.data_path):
                for file in files:
                    # We are going to deal with the file that contains information
                    # at the "communes" level.
                    # They should contain "communes" in the file name
                    # codecommunes files are matching files and contains no features
                    if (
                        file.endswith(".parquet")
                        and (("communes" in file) or ("departements" in file))
                        and (not file.startswith("."))
                    ):

                        # We exclude the files of matching codecommunes / cantons
                        if "code" in file:
                            logger.info(
                                f"File {file} was not loaded because it didn't \
                                contain relevant features (code mapping file)"
                            )
                            continue

                        logger.info(f"Processing: {file[:-len('.parquet')]}")

                        file_path = DataUtils.path_helper(
                            self.config.data_path, os.path.join(root, file)
                        )
                        df = DataLoader.load_dataset(file_path=file_path)

                        # Handle types
                        object_cols = df.select_dtypes(include=["object"]).columns
                        df[object_cols] = df[object_cols].astype(str)

                        other_cols = df.columns.difference(object_cols)
                        df[other_cols] = df[other_cols].astype(float)

                        # For some files specific to one year, we may have suffix after the year.
                        # We add the suffix before the year
                        if bool(re.search(r"(\d{4})$", file[: -len(".parquet")])):
                            df.columns = df.columns.map(
                                lambda col: (
                                    re.sub(r"(\d{4})(?!$)", "", col)
                                    + re.search(r"(\d{4})(?!$)", col).group(1)
                                    if re.search(r"(\d{4})(?!$)", col)
                                    and not re.search(r"(\d{4})$", col)
                                    else col
                                )
                            )
                            logger.warning(
                                f"The data file {file} is specific to one year. We add the suffix before the year to match the other features."
                            )

                        # menagescommunes is only available for 1975
                        if file == "menagescommunes.parquet":
                            df = self._add_year_to_file(
                                df, "menagescommunes.parquet", "1975"
                            )

                        # terrescommunes is only available for 1968
                        if file == "terrescommunes.parquet":
                            df = self._add_year_to_file(
                                df, "terrescommunes.parquet", "1968"
                            )

                        # In all files, features columns are present
                        # under the following form {feature}{year}.
                        features_with_years = [
                            col for col in df.columns if re.search(r"(\d{4})$", col)
                        ]

                        # We add commune to the file (merge)
                        if "departements" in file:
                            commune_df = self.dfc[["codecommune", "dep"]]
                            df = pd.merge(
                                commune_df, df, on="dep", how="left", validate="m:1"
                            )

                        selected_columns = ["dep", "codecommune"] + [
                            f"{feature}" for feature in features_with_years
                        ]
                        Y = df[selected_columns]

                        # We rename the columns of Y to account for the origin of the feature
                        # (and add if it is a departemental average)
                        Y.columns = [
                            (
                                f"{file[:-len('.parquet')]}/{col}"
                                if col not in ["codecommune", "dep"]
                                else col
                            )
                            for col in Y.columns
                        ]

                        # Weird behavior
                        if file == "rsacommunes.parquet":
                            Y = Y.copy()
                            Y["codecommune"] = Y["codecommune"].astype(int).astype(str)
                            Y["dep"] = Y["dep"].astype(int).astype(str)

                        # Append the dataframe to the list
                        self.dfc = pd.merge(
                            self.dfc,
                            Y,
                            on=["codecommune", "dep"],
                            how="outer",
                            validate="1:1",
                        )

                    else:
                        files_not_used.append(file)

            logger.warning(f"The following files were not considered {files_not_used}")
            # Some old commune code will be added to the dataset
            # because they are referenced in data files.
            # We remove them, the truth for codecommune being the election files.
            communes_socio_eco_data = set(self.dfc["codecommune"].dropna().unique())
            communes_election_data = set(self.dfs["codecommune"].dropna().unique())
            to_remove = communes_socio_eco_data - communes_election_data
            self.dfc = self.dfc[~self.dfc["codecommune"].isin(to_remove)]
            logger.warning(
                f"Old commune removed from dataset (number of communes removed: {len(to_remove)})"
            )
            # We check there is not duplicated communes in self.dfc (very important)
            duplicate_count = self.dfc.duplicated(subset=["codecommune"]).sum()
            if duplicate_count > 0:
                logger.warning(
                    f"Found {duplicate_count} duplicate rows based on codecommune. Keeping first occurrence and dropping {duplicate_count} duplicates."
                )
                self.dfc.drop_duplicates(
                    subset=["codecommune"], keep="first", inplace=True
                )

            # Optionally save to cache if it doesn't exist
            fs = (
                DataUtils._create_fs()
                if DataUtils._detect_s3(cache_file_path)
                else None
            )
            if not DataUtils._exists(file_path=cache_file_path, fs=fs):
                if not DataUtils._detect_s3(cache_file_path):
                    os.makedirs(
                        os.path.dirname(
                            self.config.data_path + relative_path_cache_file
                        ),
                        exist_ok=True,
                    )
                DataLoader.write_dataset(
                    data=self.dfc,
                    file_path=self.config.data_path + relative_path_cache_file,
                )
                logger.info(f"Cached data saved to {relative_path_cache_file}")

        logger.success(f"Socio-economic data loaded: {self.dfc.shape}")
        fl = list(
            set([col[:-4] for col in self.dfc.columns if re.search(r"\d{4}$", col)])
        )
        logger.info(f"Features available: (#{len(fl)})")

    def project_socioeconomic_data(self):
        """
        Projection of socio-economic features based on a projection model.
        """
        if not self.config.projections:
            logger.info("Skipping projections of socioeconomic data")
            return None

        # Hardcoded parameters
        reference_year = 2022
        target_year = 2027

        logger.info(
            f"Projection of socio-economic features from {reference_year} until {target_year}"
        )

        model = ProjectionModel(p=5, alpha=0.2)
        model.fit(data=self.dfc)
        new_columns = {}

        features_list = (
            self.dfc.columns.str.replace(r"\d{4}$", "", regex=True).unique().to_list()
        )
        not_projected_features, projected_features = [], []
        features_list.remove("codecommune")
        features_list.remove("dep")
        for feature in features_list:
            last_year = ProjectionUtils.find_last_year_of(feature, self.dfc)
            freq = ProjectionUtils.find_freq_of(feature, self.dfc)
            if (freq != 1) or (last_year != 2022):
                not_projected_features.append(feature)
            else:
                projected_features.append(feature)
                new_columns = {
                    **new_columns,
                    **model.predict_linear(
                        start=reference_year + 1,
                        end=target_year,
                        feature=feature,
                        reference_year=reference_year,
                    ),
                }

        df_new = pd.DataFrame(new_columns, index=self.dfc.index)
        self.dfc = pd.concat([self.dfc, df_new], axis=1)

        logger.success(
            f"Projection of {len(projected_features)} features completed : {self.dfc.shape}"
        )

        logger.success(
            f"Adding to the vote datasets dummy election results for 2027 {self.config.include_elections_of_type}"
        )

        dummy_election_columns = []
        for election_type in self.config.include_elections_of_type:
            dummy_election_columns += [
                f"inscrits2027-{election_type}",
                f"pvoteppar2027-{election_type}",
                f"pvotepvoteG2027-{election_type}",
                f"pvotepvoteC2027-{election_type}",
                f"pvotepvoteD2027-{election_type}",
                f"pvotepvoteCG2027-{election_type}",
                f"pvotepvoteCD2027-{election_type}",
            ]

        for col in dummy_election_columns:
            self.dfs[col] = 0.0

        logger.success(
            f"Successfully added dummy election results for 2027 {self.config.include_elections_of_type} : {self.dfs.shape}"
        )

    def _find_feat_and_year(self, feature):
        year = re.search(r"(\d{4})$", feature)
        if year is not None:
            year = year.group(1)
        else:
            return None, None
        feat = re.sub(r"(\d{4})$", "", feature)
        return feat, year

    def apply_feature_engineering(self):
        """Apply feature engineering to create rank and delta features"""

        if self.config.features_aug == []:
            logger.info("Skipping feature engineering.")
            return None

        logger.info("Applying feature engineering...")
        logger.debug(f"Augmentation selected: {self.config.features_aug}")

        features_all = list(
            set([col for col in self.dfc.columns if re.search(r"\d{4}$", col)])
        )
        new_columns = {aug: {} for aug in self.config.features_aug}

        aug_log = {
            "rank": lambda x, y: x.rank(pct=True),
            "winsor": lambda x, y: x.clip(
                lower=x.quantile(0.01), upper=x.quantile(0.99)
            ),
            "delta": lambda x, y: x - y,
            "lag": lambda x, y: y,
            "pct_change": lambda x, y: (x - y / x),
        }

        current_feature_fam = features_all[0].split("/")[0]
        logger.info(f"Processing features in: {current_feature_fam}")
        features_all = sorted(features_all)
        for feature in features_all:
            if feature in ["codecommune", "dep"]:
                continue

            feature_fam = feature.split("/")[0]
            if feature_fam != current_feature_fam:
                logger.info(f"Processing features in: {feature_fam}")
                current_feature_fam = feature_fam

            previous_feature = self._find_previous_feature(feature, features_all)
            feat, year = self._find_feat_and_year(feature)

            for aug in self.config.features_aug:

                if previous_feature is None:
                    logger.debug(f"No previous feature was found for {feature}.")
                    continue
                if isinstance(self.dfc[feature].iloc[0], str):
                    logger.warning(
                        f"{feature} is not a numerical feature. Check why, skipping for now."
                    )
                    continue

                new_columns[aug][f"{feat}_{aug}{year}"] = aug_log[aug](
                    x=self.dfc[feature].astype(float),
                    y=self.dfc[previous_feature].astype(float),
                )

        # dfc_aug = self.dfc

        pl_target = pl.from_pandas(self.dfc)
        # Provision 100GB per augmentation...

        for aug in self.config.features_aug:
            logger.info(f"Adding to the dataframe the feature augmentation: {aug}")

            pl_target = pl_target.with_columns(
                pl.from_pandas(pd.DataFrame(new_columns[aug]))
            )
            del new_columns[aug]
            gc.collect()

            # join based on index
            # dfc_aug = dfc_aug.join(pd.DataFrame(new_columns[aug]), how="left", validate="1:1")
            # Free up memory
            # del new_columns[aug]

        self.dfc = pl_target.to_pandas()
        # self.dfc = dfc_aug
        logger.success(f"Feature engineering completed: {self.dfc.shape}")

    def apply_quality_filter(self):
        """
        We apply a quality filter to remove low-quality features.
        For each feature, we count the number of missing values and if it exceeds more than half
        the total number of rows, we drop the feature.
        """
        if not self.config.quality_filter:
            logger.info("Skipping quality filter")
            return None

        missing_values = self.global_dataset.isna().sum()
        high_missing = missing_values[
            missing_values > self.global_dataset.shape[0] / 2
        ].index
        self.global_dataset.drop(columns=high_missing, inplace=True)
        logger.warning(
            f"The following features are removed (missing for more than half of the rows): {high_missing.tolist()}"
        )
        logger.info(f"Shape of the dataset is now: {self.global_dataset.shape}")
        logger.success("Quality filter applied")

    def add_geographical_data(self):
        """Add geographical coordinates from GeoJSON file"""
        logger.info("Adding geographical data...")

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

        geo_df = pd.DataFrame(geo_data)
        self.dfc = pd.merge(self.dfc, geo_df, on="codecommune", how="left")

        # We have 11 communes where geo data is missing because they were created recently.
        # We have 1453 communes that were deleted or merged (recently) where geo data is missing .

        # We input the geo data of the first commune of the departement...
        logger.warning(
            "Commune with no geodata will be imputed with the geo data of the departement (first commune of the departement)"
        )
        missing = self.dfc[self.dfc["lat"].isna()]["codecommune"]
        missing_deps = missing.str[:2]
        geo_mapping = (
            geo_df.groupby(geo_df["codecommune"].str[:2])
            .first()[["lat", "long"]]
            .to_dict(orient="index")
        )
        lat_long_updates = missing_deps.map(
            lambda dep: geo_mapping.get(dep, {"lat": None, "long": None})
        )
        self.dfc.loc[self.dfc["lat"].isna(), ["lat", "long"]] = pd.DataFrame(
            lat_long_updates.tolist(), index=missing.index
        )

        logger.success("Geographical data added successfully")

    def create_dataset_common(self, year, election_type):
        """Create dataset for a specific election year and type"""
        # 1. Select socio-economic features
        features_year_list = set(
            [col[:-4] for col in self.dfc.columns if re.search(r"\d{4}$", col)]
        )

        # Some features are available in the given year
        year_pattern = r"(\d{4})$"
        columns_to_keep = [
            col
            for col in self.dfc.columns
            if not re.search(year_pattern, col)  # 'lat', 'long', 'codecommune'
            or int(re.search(year_pattern, col).group(1)) == year
        ]  # Feature of the year 2022
        dataset = self.dfc[columns_to_keep]

        # If the feature is not available for the given year,
        # we try to find a value in a previous year (up to 10 years before)
        self.feature_backfill_map = {}
        new_columns = {}
        dataset = dataset.copy()
        max_past_year = 10
        missing_year_features = features_year_list - set(
            [re.sub(rf"{year}$", "", col) for col in dataset.columns]
        )
        for feature in missing_year_features:
            for year_offset in range(1, max_past_year + 1):
                previous_year = year - year_offset
                if previous_year < year - max_past_year:
                    break
                if f"{feature}{previous_year}" in self.dfc.columns:
                    new_columns[f"{feature}{year}"] = self.dfc[
                        f"{feature}{previous_year}"
                    ]
                    # dataset.loc[:, f'{feature}{year}'] = self.dfc.loc[:,f'{feature}{previous_year}']
                    # We save in a dict that we cheated for this feature
                    self.feature_backfill_map[f"{feature}{year}"] = (
                        f"{feature}{previous_year}"
                    )
                    break

        if new_columns:
            new_columns_df = pd.DataFrame(new_columns, index=self.dfc.index)
            dataset = pd.concat([dataset, new_columns_df], axis=1)
            logger.warning(
                f"We imputed the following features with value within the last {max_past_year} years : {self.feature_backfill_map}"
            )

        # Remove duplicated communes
        duplicate_count = dataset.duplicated(subset=["codecommune"]).sum()
        if duplicate_count > 0:
            logger.warning(
                f"Found {duplicate_count} duplicate rows based on codecommune. Keeping first occurrence and dropping {duplicate_count} duplicates."
            )
            dataset.drop_duplicates(subset=["codecommune"], keep="first", inplace=True)

        # 2. Gather election data

        # Select election
        election = f"{year}-{election_type}"
        previous_election = self._find_previous_election(election)
        target_cols = ["codecommune", f"inscrits{election}"] + [
            f"pvote{var}{year}-{election_type}" for var in self.config.vote_variables
        ]

        for var in self.config.vote_variables:
            if previous_election is not None:
                target_cols.append(f"pvote{var}{previous_election}")
                previous_previous_election = self._find_previous_election(
                    previous_election
                )
                if previous_previous_election is not None:
                    target_cols.append(f"pvote{var}{previous_previous_election}")

        Y = self.dfs[target_cols].copy()

        # For referendum 1946 we have 'None' value instead of nan. In this specific case, we replace 'None' with np.nan
        if year == 1946 and election_type == "referendum":
            Y.dropna(subset=["codecommune"], inplace=True)
            logger.warning(
                "For referendum 1946 we have None value instead of nan. In this specific case, we replace None with np.nan"
            )

        # Some commune won't have election data for this election and we drop them
        logger.info(
            "Drop commune that have no election results for the given vote statistics"
        )
        for var in self.config.vote_variables:
            rows_before = Y.shape[0]
            Y.dropna(subset=[f"pvote{var}{election}"], inplace=True)
            rows_after = Y.shape[0]
            communes_dropped = rows_before - rows_after
            logger.warning(f"Number of communes dropped for {var}: {communes_dropped}")

        # 3. Merge
        dataset_common = pd.merge(
            Y, dataset, on="codecommune", how="left", validate="one_to_one"
        )

        # Rename election columns
        dataset_common.rename(
            columns={f"inscrits{election}": f"inscrits{year}"}, inplace=True
        )
        for var in self.config.vote_variables:
            dataset_common.rename(
                columns={f"pvote{var}{election}": f"pvote{var}{year}"}, inplace=True
            )
            if previous_election is not None:
                dataset_common.rename(
                    columns={
                        f"pvote{var}{previous_election}": f"pvoteprevious{var}{year}"
                    },
                    inplace=True,
                )
                previous_previous_election = self._find_previous_election(
                    previous_election
                )
                if previous_previous_election is not None:
                    dataset_common.rename(
                        columns={
                            f"pvote{var}{previous_previous_election}": f"pvotepreviousprevious{var}{year}"
                        },
                        inplace=True,
                    )

        return dataset_common

    def create_global_dataset(self):
        """Create the global dataset for modeling"""

        logger.info("Creating global dataset...")

        if self.config.first_election_only:
            # Single election dataset
            self.global_dataset = self.create_dataset_common(
                self.config.first_election_target_year,
                self.config.first_election_target_type,
            )

            # We make the columns year agnostic
            self.global_dataset.rename(
                columns=lambda col: re.sub(r"(\d{4})$", "", col), inplace=True
            )

            # We store year and election type as new features
            self.global_dataset["annee"] = self._map_election_year(
                float(self.config.first_election_target_year)
            )
            self.global_dataset["type"] = self._map_election_type(
                self.config.first_election_target_type, self.config.encoding_type
            )
            election_included = f"{self.config.first_election_target_type}{self.config.first_election_target_year}"
        else:
            # Multiple elections dataset
            first_dataset = True
            election_included = []
            for election_type in self.config.include_elections_of_type:
                year_pattern = r"pvote[a-zA-Z]*?(\d{4})-" + re.escape(election_type)
                relevant_years = list(
                    set(
                        int(re.search(year_pattern, col).group(1))
                        for col in self.dfs.columns
                        if re.search(year_pattern, col)
                    )
                )
                for year in relevant_years:
                    if year >= self.config.include_elections_after:
                        logger.info(f"Processing {election_type} {year}")
                        dataset_common = self.create_dataset_common(year, election_type)
                        dataset_common.rename(
                            columns=lambda col: re.sub(r"(\d{4})$", "", col),
                            inplace=True,
                        )
                        dataset_common["annee"] = self._map_election_year(year)
                        dataset_common["type"] = self._map_election_type(
                            election_type, self.config.encoding_type
                        )
                        election_included.append(f"{election_type}{year}")
                        if first_dataset:
                            self.global_dataset = dataset_common
                            first_dataset = False
                        else:
                            self.global_dataset = pd.concat(
                                [self.global_dataset, dataset_common],
                                axis=0,
                                ignore_index=True,
                            )

        logger.success(f"Global dataset created: {self.global_dataset.shape}")
        return election_included

    def save_processed_data(self):
        """Save all processed datasets and configuration file"""
        if not DataUtils._detect_s3(self.config.data_path):
            os.makedirs("derived/processed/", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save global dataset
        if self.global_dataset is not None:
            filename = f"data_{'_'.join(self.config.vote_variables)}_{self.config.include_elections_after}_{'_'.join(self.config.include_elections_of_type)}_{timestamp}.parquet"

            if "basesfiscalescommunes/basefonciere" in self.global_dataset.columns:
                self.global_dataset["basesfiscalescommunes/basefonciere"] = (
                    self.global_dataset["basesfiscalescommunes/basefonciere"].astype(
                        float
                    )
                )
                logger.warning(
                    "Weird behavior for feature basesfiscalescommunes/basefonciere. Performing float conversion"
                )

            DataLoader.write_dataset(
                self.global_dataset,
                self.config.data_path + "derived/processed/" + filename,
            )
            # if self.use_s3:
            #     self.global_dataset.to_parquet(
            #         self.config.data_path + "derived/processed/" + filename,
            #         index=False,
            #         filesystem=self.fs,
            #     )
            # else:
            #     self.global_dataset.to_parquet(
            #         os.path.join(self.config.data_path + "derived/processed/", filename),
            #         index=False,
            #     )

        logger.success(
            f"All processed data saved to {self.config.data_path+'derived/processed/'}"
        )

    def _find_previous_feature(self, feature, features_all):
        """Find the previous year's feature for a given feature"""
        # Extract the year and type from the election string
        year = re.search(r"(\d{4})$", feature)
        if year is not None:
            year = year.group(1)
        else:
            return None
        feat = re.sub(r"(\d{4})$", "", feature)
        year = int(year)

        # Extract years from the relevant columns
        years = [
            int(re.search(r"(\d{4})$", col).group(1))
            for col in features_all
            if re.sub(r"(\d{4})$", "", col) == feat and re.search(r"(\d{4})$", col)
        ]

        # Find the previous year
        previous_years = [y for y in years if y < year]
        if len(previous_years) > 0:
            previous_year = max(previous_years)  # Get the most recent previous year
            return f"{feat}{previous_year}"
        else:
            return None

    def _find_previous_election(self, election):
        """Find the previous election of the same type"""
        # Extract the year and type from the election string
        year, election_type = election.split("-")
        year = int(year)

        # Filter columns in dfs for the same election type
        relevant_columns = [col for col in self.dfs.columns if election_type in col]

        # Extract years from the relevant columns
        years = [
            int(re.search(r"(\d{4})", col).group(1))
            for col in relevant_columns
            if re.search(r"(\d{4})", col)
        ]

        # Find the previous year
        previous_years = [y for y in years if y < year]
        if previous_years:
            previous_year = max(previous_years)
            return f"{previous_year}-{election_type}"

        return None

    def _map_election_year(self, year):
        """
        For now, pass
        Encode with gini index (we need to find data for a long period of time)
        """
        return year

    def _map_election_type(self, election_type, encoding_logic="no"):
        """Map election type to numerical encoding"""
        read_config_arg = encoding_logic.split("_")
        choice = read_config_arg[0]
        if choice == "average_vote":
            var = read_config_arg[1]
            logger.info(f"Using average vote {var} mapping")
            mapping = {
                "presidentiel": (
                    self.dfs[
                        [
                            col
                            for col in self.dfs.columns
                            if "presidentiel" in col and f"pvote{var}" in col
                        ]
                    ]
                    .mean()
                    .mean()
                    if any(
                        "presidentiel" in col and f"pvote{var}" in col
                        for col in self.dfs.columns
                    )
                    else None
                ),
                "legislative": (
                    self.dfs[
                        [
                            col
                            for col in self.dfs.columns
                            if "legislative" in col and f"pvote{var}" in col
                        ]
                    ]
                    .mean()
                    .mean()
                    if any(
                        "legislative" in col and f"pvote{var}" in col
                        for col in self.dfs.columns
                    )
                    else None
                ),
                "referendum": (
                    self.dfs[
                        [
                            col
                            for col in self.dfs.columns
                            if "referendum" in col and f"pvote{var}" in col
                        ]
                    ]
                    .mean()
                    .mean()
                    if any(
                        "referendum" in col and f"pvote{var}" in col
                        for col in self.dfs.columns
                    )
                    else None
                ),
            }
        else:
            mapping = {"presidentiel": 0, "legislative": 1, "referendum": 2}
            if choice == "no":
                logger.info("Using no mapping")
            else:
                logger.info("mapping not recognized, using no mapping")

        return mapping.get(election_type, -1)

    def _choose_plm(self):
        """ "During the dataprocessing Paris, Mareseille and Lyon are treated as a commune and their arrondissement also.
        Choose whether to keep the commune at the aggregated of arrondissement level"""
        logger.info(f"PLM policy : {self.config.plm_policy}")
        PARIS = "75056"
        PARIS_arr = [str(x) for x in list(range(75101, 75121))]

        LYON = "69123"
        LYON_arr = [str(x) for x in list(range(69381, 69390))]

        MARSEILLE = "13055"
        MARSEILLE_arr = [str(x) for x in list(range(13201, 13217))]

        PLM = (
            PARIS,
            LYON,
            MARSEILLE if self.config.plm_policy == "Arr" else PARIS_arr,
            LYON_arr,
            MARSEILLE_arr,
        )
        n, p = self.global_dataset.shape
        self.global_dataset = self.global_dataset.loc[
            ~self.global_dataset["codecommune"].astype(str).isin(PLM), :
        ]
        n_, p_ = self.global_dataset.shape
        logger.info(f"Deleted {n-n_} row to comply with PLM policy")

    def _encode_dep_numeric(self):
        dep_col = self.global_dataset["dep"]
        dep_col = dep_col.replace("2A", "2").replace("2B", "2")
        self.global_dataset["dep_num"] = dep_col.astype(float)

    def post_processing(self):
        """Apply a few post-processsing"""

        # Choose PLM
        self._choose_plm()

        # Encode dep to numeric columns
        self._encode_dep_numeric()


def main():
    """Main function to run the data processing pipeline"""
    # Initialize processor
    processor = ElectionDataProcessor()

    # Load election data
    processor.load_electoral_data()

    # Load socioeconomic data
    processor.load_socioeconomic_data()

    # Create projections for socio-economic data
    processor.project_socioeconomic_data()

    # Feature engineering
    processor.apply_feature_engineering()

    # Geo data
    processor.add_geographical_data()

    # Create global dataset
    election_included = processor.create_global_dataset()

    # Apply a few post-processing
    processor.post_processing()

    # Quality filter
    processor.apply_quality_filter()

    # Save processed data
    processor.save_processed_data()

    logger.success("Data processing pipeline completed!")
    logger.debug(f"Election include: {election_included}")

    return processor.global_dataset


if __name__ == "__main__":
    dataset = main()
