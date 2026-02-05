from typing import List, Literal

from pydantic import BaseModel, Field, HttpUrl, validator


class BackTesterConfig(BaseModel):
    # Political trends/variables
    political_trends: List[List[str]] = Field(
        default=["voteCG", "voteCD", "voteC", "voteD", "voteG", "par"],
        description="Political trend variables to include in modeling",
    )

    # Dataset configuration
    data_path: str

    dataset_path: str = Field(
        description="Path to the processed dataset (S3 or local path)"
    )

    random_seed: int = Field(
        default=42, ge=0, description="Random seed for reproducibility"
    )

    # Test/validation configuration
    k_year: List[int]

    k_type: List[Literal["pres", "leg", "ref"]] = Field(
        description="Type of test election (abbreviated)"
    )

    model: Literal[
        "trivial_1",
        "trivial_2",
        "boosting",
        "linear",
        "meta_boosting",
        "meta_boosting_multiple",
    ] = Field(default="trivial_2", description="Model to use")

    @validator("political_trends")
    def validate_political_trends(cls, v):
        """Validate that political_trends contains only allowed combinations."""

        # Define allowed variable sets
        allowed_sets = [
            {"par", "voteTD", "voteTG"},
            {"par", "voteGCG", "voteC", "voteDCD"},
            {"voteCG", "voteCD", "voteC", "voteD", "voteG", "par"},  # existing default
        ]

        # Ensure the input is a list of lists
        if not isinstance(v, list) or not all(isinstance(inner, list) for inner in v):
            raise ValueError("political_trends must be a list of lists.")

        # Validate each inner list
        for inner_list in v:
            input_set = set(inner_list)
            if not any(
                set(input_set) == set(allowed_set) for allowed_set in allowed_sets
            ):
                allowed_combinations = [list(s) for s in allowed_sets]
                raise ValueError(
                    f"Each inner list in political_trends must match one of the allowed combinations: {allowed_combinations}."
                    f"Got: {inner_list}"
                )

        return v


class TrainModelsConfig(BaseModel):
    """Configuration class for the model training pipeline."""

    # Target
    vote_variable: Literal[
        "ppar",
        "ppar",
        "pvoteG",
        "pvoteC",
        "pvoteD",
        "pvoteCG",
        "pvoteCD",
        "pvoteTG",
        "pvoteTD",
        "pvoteGCG",
        "pvoteDCD",
    ] = Field(default="ppar", description="Model for this variable (target)")

    # Data
    dataset_path: str = Field(description="Path to the dataset file (local or S3)")
    remove_previous_features: bool = Field(
        default=False, description="Whether to remove previous election target"
    )

    # Split
    split_method: str = Field(
        default="shuffle", description="Method for splitting the data"
    )

    predict_delta: bool = Field(
        default=False,
        description="Predict the delta in vote statistics with previous or vote statistics",
    )

    val_size: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Proportion of data to use for validation",
    )
    test_size: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Proportion of data to use for testing"
    )

    # Random seed
    random_state: int = Field(default=42, description="Random seed for reproducibility")

    # Parameters tuning
    param_search_methods: List[str] = Field(
        default=["none"], description="Methods for hyperparameter tuning"
    )
    feature_selection_methods: List[str] = Field(
        default=["none"], description="Methods for feature selection"
    )
    boosting_methods: List[Literal["xgboost", "gpboost", "catboost"]] = Field(
        default=["xgboost"], description="Boosting method to use"
    )
    top_n_features: int = Field(
        default=100, gt=0, description="Number of top features to use for XGBoost"
    )

    # MLFlow
    use_MLFlow: bool = Field(
        default=False, description="Whether to use MLFlow for experiment tracking"
    )

    # Models
    models: List[
        Literal[
            "trivial_1",
            "trivial_2",
            "linear_reg",
            "elastic_net",
            "boosting",
            "meta_boosting",
            "meta_boosting_multiple",
        ]
    ] = Field(
        default=["trivial_1", "trivial_2", "linear_reg"], description="Models to train"
    )


class DataProcessingConfig(BaseModel):
    """Configuration class for the data processing pipeline."""

    # Vote variables
    vote_variables: List[
        Literal[
            "ppar",
            "ppar",
            "pvoteG",
            "pvoteC",
            "pvoteD",
            "pvoteCG",
            "pvoteCD",
            "pvoteTG",
            "pvoteTD",
            "pvoteGCG",
            "pvoteDCD",
        ]
    ] = Field(
        default=["ppar", "pvoteG", "pvoteC", "pvoteD", "pvoteCG", "pvoteCD"],
        description="Vote variables to include in processing",
    )

    # Data location
    data_path: str = Field(
        default="s3://arthurmanceau/election_modeling_uhcp/data/",
        description="S3 path for data storage (or local path like 'data/')",
    )

    # Single election configuration
    first_election_only: bool = Field(
        default=False, description="Whether to create dataset for single election only"
    )

    first_election_target_year: int = Field(
        default=2022,
        ge=1700,
        le=2100,
        description="Target year if processing single election",
    )

    first_election_target_type: str = Field(
        default="presidentiel",
        description="Target election type if processing single election",
    )

    # Election filtering
    include_elections_after: int = Field(
        default=1970,
        ge=1789,
        le=2022,
        description="Minimum year for elections to include",
    )

    include_elections_of_type: List[
        Literal["presidentiel", "legislative", "referundum"]
    ] = Field(
        default=["presidentiel", "legislative"],
        description="Types of elections to include in the dataset",
    )

    # Feature engineering
    features_aug: List[Literal["rank", "delta", "lag", "pct_change", "winsor"]] = Field(
        default=[], description="Feature augmentation methods to apply"
    )

    # Encoding configuration
    encoding_type: Literal["no", "vote_variable"] = Field(
        default="no",
        description="Type of encoding to apply ('no' or 'vote_variable' for average vote)",
    )
    encoding_year: Literal["no"] = Field(
        default="no", description="Year-based encoding method ('no' or 'gini')"
    )

    # Quality filtering
    quality_filter: bool = Field(
        default=False, description="Whether to apply quality filtering"
    )

    # Projections
    projections: bool = Field(
        default=False, description="Whether to project socio-economic data"
    )

    # PLM policy
    plm_policy: Literal["Agg", "Arr"] = Field(
        default="Arr", description="Whether to keep the arrondissement of PLM commune"
    )

    polls_data: bool = Field(default=False, description="Whether to add polling data")

    @property
    def elections_to_exclude(self) -> List[str]:
        """
        Exclude referendum files if vote_variables contains anything other than 'ppar'.
        """
        exclusions = []
        if any(var != "ppar" for var in self.vote_variables):
            exclusions.extend(
                [
                    "ref1795comm.parquet",
                    "ref1793comm.parquet",
                    "ref1992comm.parquet",
                    "ref1946comm.parquet",
                    "ref2005comm.parquet",
                ]
            )
        return exclusions


class DownloadDataConfig(BaseModel):
    # Source
    url: HttpUrl = Field(description="URL to download data from")

    geo_url: HttpUrl = Field(description="URL to download geodata from")

    # Data path (to download the data)
    data_path: str = Field(
        default="data/raw", description="Local path for data storage"
    )


class ExplanabilityConfig(BaseModel):
    model_version: str
    years: List[int]
    types: List[str]
    vars_: List[str]
    data_path: str
    steps: List[str]


class CounterfactualConfig(BaseModel):
    model_version: str
    year: int
    type_: str
    var: str
    data_path: str


class AppConfig(BaseModel):
    model_version: str
    years_to_display: List[int]
    types_to_display: List[str]
    data_path: str
