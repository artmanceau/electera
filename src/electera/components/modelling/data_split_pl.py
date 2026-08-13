import polars as pl
import polars.selectors as cs
from loguru import logger
from sklearn.model_selection import train_test_split


def split_method(
    data,
    way="random",
    election_type=None,
    test_year=None,
    train_year=None,
    validation_year=None,
):
    if way == "last-try-seq":
        # 1. Get all available years before the test year, sorted most recent first
        available_years = (
            data.filter(
                (pl.col("election_type") == election_type)
                & (pl.col("annee") < int(test_year))
            )
            .filter(pl.col("annee") >= 1960)
            .select("annee")
            .unique()
            .sort("annee", descending=True)
            .to_series()
            .to_list()
        )
        available_years = available_years[:3]
        sampled_dfs = []

        for i, year in enumerate(available_years, start=1):
            # Filter data for this specific election year
            year_data = data.filter(
                (pl.col("election_type") == election_type) & (pl.col("annee") == year)
            )

            # Sample with fraction 1/i (1/1, 1/2, 1/3...)
            sampled_year = year_data.sample(fraction=1 / i, seed=42)
            sampled_dfs.append(sampled_year)

        data_train_val = pl.concat(sampled_dfs)
        data_test = data.filter(
            (pl.col("election_type") == election_type)
            & (pl.col("annee") == int(test_year))
        )

        # 5. Split into train and validation
        data_train, data_validation = train_test_split(
            data_train_val, test_size=0.25, random_state=42, shuffle=True
        )

    elif way == "last-try":
        data_train_val = (
            data.filter(pl.col("election_type") == election_type)
            .filter(pl.col("annee") < int(test_year))
            .filter(pl.col("annee") >= 1960)
            .filter(pl.col("annee") >= int(test_year) - 15)
        )
        data_test = data.filter(pl.col("election_type") == election_type).filter(
            pl.col("annee") == int(test_year)
        )
        data_train, data_validation = train_test_split(
            data_train_val, test_size=0.25, random_state=42, shuffle=True
        )
    elif way == "time-serie-cv":
        data_train = (
            data.filter(pl.col("election_type") == election_type)
            .filter(pl.col("annee") <= int(validation_year))
            .filter(pl.col("annee") >= 1960)
            .filter(pl.col("annee") >= int(validation_year) - 20)
        )
        data_test = data.filter(pl.col("election_type") == election_type).filter(
            pl.col("annee") == int(test_year)
        )
        data_validation = data.filter(pl.col("election_type") == election_type).filter(
            pl.col("annee") == int(train_year)
        )
    elif way == "last-only":
        data_train = data.filter(pl.col("election_type") == election_type).filter(
            pl.col("annee") == int(train_year)
        )
        data_test = data.filter(pl.col("election_type") == election_type).filter(
            pl.col("annee") == int(test_year)
        )
        data_validation = data.filter(pl.col("election_type") == election_type).filter(
            pl.col("annee") < int(validation_year)
        )
    elif way == "random":
        # Time limit on elections
        data = data.filter(pl.col("annee") >= 1960).filter(pl.col("annee") <= 2022)
        data_train_val, data_test = train_test_split(
            data, test_size=0.2, random_state=42, shuffle=True
        )
        data_train, data_validation = train_test_split(
            data_train_val, test_size=0.25, random_state=42, shuffle=True
        )
    else:
        logger.error("Split method not implemented, no splits implemented")
        data_train, data_test, data_validation = data, data, data

    return data_train, data_test, data_validation


def get_Xy_pl(
    data,
    vote_variable="pvotepar",
    year=None,
    election_type=None,
    predict_delta=False,
    predict_perc=False,
    selected_groups=[
        "raw",
        "rank",
        "delta",
        "lag",
        "geo",
        "previous_vote",
        "type",
        "year",
        "inscrits",
        "pct_change",
    ],
    selected_features=None,
    split_method_way="random",
):
    # Tau is monotonic
    vote_variable_perc = (
        vote_variable
        if "tau" not in vote_variable
        else vote_variable.replace("tau", "")
    )

    data = data.with_columns(
        [
            pl.col(col).fill_null(pl.col(col).mean().over("dep", "annee"))
            for col in [
                f"previous{vote_variable}",
                f"previousprevious{vote_variable}",
                f"previouspercentile{vote_variable_perc}",
            ]
        ],
    ).with_columns(
        (pl.col(vote_variable) - pl.col(f"previous{vote_variable}")).alias(
            f"delta{vote_variable}"
        ),
        (
            pl.col(f"percentile{vote_variable_perc}")
            - pl.col(f"previouspercentile{vote_variable_perc}")
        ).alias(f"deltapercentile{vote_variable}"),
        (pl.lit(0.0)).alias(f"previousdeltapercentile{vote_variable}"),
        (
            pl.col(f"previous{vote_variable}")
            - pl.col(f"previousprevious{vote_variable}")
        ).alias(f"previousdelta{vote_variable}"),
    )

    if predict_perc:
        vote_variable = f"percentile{vote_variable}"

    if predict_delta:
        y = f"delta{vote_variable}"
        y_prev = f"previousdelta{vote_variable}"
        previous_delta = f"previousdelta{vote_variable}"

    else:
        y = vote_variable
        y_prev = f"previous{vote_variable}"
        # y_prev_prev = f"previousprevious{vote_variable}"
        previous_delta = f"previousdelta{vote_variable}"

    data = data.select(
        # Features
        list(cs.expand_selector(data, cs.starts_with("F_")))
        # Other columns, target and previous vote cols
        + [
            "inscrits",
            "type",
            "election_type",
            "annee",
            "lat",
            "long",
            "distanceparis",
            "distancelyon",
            "distancemarseille",
            "dep_num",
            y,
            y_prev,
            # y_prev_prev,
        ]
        + ["codecommune", "dep"]
    )
    data = data.drop_nulls(subset=y)

    # Assert no NaN
    assert data.select(pl.sum_horizontal(cs.float().is_nan())).sum().item() == 0

    # Assert no inf
    assert data.select(pl.sum_horizontal(cs.float().is_infinite())).sum().item() == 0

    available_years = sorted(
        data.filter(pl.col("election_type") == election_type)
        .unique("annee")
        .get_column("annee")
        .to_list()
    )
    test_year = year
    x = available_years.index(test_year)
    train_year, validation_year = available_years[x - 1], available_years[x - 2]

    # if x < 2:
    #     logger.warning(
    #         "Not possible because we don't have enough past elections. Choosing random elections years instead"
    #     )
    data_train, data_test, data_validation = split_method(
        data,
        way=split_method_way,
        election_type=election_type,
        test_year=test_year,
        train_year=train_year,
        validation_year=validation_year,
    )

    assert len(data_train) > 0

    logger.debug(
        f"Test election: {data_test.unique('annee').get_column('annee').to_list()}, train election: {data_train.unique('annee').get_column('annee').to_list()}, validation election: {data_validation.unique('annee').get_column('annee').to_list()}"
    )

    feature_groups = {
        "rank": list(cs.expand_selector(data_train, cs.starts_with("F_rank"))),
        "raw": list(cs.expand_selector(data_train, cs.starts_with("F_raw"))),
        "pct_change": list(
            cs.expand_selector(data_train, cs.starts_with("F_pct_change"))
        ),
        "delta": list(cs.expand_selector(data_train, cs.starts_with("F_delta"))),
        "lag": list(cs.expand_selector(data_train, cs.starts_with("F_lag"))),
        "geo": [
            "lat",
            "long",
            "distanceparis",
            "distancelyon",
            "distancemarseille",
            "dep_num",
        ],
        "previous_vote": set(
            [y_prev, f"previous{y_prev}"] + [previous_delta]
        ).intersection(set(data_train.columns)),
        "inscrits": ["inscrits"],
        "year": ["annee"],
        "type": ["type"],
    }

    if selected_features is not None:
        features = selected_features

    else:
        features = [
            col for group in selected_groups for col in feature_groups.get(group, [])
        ]

    y_train, y_val, y_test = (
        data_train.get_column(y),
        data_validation.get_column(y),
        data_test.get_column(y),
    )
    y_previous = data_test.get_column(y_prev)

    remove_nulls = True
    if remove_nulls:
        null_cols_train = set([s.name for s in data_train if s.has_nulls()])
        null_cols_test = set([s.name for s in data_test if s.has_nulls()])
        null_cols_validation = set([s.name for s in data_validation if s.has_nulls()])
        null_cols = (null_cols_train.union(null_cols_test)).union(null_cols_validation)

        features = list(set(features) - null_cols)

    # + [f'previous{vote_variable}', f'previousprevious{vote_variable}']
    X_train, X_test, X_val = (
        data_train.select(features),
        data_test.select(features),
        data_validation.select(features),
    )
    meta_cols = ["codecommune", "dep", "inscrits"]

    meta_train, meta_test, meta_val = (
        data_train.select(meta_cols),
        data_test.select(meta_cols),
        data_validation.select(meta_cols),
    )

    # Assert no null
    # for df in [X_train, X_test, X_val]:
    #    assert df.select(pl.sum_horizontal(pl.all().is_null())).sum().item() == 0

    return (
        X_train.to_pandas(),
        X_val.to_pandas(),
        X_test.to_pandas(),
        y_train.to_pandas(),
        y_val.to_pandas(),
        y_test.to_pandas(),
        y_previous.to_pandas(),
        meta_train.to_pandas(),
        meta_val.to_pandas(),
        meta_test.to_pandas(),
    )
