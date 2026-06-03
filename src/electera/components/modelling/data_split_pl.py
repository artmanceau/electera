import polars as pl
import polars.selectors as cs
from loguru import logger


def get_Xy_pl(
    data,
    vote_variable,
    year,
    election_type,
    predict_delta=False,
    predict_perc=False,
    selected_groups=["raw", "rank", "delta", "geo", "previous_vote", "other"],
    selected_features=None,
):
    data = data.with_columns(
        [
            pl.col(col).fill_nan(pl.col(col).mean().over("dep"))
            for col in [f"previous{vote_variable}", f"previousprevious{vote_variable}"]
        ],
        y_delta=pl.col(vote_variable) - pl.col(f"previous{vote_variable}"),
    )

    if predict_delta:
        y = "y_delta"
    elif predict_perc:
        y = f"percentile{vote_variable}"
    else:
        y = vote_variable

    data.select(
        # Features
        list(cs.expand_selector(data, cs.starts_with("F_")))
        # Other columns, target and previous vote cols
        + [
            "inscrits",
            "election_type",
            "annee",
            "lat",
            "long",
            "distanceparis",
            "dep_num",
            y,
            f"previous{vote_variable}",
            f"previousprevious{vote_variable}",
        ]
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

    if x < 2:
        logger.warning(
            "Not possible because we don't have enough past elections. Choosing random elections years instead"
        )

    data_train = data.filter(pl.col("election_type") == election_type).filter(
        pl.col("annee") == int(train_year)
    )
    data_test = data.filter(pl.col("election_type") == election_type).filter(
        pl.col("annee") == int(test_year)
    )
    data_validation = data.filter(pl.col("election_type") == election_type).filter(
        pl.col("annee") == int(validation_year)
    )

    logger.debug(
        f"Test election: {test_year}, train election: {train_year}, validation election: {validation_year}"
    )

    # Drop missing values (means the feature is not available)
    cols_to_drop = list(
        set(
            s.name
            for df in [data_train, data_test, data_validation]
            for s in df
            if s.null_count() > 0
        )
    )
    data_train, data_test, data_validation = [
        df.drop(cols_to_drop) for df in [data_train, data_test, data_validation]
    ]

    # Assert no null
    for df in [data_train, data_test, data_validation]:
        assert df.select(pl.sum_horizontal(pl.all().is_null())).sum().item() == 0

    if selected_features is not None:
        features = selected_features
    else:
        feature_groups = {
            "rank": list(cs.expand_selector(data_train, cs.starts_with("F_rank"))),
            "raw": list(cs.expand_selector(data_train, cs.starts_with("F_raw"))),
            "pct_change": list(
                cs.expand_selector(data_train, cs.starts_with("F_rank"))
            ),
            "delta": list(cs.expand_selector(data_train, cs.starts_with("F_delta"))),
            "lag": list(cs.expand_selector(data_train, cs.starts_with("F_lag"))),
            "geo": ["lat", "long"],
            "previous_vote": set(
                [f"previous{vote_variable}", f"previousprevious{vote_variable}"]
            ).intersection(set(data_train.columns)),
            "other": ["inscrits", "dep_num"],
        }
        features = [
            col for group in selected_groups for col in feature_groups.get(group, [])
        ]

    X_train, X_val, X_test = (
        data_train.select(features).to_pandas(),
        data_test.select(features).to_pandas(),
        data_validation.select(features).to_pandas(),
    )
    y_train, y_val, y_test = (
        data_train.get_column(y).to_pandas(),
        data_test.get_column(y).to_pandas(),
        data_validation.get_column(y).to_pandas(),
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
