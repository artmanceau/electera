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
            pl.col(col).fill_null(pl.col(col).mean().over("dep"))
            for col in [
                f"previous{vote_variable}",
                f"previousprevious{vote_variable}",
                f"previouspercentile{vote_variable}"
            ]
        ],
    ).with_columns(
        (
            pl.col(vote_variable) - pl.col(f"previous{vote_variable}")
        ).alias(
            f"delta{vote_variable}"
        ),
        (
            pl.col(f"percentile{vote_variable}")-pl.col(f"previouspercentile{vote_variable}")
        ).alias(
            f"deltapercentile{vote_variable}"
        ),
        (
            pl.lit(0.0)
        ).alias(
            f"previousdeltapercentile{vote_variable}"
        ),
        (
            pl.col(f"previous{vote_variable}")
            - pl.col(f"previousprevious{vote_variable}")
        ).alias(
            f"previousdelta{vote_variable}"
        ),
    )

    if predict_perc:
        vote_variable = f"percentile{vote_variable}"

    if predict_delta:
        y = f"delta{vote_variable}"
        y_prev = f"previousdelta{vote_variable}"
    else:
        y = vote_variable
        y_prev = f"previous{vote_variable}"

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
            y_prev,
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

    if x < 2:
        logger.warning(
            "Not possible because we don't have enough past elections. Choosing random elections years instead"
        )

    # Change here the splitting logic
    data_train = data.filter(pl.col("election_type") == election_type).filter(
        pl.col("annee") <= int(train_year)
    )
    data_test = data.filter(pl.col("election_type") == election_type).filter(
        pl.col("annee") == int(test_year)
    )
    data_validation = data.filter(pl.col("election_type") == election_type).filter(
        pl.col("annee") <= int(validation_year)
    )

    logger.debug(
        f"Test election: {data_test.unique('annee').get_column('annee').to_list()}, train election: {data_train.unique('annee').get_column('annee').to_list()}, validation election: {data_validation.unique('annee').get_column('annee').to_list()}"
    )

    if selected_features is not None:
        features = selected_features
    else:
        feature_groups = {
            "rank": list(cs.expand_selector(data_train, cs.starts_with("F_rank"))),
            "raw": list(cs.expand_selector(data_train, cs.starts_with("F_raw"))),
            "pct_change": list(
                cs.expand_selector(data_train, cs.starts_with("F_pct_change"))
            ),
            "delta": list(cs.expand_selector(data_train, cs.starts_with("F_delta"))),
            "lag": list(cs.expand_selector(data_train, cs.starts_with("F_lag"))),
            "geo": ["lat", "long"],  # 'distanceparis'
            "previous_vote": set([y_prev, f"previous{y_prev}"]).intersection(
                set(data_train.columns)
            ),
            "other": ["inscrits", "dep_num"],
        }
        features = [
            col for group in selected_groups for col in feature_groups.get(group, [])
        ]

    y_train, y_val, y_test = (
        data_train.get_column(y),
        data_validation.get_column(y),
        data_test.get_column(y),
    )
    y_previous = (
        data.filter(pl.col("election_type") == election_type)
        .filter(pl.col("annee") == int(test_year))
        .get_column(y_prev)
    )

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
    #     assert df.select(pl.sum_horizontal(pl.all().is_null())).sum().item() == 0

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
