import polars as pl
import polars.selectors as cs
from loguru import logger
from sklearn.model_selection import train_test_split


def split_method(data, way='random', election_type=None, test_year=None, train_year=None, validation_year=None):
    if way == 'time-serie-cv':
        data_train = data.filter(pl.col("election_type") == election_type).filter(
            pl.col("annee") < int(test_year)
        )
        data_test = data.filter(pl.col("election_type") == election_type).filter(
            pl.col("annee") == int(test_year)
        )
        data_validation = data.filter(pl.col("election_type") == election_type).filter(
            pl.col("annee") < int(test_year)
        )
    elif way == 'last-only':
        data_train = data.filter(pl.col("election_type") == election_type).filter(
            pl.col("annee") == int(train_year)
        )
        data_test = data.filter(pl.col("election_type") == election_type).filter(
            pl.col("annee") == int(test_year)
        )
        data_validation = data.filter(pl.col("election_type") == election_type).filter(
            pl.col("annee") < int(validation_year)
        )
    elif way == 'random':
        # Only "recent" elections
        data = data.filter(pl.col('annee')>1950)
        data_train_val, data_test = train_test_split(
            data, test_size=0.33, random_state=42, shuffle=True)
        data_train, data_validation = train_test_split(
            data_train_val, test_size=0.33, random_state=42, shuffle=True)
    else:
        logger.error('Split method not implemented, no splits implemented')
        data_train, data_test, data_validation = data, data, data
    return data_train, data_test, data_validation


def get_Xy_pl(
    data,
    vote_variable,
    year,
    election_type,
    predict_delta=False,
    predict_perc=False,
    selected_groups=["raw", "rank", "delta", "geo", "previous_vote", "other", 'meta'],
    selected_features=None,
):
    # Tau is monotonic
    vote_variable_perc = vote_variable if 'tau' not in vote_variable else vote_variable.replace('tau', '')

    data = data.with_columns(
        [
            pl.col(col).fill_null(pl.col(col).mean().over("dep", 'annee'))
            for col in [
                f"previous{vote_variable}",
                f"previousprevious{vote_variable}",
                f"previouspercentile{vote_variable_perc}"
            ]
        ],
    ).with_columns(
        (
            pl.col(vote_variable) - pl.col(f"previous{vote_variable}")
        ).alias(
            f"delta{vote_variable}"
        ),
        (
            pl.col(f"percentile{vote_variable_perc}")-pl.col(f"previouspercentile{vote_variable_perc}")
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
            "type",
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

    # if x < 2:
    #     logger.warning(
    #         "Not possible because we don't have enough past elections. Choosing random elections years instead"
    #     )

    data_train, data_test, data_validation = split_method(data, way='random', election_type=election_type, test_year=test_year, train_year=train_year, validation_year=validation_year)

    assert len(data_train) > 0

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
            "geo": ["lat", "long", "distanceparis"],
            "previous_vote": set([y_prev, f"previous{y_prev}"]).intersection(
                set(data_train.columns)
            ),
            "other": ["inscrits", "dep_num"],
            "meta": ['annee', 'type']
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
        data_test.get_column(y_prev)
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
