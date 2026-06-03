import polars.selectors as cs

def get_Xy(self, data, predict_delta=False, selected_features=None):
        feature_cols = list(cs.expand_selector(data, cs.starts_with("F_")))
        data = data.with_columns(
            [
                pl.col(col).fill_nan(
                    pl.col(col).mean().over("dep")
                )
                for col in previous_vote_cols
            ],
            y_delta = pl.col(self.vote_variable) - pl.col(f"previouspvote{self.var}")
            )
        data.select(feature_cols + ["inscrits", "type", "annee", "lat", "long", "dep_num", self.vote_variable, y_delta, f"previous{self.vote_variable}", f"previousprevious{self.vote_variable}"]).cast(Float64)

        # Assert no null
        assert data.select(pl.sum_horizontal(pl.all().is_null())).sum().item() == 0

        # Assert no NaN
        assert data.select(pl.sum_horizontal(cs.float().is_nan())).sum().item() == 0

        # Assert no inf
        assert data.select(pl.sum_horizontal(cs.float().is_infinite())).sum().item() == 0
        
        # Replace inf with nan
        # (handled by models - all proposed models should handle missing values)
        inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        assert inf_count == 0

        # Necessary ———

        # Make sure there is no nan
        indices_to_drop = y[y.isna()].index
        if len(indices_to_drop) > 0:
            X.drop(index=indices_to_drop, errors="ignore", inplace=True)
            y.drop(index=indices_to_drop, errors="ignore", inplace=True)
            y_split.drop(index=indices_to_drop, errors="ignore", inplace=True)
            logger.warning(
                f"{len(indices_to_drop)} rows were dropped because the target was missing!"
            )

        available_years = data.filter(pl.col('type')==float(t)).unique('annee').get_column('annee').to_list()
        test_year = available_years.index(float(y))
        train_year = available_years[x - 1]
        validation_year = available_years[x - 2]

        if x < 2:
            logger.warning(
                "Not possible because we don't have enough past elections. Choosing random elections years instead"
            )

        data_train = data.filter(pl.col('type')==float(t)).filter(pl.col('annee')==float(train_year))
        data_test = data.filter(pl.col('type')==float(t)).filter(pl.col('annee')==float(test_year))
        data_validation = data.filter(pl.col('type')==float(t)).filter(pl.col('annee')==float(validation_year))

        logger.debug(
            f"Test election: {y}, train election: {t_year}, validation election: {v_year}"
        )

        # Drop missing values (means the feature is not available)
        cols_to_drop = list(set(s.name for df in [data_train, data_test, data_validation] for s in df if s.null_count() > 0))
        data_train, data_test, data_validation = [df.drop(cols_to_drop) for df in [data_train, data_test, data_validation]]

        if selected_features is not None:
            features = selected_features
        else:
            feature_groups = {
                "rank": list(cs.expand_selector(data, cs.starts_with("F_rank"))),
                "raw": list(cs.expand_selector(data, cs.starts_with("F_raw"))),
                "pct_change": list(cs.expand_selector(data, cs.starts_with("F_rank"))),
                "delta": list(cs.expand_selector(data, cs.starts_with("F_delta"))),
                "lag": list(cs.expand_selector(data, cs.starts_with("F_lag"))),
                "geo": ['lat', 'long'],
                "previous_vote": [f"previous{self.vote_variable}", f"previousprevious{self.vote_variable}"],
                "other": ["inscrits", "dep_num"],
            }
            selected_groups = ["raw", "rank", "delta", "geo", "previous_vote", "other"]
            features = [
                col
                for group in selected_groups
                for col in feature_groups.get(group, [])
            ]

        X_train, X_val, X_test = data_train.select(features).to_pandas(), data_test.select(features).to_pandas(), data_validation.select(features).to_pandas()        
        y = 'y_delta' if predict_delta else self.vote_variable
        y_train, y_val, y_test = data_train.get_column(y).to_pandas(), data_test.get_column(y).to_pandas(), data_validation.get_column(y).to_pandas()

        return X_train, X_val, X_test, y_train, y_val, y_test
