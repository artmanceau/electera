# 3. For each political trend train a XGBoost model
# for trend in self.config.political_trends:
#     logger.info(f"Training model for trend: {trend}")

#     # XGboost
#     suffixes = []
#     for (
#         param_search_method
#     ) in (
#         self.config.param_search_methods
#     ):  # List of hyperparameter tuning methods
#         for (
#             feature_selection_method
#         ) in (
#             self.config.feature_selection_methods
#         ):  # List of feature selection methods

#             logger.info(
#                 f"Running pipeline with feature selection: {feature_selection_method}, parameter search: {param_search_method}"
#             )

#             # 0. Boosting algorithm
#             boosting_model = BoostingModel()
#             boosting_model.set_boosting_method("xgboost")

#             # 1. Feature selection
#             feat_sel = boosting_model.feature_selection(
#                 feature_selection_method,
#                 self.config.top_n_features,
#                 X_val=self.X_val[trend],
#                 y_val=self.y_val[trend],
#             )

#             # 2. Grid search to tune hyperparameters
#             boosting_model.parameter_search(
#                 param_search_method,
#                 X_val=self.X_val[trend],
#                 y_val=self.y_val[trend],
#             )

#             # 3. Train
#             model, signature = boosting_model.train(
#                 X_train=self.X_train[trend],
#                 y_train=self.y_train[trend],
#                 X_val=self.X_val[trend],
#                 y_val=self.y_val[trend],
#             )
#             model_name = boosting_model.get_model_name() + f"_{trend}"
#             suffixes.append(model_name)
#             self.models[model_name] = model
#             self.features_after_selection[model_name] = feat_sel
#             logger.info(f"Boosting model trained {model_name}...")

#             # 4. Evaluate
#             self.results[model_name] = ModelEvaluator.evaluate(
#                 self.y_test[trend],
#                 boosting_model.infer(self.X_test[trend]),
#                 model_name,
#             )

# Logic to get the final suffix as the best model among all trained for each political trend (for now only one suffix allowed for all)
# best_model_per_trend = self.find_best_model(suffixes)
