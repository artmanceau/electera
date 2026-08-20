import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from loguru import logger
from PyALE import ale
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from supertree import SuperTree

from electera.components.data_processing.data_loader import DataLoader, DataUtils
from electera.components.explanability.core_explanability import ExplainCore
from electera.components.explanability.feature_importance import FeatureImportance
from electera.components.utils.config import ExplanabilityConfig
from electera.components.utils.read_config import ConfigReader


class Explainer:
    def __init__(self, **kwargs):
        """Initialize the explainability pipeline with a configuration."""

        self.config = ConfigReader._read_config(
            "../config/explainability.json", ExplanabilityConfig
        )
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        self.data_path = self.config.data_path
        self.model_version = self.config.model_version
        self.steps = self.config.steps
        self.shap_values_computed = False

        self.output_dir = f"{self.data_path}output/explain/"
        if not DataUtils._detect_s3(self.data_path):
            os.makedirs(self.output_dir, exist_ok=True)
            self.local_output_dir = self.output_dir
        else:
            # Assume data path
            self.local_output_dir = "output/explain/"
            os.makedirs(self.local_output_dir, exist_ok=True)

        self.n_models = None
        self.models = None

    def generate_feature_importance(self, X, y, shap_values=None):
        """Generate and save feature importance plots using multiple methods."""
        logger.info("Generating feature importance plots...")

        features = (
            self.model.models[self.var_c].features
            if self.model.models[self.var_c].features is not None
            else self.model.features[self.var_c]
        )
        models = self.model.models[self.var_c].best_models

        # ------------------------------------------------------------------
        # 1. XGBoost booster importances
        # ------------------------------------------------------------------
        logger.info("Calculating booster feature importances...")

        importance_types = [
            "weight",
            "gain",
            "cover",
            "total_gain",
            "total_cover",
        ]

        booster_dfs = []

        for importance_type in importance_types:
            importance = pd.DataFrame(index=features)

            for model in models:
                score = model.get_booster().get_score(importance_type=importance_type)
                importance[model] = pd.Series(score)

            importance_mean = importance.mean(axis=1).fillna(0.0)

            booster_df = (
                importance_mean.rename_axis("Feature")
                .reset_index(name="Importance")
                .sort_values("Importance", ascending=False)
            )

            booster_dfs.append(
                booster_df.rename(columns={"Importance": importance_type})
            )

        booster_importance_df = booster_dfs[0]
        for df in booster_dfs[1:]:
            booster_importance_df = booster_importance_df.merge(
                df,
                on="Feature",
                how="outer",
            )

        # ------------------------------------------------------------------
        # 2. Permutation importance
        # ------------------------------------------------------------------
        PERMUTATION = False

        if PERMUTATION:
            logger.info("Calculating permutation feature importance...")

            perm_importance_df = FeatureImportance.compute_importance(
                models=models,
                features=features,
                _get_importance_method=lambda model: permutation_importance(
                    model,
                    X,
                    y,
                    n_repeats=2,
                    random_state=42,
                ),
            )
        else:
            logger.warning("Skipping permutation feature importance")

            perm_importance_df = pd.DataFrame(
                {
                    "Feature": features,
                    "Importance": np.zeros(len(features)),
                }
            ).sort_values("Importance", ascending=False)

        # ------------------------------------------------------------------
        # 3. SHAP importance
        # ------------------------------------------------------------------
        if self.shap_values_computed and shap_values is not None:
            logger.info("Calculating SHAP feature importance...")

            shap_importance_df = pd.DataFrame(
                {
                    "Feature": features,
                    "Importance": np.abs(shap_values).mean(axis=0),
                }
            ).sort_values("Importance", ascending=False)
        else:
            logger.warning("Skipping SHAP feature importance")

            shap_importance_df = pd.DataFrame(
                {
                    "Feature": features,
                    "Importance": np.zeros(len(features)),
                }
            ).sort_values("Importance", ascending=False)

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        global_df = booster_importance_df.merge(
            perm_importance_df.rename(columns={"Importance": "permutation"}),
            on="Feature",
            how="left",
        ).merge(
            shap_importance_df.rename(columns={"Importance": "shap"}),
            on="Feature",
            how="left",
        )

        global_df.to_parquet(
            self.output_dir
            + f"feature_importance_{self.vars_}_{self.var_c}_{self.year}_{self.type_}_{self.model_version}.parquet",
            index=False,
        )

        logger.info(
            f"Feature importance saved to {self.output_dir} using multiple methods."
        )

        return (
            booster_importance_df,
            perm_importance_df,
            shap_importance_df,
        )

    def compute_shap_values(self, X):
        """Compute the average SHAP values and base value across the ensemble."""
        logger.info("Generating SHAP analysis...")

        features = (
            self.model.models[self.var_c].features
            if self.model.models[self.var_c].features is not None
            else self.model.features[self.var_c]
        )
        X_features = X[features]

        shap_values_per_model = []
        base_values = []

        for k in range(self.n_models):
            model = self.model.models[self.var_c].best_models[k]
            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(X_features)
            shap_values_per_model.append(np.asarray(shap_values, dtype=float))

            base_value = explainer.expected_value
            base_values.append(float(base_value))

        shap_values = np.mean(shap_values_per_model, axis=0)
        base_value = float(np.mean(base_values))

        self.shap_values_computed = True

        return shap_values, base_value, X_features.columns

    def save_shap_values(self, entities, features, shap_values, base_value):
        """Save SHAP values together with the entity identifier and base value."""

        shap_values_aug = pd.DataFrame(
            shap_values,
            index=entities,
            columns=features,
        )

        shap_values_aug = shap_values_aug.reset_index(names="codecommune")
        shap_values_aug["base_value"] = base_value

        shap_values_aug.to_parquet(
            self.output_dir
            + f"shap_values_{self.vars_}_{self.var_c}_{self.year}_{self.type_}_{self.model_version}.parquet",
            index=False,
        )

    def plot_shap_summary(self, shap_values):
        # Summary plot
        shap.summary_plot(shap_values, self.model.features[self.var], show=False)
        summary_plot_path = os.path.join(self.local_output_dir, "shap_summary_plot.png")
        plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"SHAP analysis saved to {self.local_output_dir}")

        return shap_values

    def plot_feature_impact(self, X, shap_values, batch_size=10, k=0):
        """
        Generate a series of plots to analyze the feature impact in batches.

        Args:
            features: DataFrame containing feature data.
            shap_values: SHAP explanation values for the features.
            model: Trained model for generating PDP and ALE plots.
            output_dir: Directory to save the plots.
            batch_size: Number of features to include in each batch of plots.

        Returns:
            None: The plots are saved to the specified path.
        """
        logger.info(
            f"Creating comprehensive explanation plots for {len(X.columns)} features"
        )
        features = (
            self.model.models[self.var_c].features
            if self.model.models[self.var_c].features is not None
            else self.model.features[self.var_c]
        )
        num_features = len(features)
        num_batches = (
            num_features + batch_size - 1
        ) // batch_size  # Calculate the number of batches

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_features)
            batch_features = features[start_idx:end_idx]

            logger.debug(
                f"\nProcessing batch {batch_idx + 1}/{num_batches} \
                (features {start_idx + 1} to {end_idx})"
            )

            # Create figure with subplots for the batch
            fig, axes = plt.subplots(
                len(batch_features), 5, figsize=(25, 5 * len(batch_features))
            )
            if len(batch_features) == 1:
                axes = [axes]  # Ensure axes is iterable for a single feature

            for i, feature in enumerate(batch_features):
                logger.debug(f"Feature {start_idx + i + 1}: {feature}")

                # 1. Partial Dependence Plot (PDP)
                try:
                    PartialDependenceDisplay.from_estimator(
                        estimator=self.model.models[self.var_c].best_models[k],
                        X=X[features].dropna(subset=[feature]),
                        features=[feature],
                        grid_resolution=50,
                        kind="both",
                        centered=True,
                        subsample=50,
                        random_state=0,
                        ax=axes[i][0],
                    )
                    axes[i][0].set_title("Partial Dependence Plot", fontweight="bold")
                except Exception as e:
                    axes[i][0].text(
                        0.5,
                        0.5,
                        f"Error: {str(e)[:30]}...",
                        ha="center",
                        va="center",
                        transform=axes[i][0].transAxes,
                    )
                    axes[i][0].set_title("PDP - Error", fontweight="bold")

                # 2. SHAP Dependence Plot
                try:
                    shap.dependence_plot(
                        feature, shap_values, X, ax=axes[i][1], show=False
                    )
                    axes[i][1].set_title("SHAP Dependence Plot", fontweight="bold")
                except Exception as e:
                    axes[i][1].text(
                        0.5,
                        0.5,
                        f"Error: {str(e)[:30]}...",
                        ha="center",
                        va="center",
                        transform=axes[i][1].transAxes,
                    )
                    axes[i][1].set_title("SHAP - Error", fontweight="bold")

                # 3. Accumulated Local Effect (ALE)
                try:
                    ale(
                        X=X[features].dropna(subset=[feature]),
                        model=self.model.models[self.var_c].best_models[k],
                        feature=[feature],
                        grid_size=50,
                        include_CI=True,
                        C=0.95,
                        ax=axes[i][2],
                        fig=fig,
                    )

                    axes[i][2].set_title("ALE Plot", fontweight="bold")
                except Exception as e:
                    axes[i][2].text(
                        0.5,
                        0.5,
                        f"Error: {str(e)}...",
                        ha="center",
                        va="center",
                        transform=axes[i][2].transAxes,
                    )
                    axes[i][2].set_title("ALE - Error", fontweight="bold")

                # 4. Feature Distribution
                sns.histplot(X[feature], kde=True, bins=30, color="blue", ax=axes[i][3])
                axes[i][3].set_title(f"Distribution of {feature}", fontweight="bold")
                axes[i][3].set_xlabel(feature)
                axes[i][3].set_ylabel("Frequency")

                # 5. Feature Statistics
                stats = X[feature]
                stats_text = (
                    f"μ:{stats.mean():.3f} σ:{stats.std():.3f}\nMin:{stats.min():.3f} \
                    Max:{stats.max():.3f}"
                )
                axes[i][4].text(
                    0.5,
                    0.5,
                    stats_text,
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
                )
                axes[i][4].set_title("Feature Statistics", fontweight="bold")
                axes[i][4].axis("off")

            plt.tight_layout()

            # Save the batch figure
            batch_path = os.path.join(
                self.local_output_dir, f"feature_batch_{batch_idx + 1}.png"
            )
            plt.savefig(batch_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            logger.debug(f"Batch {batch_idx + 1} saved to {batch_path}")

        logger.debug(f"\n{'=' * 60}\nAnalysis completed for all features!\n{'=' * 60}")

    def generate_tree_visualization(self, k=0):
        """Generate and save the tree visualization as an HTML file."""
        logger.info("Generating tree visualization...")

        # Ensure the model is a tree-based model
        if hasattr(self.model.models[self.var_c].best_models[k], "get_booster"):
            st = SuperTree(
                self.model.get_booster(),
                self.features,
                self.target,
                self.features.columns.to_list(),
                self.target_col,
            )

            # Visualize the tree
            tree_html_path = os.path.join(
                self.local_output_dir, "tree_visualization.html"
            )
            st.save_html(filename=tree_html_path)
            logger.info(f"Tree visualization saved to {tree_html_path}")
        else:
            logger.info("Model does not support tree visualization.")

    def explain(self, var, year, type_, vars_):
        """Run the explainability pipeline.
        The pipeline consists of multiple explanability tools that can be selected in the config.

        1. Perform pre-explanability task from the model object,
        such as loading a stratified sample from the training data.
        We are only dealing with one target variable.

        2. Build the explanability steps:
            - Feature importance computation (global) (mean across all boosting models)
            - Shap values computations (for all instances) (mean across all boosting models)
            - Feature impact (for one model of the meta-booster - 1st one) (global)
                - PDP
                - ALE
                - Shap Dependance Plot
            - Tree vizualisation (for one model of the meta-booster - 1st one)
            (for one tree of the boosting model - 1st one)

        3. Run the methods successively
        """
        self.type_ = type_
        self.t = 1 if self.type_ == "pres" else 0
        self.var = var
        self.var_c = self.var.replace("tau", "")
        self.year = year
        self.vars_ = vars_
        logger.info(
            f"Computing explain model for election: {self.year}|{self.type_} and variable: {self.var} ({self.vars_})"
        )

        ec = ExplainCore(self.var, self.year, self.t)

        # 0. Get model
        self.model, self.n_models = ec._load_model(
            data_path=self.data_path,
            var=self.var_c,
            year=self.year,
            type_=self.type_,
            vars_=self.vars_,
            model_version=self.model_version,
            fs=None,
        )

        data = DataLoader.load_dataset(
            self.model.data_paths[self.var],
            fs=None,
            formate="parquet",
            columns=None,
            filters=None,
            hive_partitioning=True,
            engine="polars",
        )

        # 1. Get sample data from model
        X, y, c = ec.run(data, frac=None)

        # 2. build step factory

        # 2.1. Shap values
        if "shap_values" in self.steps:
            shap_values, base_value, features = self.compute_shap_values(X)
            self.save_shap_values(
                entities=c,
                features=features,
                shap_values=shap_values,
                base_value=base_value,
            )
        if "plot_shap_values" in self.steps:
            self.plot_shap_summary(shap_values=shap_values)

        # 2.2. Feature Importance
        if "feature_importance" in self.steps:
            if "shap_values" not in self.steps:
                shap_values = None
            _ = self.generate_feature_importance(X=X, y=y, shap_values=shap_values)

        # 2.3. Feature Impact
        if "feature_impact" in self.steps:
            self.plot_feature_impact(
                X=X,
                shap_values=shap_values,
            )

        # 2.4. Tree visualization
        if "tree" in self.steps:
            self.generate_tree_visualization()

        logger.success("Explain pipeline done")

    def run(self):
        """Runs the full explainability pipeline."""
        years = self.config.years
        types = self.config.types
        vars_ = self.config.vars_
        for type_ in types:
            for year in years[type_]:
                for vs in vars_:
                    for var in vs:
                        self.explain(var, year, type_, vs)


if __name__ == "__main__":
    Explainer().run()
