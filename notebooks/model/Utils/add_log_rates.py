import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize


def add_logarithm_turnout_rates(dfs_by_file, winsor=False):
    """
    This function processes a dictionary of DataFrames, computes the 'tau' values
    for the turnout rate at the first round of each election loaded onto the notebook.
    """
    # Create a dictionary to store selected columns with valid tau
    selected_columns_df = {}

    for key, df in dfs_by_file.items():
        p = np.where(df["inscrits"] == 0.0, -1.0, df["votants"] / df["inscrits"])

        with np.errstate(divide="ignore", invalid="ignore"):
            df["tau"] = np.where(
                p == -1.0,
                0.0,
                np.where(
                    p == 0.0,
                    0.5 / df["inscrits"],
                    np.where(
                        p == 1.0,
                        1 - 0.5 / df["inscrits"],
                        np.where((p > 0.0) & (p < 1.0), np.log(p / (1 - p)), 0.0),
                    ),
                ),
            )

        if winsor:
            df["tau"] = winsorize(df["tau"], limits=[0.01, 0.01])

        selected_columns_df[key] = df[["codecommune", "tau", "inscrits"]]

    return selected_columns_df


def add_logarithm_turnout_rates_T2(dfs_by_file, winsor=False):
    """
    This function processes a dictionary of DataFrames, computes the 'tau' values
    for the turnout rate at the second round of elections.
    """
    selected_columns_df = {}

    for key, df in dfs_by_file.items():
        if "pres" in key and "1848" not in key:
            p = np.where(
                df["inscritsT2"] == 0.0, -1.0, df["votantsT2"] / df["inscritsT2"]
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                df["tau"] = np.where(
                    p == -1.0,
                    0.0,
                    np.where(
                        p == 0.0,
                        0.5 / df["inscritsT2"],
                        np.where(
                            p == 1.0,
                            1 - 0.5 / df["inscritsT2"],
                            np.where((p > 0.0) & (p < 1.0), np.log(p / (1 - p)), 0.0),
                        ),
                    ),
                )

            df["inscrits"] = df["inscritsT2"]

            if winsor:
                df["tau"] = winsorize(df["tau"], limits=[0.01, 0.01])

            selected_columns_df[key] = df[["codecommune", "tau", "inscrits"]]

    return selected_columns_df


# Only applies to ref and presT2 elections
def add_logarithm_WV_rate(dfs_by_file, winsor=False):
    pass


def add_logarithm_tendency_rate(dfs_by_file, tendency="TG", winsor=False):
    selected_columns_df = {}

    m = "pvote" + tendency

    for key, df in dfs_by_file.items():
        if m not in df.columns:
            continue

        p = df[m].fillna(0)

        with np.errstate(divide="ignore", invalid="ignore"):
            df["tau"] = np.where(
                p == 0.0,
                0.5 / df["inscrits"],
                np.where(
                    p == 1.0,
                    1 - 0.5 / df["inscrits"],
                    np.where((p > 0.0) & (p < 1.0), np.log(p / (1 - p)), 0.0),
                ),
            )

        df["tau"] = df["tau"].fillna(0)

        if winsor:
            df["tau"] = winsorize(df["tau"], limits=[0.01, 0.01])

        if "ref" not in key:
            selected_columns_df[key] = df[["codecommune", "tau", "inscrits"]]

    return selected_columns_df


def add_logarithm(dfs_by_file, type="all", metric="tau", poltrend="TG", winsor=False):
    if metric == "tau_poltrend":
        return add_logarithm_tendency_rate(
            dfs_by_file, tendency=poltrend, winsor=winsor
        )

    if metric == "tau_WV":
        return add_logarithm_WV_rate(dfs_by_file, winsor)

    if metric == "tau":
        if type == "T1":
            return add_logarithm_turnout_rates(dfs_by_file, winsor)
        if type == "T2":
            return add_logarithm_turnout_rates_T2(dfs_by_file, winsor)
        if type == "all":
            s1 = add_logarithm_turnout_rates(dfs_by_file, winsor)
            s2 = add_logarithm_turnout_rates_T2(dfs_by_file, winsor)
            merged = {**s1, **{key + "/T2": value for key, value in s2.items()}}
            return merged

    if metric == "tau_LTR":
        sl = add_logarithm_turnout_rates(dfs_by_file, winsor)
        for key in sl.keys():
            df = sl[key].copy()
            tau = df.copy()  # Extract the 'tau' column
            # Compute adjusted tau
            tau_avg = np.mean(tau)
            tau_std = np.std(tau)
            adjusted_tau = (tau - tau_avg) / tau_std
            sl[key]["tau"] = adjusted_tau
        return sl


def create_size_bins(df, num_bins, criteria="inscrits"):
    """
    Assign communes to bins where each bin has approximately the same number of communes
    based on their 'inscrits' values.

    Parameters:
    df (dict): A dictionary where keys are election identifiers and values are DataFrames
                                containing 'codecommune', 'tau', and 'inscrits' columns.

    Returns:
    commune_counts (dict): A dictionary where keys are election identifiers and values are
                           the counts of communes in each bin.
    """

    if criteria == "inscrits":
        global_min = 0.0
        global_max = 1.362500e07
    elif criteria == "pib":
        global_min = 0.0
        global_max = 1
    else:
        global_min = 0.0
        global_max = 1.0

    # Sort the communes by 'inscrits' to ensure we can divide them evenly
    try:
        df_sorted = df.sort_values(criteria)
    except:
        df_sorted = df

    d = df_sorted[criteria].dropna()

    # Calculate the bin edges based on the percentiles of 'inscrits'
    bin_edges = np.percentile(d, np.linspace(0, 100, num_bins + 1))
    #
    bin_edges = bin_edges[:-1]

    bin_edges = np.concatenate(
        [np.array([global_min]), bin_edges, np.array([global_max])]
    )

    # Ensure bin_edges are sorted
    bin_edges = np.sort(bin_edges)

    df = df.copy()

    # Assign each 'inscrits' value to a bin using np.digitize
    df.loc[:, "bin"] = np.digitize(df[criteria], bin_edges) - 1

    bin_avg_values = df.groupby("bin")[criteria].mean()

    # Count how many communes are in each bin
    commune_count = df["bin"].value_counts().sort_index()

    return commune_count, bin_edges, bin_avg_values


def create_size_bins_(df, num_bins, criteria="inscrits"):
    """
    Assign communes to bins with approximately equal counts based on the specified criteria
    (such as 'inscrits') using logarithmic binning.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'codecommune', 'tau', and 'inscrits' columns.
    num_bins (int): Number of bins to create.
    criteria (str): The column name on which to base the binning (default is 'inscrits').

    Returns:
    commune_counts (pd.Series): Count of communes in each bin.
    bin_edges (np.array): The edges of the bins.
    bin_avg_values (pd.Series): The average value of 'criteria' in each bin.
    """
    # Ensure 'df' has no missing values in the 'criteria' column
    df = df.dropna(subset=[criteria])

    # Get the minimum and maximum values of the 'criteria' column
    min_value = df[criteria].min()
    max_value = df[criteria].max()

    # Create logarithmic bin edges based on the data's min and max values
    log_min = np.log10(min_value + 1)  # Avoid log(0) by adding 1
    log_max = np.log10(max_value + 1)  # Avoid log(0) by adding 1
    bin_edges = np.logspace(
        log_min, log_max, num_bins + 1
    )  # Create evenly spaced bins on log scale

    # Ensure bin edges are sorted
    bin_edges = np.sort(bin_edges)

    df = df.copy()
    # Assign each 'criteria' value to a bin using np.digitize
    df["bin"] = np.digitize(df[criteria], bin_edges) - 1

    # Calculate the mean value of 'criteria' in each bin
    bin_avg_values = df.groupby("bin")[criteria].mean()

    # Count how many communes are in each bin
    commune_counts = df["bin"].value_counts().sort_index()

    return commune_counts, bin_edges, bin_avg_values


def compute_tau_means(data, num_bins, criteria="inscrits"):
    """
    Compute the mean of 'tau' values for each bin defined by 'size_bins' across multiple DataFrames.

    Parameters:
        size_bins: Array-like, edges of bins for categorizing 'inscrits'.
        data: Dictionary of DataFrames containing 'inscrits' and 'tau' columns.

    Returns:
        Dictionary of Series where each Series contains the mean 'tau' value for each bin.
        Missing bins are filled with 0.
    """

    # Initialize dictionary to store results
    tau_mean_bin = {}
    tau_std_bin = {}
    bin_avg_values_dict = {}

    # Process each DataFrame in the input dictionary
    for key, df in data.items():
        df = df.copy()
        df.dropna()

        commune_counts, size_bins, bin_avg_values = create_size_bins(
            df, num_bins, criteria
        )
        bin_indices = np.arange(len(size_bins) - 1)

        # Skip if DataFrame is empty or lacks required columns
        if df.empty or criteria not in df.columns or "tau" not in df.columns:
            tau_mean_bin[key] = pd.Series(0, index=bin_indices, dtype=float)
            continue

        # Assign each 'inscrits' value to a bin using np.digitize

        df.loc[:, "bin"] = np.digitize(df[criteria], size_bins) - 1

        # Compute the mean 'tau' value for each bin
        bin_avg_value = df.groupby("bin")[criteria].mean()
        tau_mean = df.groupby("bin")["tau"].mean()
        tau_std = df.groupby("bin")["tau"].std()

        # Align the resulting Series to have an entry for every bin, filling missing values with 0
        tau_mean_aligned = pd.Series(0, index=bin_indices, dtype=float)
        tau_mean_aligned.update(tau_mean)
        tau_std_aligned = pd.Series(0, index=bin_indices, dtype=float)
        tau_std_aligned.update(tau_std)
        bin_avg_value_aligned = pd.Series(0, index=bin_indices, dtype=float)
        bin_avg_value_aligned.update(bin_avg_value)

        # Store the aligned Series in the result dictionary
        tau_mean_bin[key] = tau_mean_aligned[5:]
        tau_std_bin[key] = tau_std_aligned[5:]
        bin_avg_values_dict[key] = bin_avg_value_aligned[5:]

    return tau_mean_bin, tau_std_bin, bin_avg_values_dict


def create_equally_distributed_bins(data, criteria, num_bins):
    bin_edges = np.percentile(data[criteria], np.linspace(0, 100, num_bins - 1))
    return bin_edges


def create_equally_spaced_bins(data, criteria, num_bins):
    # Get the minimum and maximum values of the data for the given criteria
    min_val = np.min(data[criteria])
    max_val = np.max(data[criteria])

    # Generate equally spaced bin edges between min_val and max_val
    bin_edges = np.linspace(
        min_val, max_val, num_bins + 1
    )  # num_bins + 1 to include the upper edge

    return bin_edges


def compute_tau_means_2D_bins(data, criteria="pib"):
    """
    Compute the mean of 'tau' values for each bin defined by 'size_bins' across multiple DataFrames.

    Parameters:
        size_bins: Array-like, edges of bins for categorizing 'inscrits'.
        data: Dictionary of DataFrames containing 'inscrits' and 'tau' columns.

    Returns:
        Dictionary of Series where each Series contains the mean 'tau' value for each bin.
        Missing bins are filled with 0.
    """

    # Initialize dictionary to store results
    result = {}
    s_bins_values = {}
    p_bins_values = {}

    # Process each DataFrame in the input dictionary
    for key, df in data.items():
        _, size_bins, _ = create_size_bins(data[key], 50, "inscrits")
        pib_bins = create_equally_distributed_bins(
            data[key], criteria, 4
        )  # Remettre 4 pour sép mediane

        num_bins_pib = len(pib_bins) - 1
        num_bins_inscrits = len(size_bins) - 1
        # Skip if DataFrame is empty or lacks required columns
        if (
            df.empty
            or "inscrits" not in df.columns
            or "tau" not in df.columns
            or criteria not in df.columns
        ):
            continue

        # Assign each 'inscrits' value to a bin using np.digitize
        df = df.copy()

        df["bin_inscrits"] = np.digitize(df["inscrits"], size_bins) - 1
        df["bin_" + criteria] = np.digitize(df[criteria], pib_bins) - 1

        size_bins_avg_values = df.groupby("bin_inscrits")["inscrits"].mean()
        pib_bins_avg_values = df.groupby("bin_" + criteria)[criteria].median()

        # Filter invalid bins
        valid_bins = (
            (df["bin_inscrits"] >= 0)
            & (df["bin_inscrits"] < num_bins_inscrits)
            & (df["bin_" + criteria] >= 0)
            & (df["bin_" + criteria] < num_bins_pib)
        )
        df = df[valid_bins]

        # Compute mean and standard deviation for each bin combination
        grouped = (
            df.groupby(["bin_inscrits", "bin_" + criteria])["tau"]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Create an empty grid
        grid_mean = np.zeros((num_bins_inscrits, num_bins_pib))
        grid_std = np.zeros((num_bins_inscrits, num_bins_pib))

        # Populate the grid with computed values
        for _, row in grouped.iterrows():
            grid_mean[int(row["bin_inscrits"]), int(row["bin_" + criteria])] = row[
                "mean"
            ]
            grid_std[int(row["bin_inscrits"]), int(row["bin_" + criteria])] = row["std"]

        # Combine the grids into a single DataFrame for easier use
        result[key] = {
            "mean": pd.DataFrame(
                grid_mean, index=range(num_bins_inscrits), columns=range(num_bins_pib)
            ),
            "std": pd.DataFrame(
                grid_std, index=range(num_bins_inscrits), columns=range(num_bins_pib)
            ),
        }

        s_bins_values[key] = size_bins_avg_values
        p_bins_values[key] = (pib_bins_avg_values, np.median(data[key][criteria]))

    return result, s_bins_values, p_bins_values
