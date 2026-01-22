import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import winsorize


def plot_m(
    tau_mean_bin,
    bin_avg_values,
    election_type="all",
    years_to_remove=None,
    criteria="inscrits",
):
    PHI = {}
    if years_to_remove is None:
        years_to_remove = []  # Initialize as an empty list if not provided

    # Create figure for plotting
    fig, ax = plt.subplots(figsize=(20, 10), dpi=80)

    # Define color map for election years (you can change the colormap)
    colormap = cm.plasma  # You can choose another colormap, like 'viridis'
    years = sorted(
        set(int(key.split("/")[1][:4]) for key in tau_mean_bin.keys())
    )  # Extract unique years
    normalize = plt.Normalize(
        min(years), max(years)
    )  # Normalize to map years to colors
    colors = {
        year: colormap(normalize(year)) for year in years
    }  # Color map for each year

    for key, df in tau_mean_bin.items():
        # Check for election type and exclude if not matching the requested type
        if election_type != "all" and election_type not in key:
            continue  # Skip if the election type doesn't match the requested type

        # Exclude specific years if they're in the years_to_remove list
        year = int(key.split("/")[1][:4])  # Extract the year from the key

        if year in years_to_remove:

            continue  # Skip this key if it's in the exclusion list

        # Determine the color based on the election year
        color = colors.get(year, "black")  # Get the color mapped to the year

        if "T2" in key:
            linestyle = "-"
            round = "T2"
        else:
            linestyle = ":"
            round = "T1"

        PHI[key] = df

        ax.plot(
            bin_avg_values[key].values,
            df,
            label=key.split("/")[0] + "/" + key.split("/")[1] + "/" + round,
            linestyle=linestyle,
            marker="o",
            linewidth=0.5,
            markersize=2,
            color=color,
        )

    # ax.set_yscale("log")

    # Set the x-axis to a logarithmic scale
    ax.set_xscale("log")

    # Add labels and title
    ax.set_ylabel("Average of tau for the bin")
    ax.set_xlabel(criteria)
    ax.set_title(f"Average value, $m_{criteria}$")

    # Add a legend
    ax.legend()

    # Show colorbar to indicate the year
    sm = cm.ScalarMappable(cmap=colormap, norm=normalize)
    sm.set_array([])  # Set an empty array for the color mapping
    cbar = plt.colorbar(sm, ax=ax)  # Pass the 'ax' explicitly
    cbar.set_label("Year")  # Label for the colorbar

    # Display the plot
    plt.tight_layout()
    plt.show()

    return PHI


def plot_sigma(
    tau_mean_bin,
    tau_std_bin,
    bin_avg_values,
    election_type="pres",
    elections_to_remove=None,
    criteria="inscrits",
):
    if elections_to_remove is None:
        elections_to_remove = []  # Initialize as an empty list if not provided

    BPHI = {}
    X = {}
    Y = {}

    # Define line styles and markers for different election types
    line_styles = {"pres": "-", "leg": "--", "ref": "-."}
    marker_styles = {"pres": "o", "leg": "s", "ref": "D"}

    # Extract years from the keys, handle cases like '1871juil'
    years = sorted(
        set(
            int(key.split("/")[1][:4])
            for key in tau_std_bin.keys()
            if "pres" in key or "leg" in key or "ref" in key
        )
    )

    # Manually include 1871 if it's present in the keys
    years = [year if year != "1871juil" else "1871" for year in years]

    # Create a colormap based on the years
    colormap = (
        cm.plasma
    )  # Choose the color map (can change to others like 'viridis', 'inferno')
    normalize = plt.Normalize(min(years), max(years))
    colors = {year: colormap(normalize(year)) for year in years}

    # Create the figure (only one plot as specified)
    fig, ax = plt.subplots(figsize=(20, 10), dpi=80)

    # Set the plot title based on the election type
    if election_type == "pres":
        ax.set_title("Presidential Elections")
    elif election_type == "leg":
        ax.set_title("Legislative Elections")
    elif election_type == "ref":
        ax.set_title("Referendum Elections")
    else:
        ax.set_title("All Elections")

    # Set axis labels and grid
    ax.set_ylabel("Sigma_n")
    ax.set_xlabel(f"{criteria}")
    ax.grid(True)

    for key, df in tau_std_bin.items():
        # If the user specified an election type, only plot that type
        if election_type != "all" and election_type not in key:
            continue  # Skip non-matching election types

        # If the election is in the removal list, skip it
        if any(election in key for election in elections_to_remove):
            continue  # Skip this key if it's in the exclusion list

        # Extract the election type from the key
        if "pres" in key:
            current_election_type = "pres"
        elif "leg" in key:
            current_election_type = "leg"
        elif "ref" in key:
            current_election_type = "ref"
        else:
            continue  # Skip if the election type is not recognized

        # Extract the year correctly (handle cases like '1871juil')
        year = int(key.split("/")[1][:4])  # Get the first 4 characters (year) from key

        if "1871juil" in key:
            year = 1871  # Adjust year for special case

        # Get the color for the year
        color = colors.get(year, "black")  # Default to black if year not in map

        # Get line style and marker based on election type
        # line_style = line_styles.get(current_election_type, "-")
        marker_style = marker_styles.get(current_election_type, "x")

        # Calculate the values
        p = tau_mean_bin[key]
        x = df
        y = bin_avg_values[key].values

        # Create label with election type and year
        label = f"{key.split('/')[1]} ({year})"

        if "T2" in key:
            line_style = "-"
            round = "T2"
        else:
            line_style = ":"
            round = "T1"

        # Plot data with dynamic line style, marker, and color
        ax.plot(
            y,
            x,
            label=key.split("/")[1] + "/" + label + "/" + round,
            color=color,
            linestyle=line_style,
            marker=marker_style,
            linewidth=1.5,
            markersize=6,
        )

        # ax.set_xscale('log')

        # Save the last value of x for BPHI
        BPHI[key] = x.iloc[-1]
        X[key] = x
        Y[key] = y

    # Show the legend
    ax.legend()

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Show colorbar to indicate the year
    sm = cm.ScalarMappable(cmap=colormap, norm=normalize)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)  # Explicitly link the colorbar to the axes
    cbar.set_label("Year")

    # Display the plot
    plt.tight_layout()
    plt.show()

    return X, BPHI


def plot_estimation_results_winsor(results, cutoff_year=2000):
    df = results.copy()
    params = list(results.columns)
    params.remove("Key")

    df["Year"] = df["Key"].str.extract(r"(\d{4})").astype(float)
    df["Election_type"] = df["Key"].str.split("/").str.get(0)
    df["Round"] = df["Key"].str.contains("/T2").map({True: "T2", False: "T1"})

    df = df.dropna(subset=params)
    df = df[df["Year"] >= cutoff_year]

    def apply_winsorize(df, column):
        winsorized_data = winsorize(df[column], limits=[0.05, 0.05])
        df[f"{column}_winsorized"] = winsorized_data
        df[f"{column}_outliers"] = df[column] != winsorized_data
        return df

    for col in params:
        df = apply_winsorize(df, col)

    fig, axs = plt.subplots(len(params), 1, figsize=(12, 4 * len(params)), sharex=True)
    if len(params) == 1:
        axs = [axs]

    colors = plt.cm.tab10.colors

    for ax, coeff, color in zip(axs, params, colors):
        for round_type in df["Round"].unique():
            marker = "o" if round_type == "T1" else "x"
            for election_type in df["Election_type"].unique():
                linestyle = (
                    "-"
                    if "pres" in election_type
                    else ":" if "leg" in election_type else "--"
                )
                subset = df[
                    (df["Round"] == round_type) & (df["Election_type"] == election_type)
                ]
                ax.scatter(
                    subset["Year"],
                    subset[f"{coeff}_winsorized"],
                    marker=marker,
                    linestyle=linestyle,
                    label=f"{coeff} ({round_type}, {election_type})",
                    color=color,
                )
                outlier_subset = subset[subset[f"{coeff}_outliers"]]
                ax.scatter(
                    outlier_subset["Year"],
                    outlier_subset[coeff],
                    color="red",
                    label=f"{coeff} Outliers",
                )

        ax.set_ylabel(coeff)
        ax.legend(loc="best")
        ax.grid(True)
        if "R2" in coeff:
            ax.set_ylim(0, 1)
            ax.axhline(0.95, color="red", linestyle="--")
        if "omega" in coeff:
            ax.set_ylim(-1, 0)

    axs[-1].set_xlabel("Year")
    fig.suptitle(
        f"Evolution of coefficients over time (Winsorized with Outliers Highlighted)",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_estimation_results(results, remove_outl=True, R2lim=0.0):
    df = results.copy()
    params = list(results.columns)
    params.remove("Key")
    # Extract year and round
    df["Year"] = df["Key"].str.extract(r"(\d{4})").astype(float)
    df["Election_type"] = df["Key"].str.split("/").str.get(0)
    df["Round"] = df["Key"].str.contains("/T2").map({True: "T2", False: "T1"})

    # Drop NaNs
    df = df.dropna(subset=params)

    # Only keep above R2 threshold
    df = df[df["R2"] > R2lim]

    # Function to remove outliers using IQR
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

    # Remove outliers for each coefficient
    for col in params:
        if remove_outl:
            df = remove_outliers(df, col)

    # Plot coefficients over time
    fig, axs = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

    coefficients = params
    colors = ["b", "r", "g", "purple"][: len(params)]

    for ax, coeff, color in zip(axs, coefficients, colors):
        for round_type in df["Round"].unique():
            if round_type == "T1":
                marker = "o"
            else:
                marker = "x"
            for election_type in df["Election_type"].unique():
                if "pres" in election_type:
                    linestyle = "-"
                elif "leg" in election_type:
                    linestyle = ":"
                else:
                    linestyle = "--"
            subset = df[df["Round"] == round_type]
            ax.scatter(
                subset["Year"],
                subset[coeff],
                marker=marker,
                linestyle=linestyle,
                label=f"{coeff} ({round_type})",
                color=color,
            )

        ax.set_ylabel(coeff)
        ax.legend()
        ax.grid()
        if "R2" in coeff:
            ax.set_ylim(0, 1)
            ax.axhline(0.95, color="red", linestyle="-")
        if "omega" in coeff:
            ax.set_ylim(-1, 0)

    axs[-1].set_xlabel("Year")
    fig.suptitle(
        f"Evolution of {params} coefficients over time (Outliers Removed {remove_outl})",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()


def plot_estimation_results_up(results, remove_outl=True, R2lim=0.0, n_plots=5):
    df = results.copy()
    params = list(results.columns)
    params.remove("Key")

    df["Year"] = df["Key"].str.extract(r"(\d{4})").astype(float)
    df["Election_type"] = df["Key"].str.split("/").str.get(0)
    df["Round"] = df["Key"].str.contains("/T2").map({True: "T2", False: "T1"})

    df = df.dropna(subset=params)
    df = df[df["R2"] > R2lim]

    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

    if remove_outl:
        for col in params:
            df = remove_outliers(df, col)

    fig, axs = plt.subplots(
        min(n_plots, len(params)), 1, figsize=(12, 3 * n_plots), sharex=True
    )
    if n_plots == 1:
        axs = [axs]

    colors = {"pres": "purple", "leg": "green", "other": "red"}
    linestyles = {"T1": "-", "T2": "--"}

    for ax, coeff in zip(axs, params[:n_plots]):
        for (election_type, round_type), subset in df.groupby(
            ["Election_type", "Round"]
        ):
            color = next((v for k, v in colors.items() if k in election_type), "purple")
            linestyle = linestyles.get(round_type, ":")
            marker = "o" if round_type == "T1" else "x"
            ax.scatter(
                subset["Year"],
                subset[coeff],
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=f"{election_type} {round_type}",
            )

        ax.set_ylabel(coeff)
        ax.legend()
        ax.grid()
        if "R2" in coeff:
            ax.set_ylim(0, 1)
            ax.axhline(0.95, color="red", linestyle="-")
        if "omega" in coeff:
            ax.set_ylim(-1, 0)

    axs[-1].set_xlabel("Year")
    fig.suptitle(
        f"Evolution of coefficients over time (Outliers Removed: {remove_outl})",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()


def analyze_model(result_2D, sbv, pbvm, result_m, result_sigma, B_inf, criteria="pib"):
    colormap = cm.viridis

    # Number of plots per row
    plots_per_row = 3

    keys = list(result_2D.keys())
    n_rows = (len(keys) + plots_per_row - 1) // plots_per_row
    fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    lambda_m = {}
    lambda_sigma = {}

    for i, key in enumerate(keys):
        pbv, m = pbvm[key]
        mean_df = result_2D[key]["mean"]
        std_df = result_2D[key]["std"]

        X = sbv[key][1:]
        Y_1, Z_1 = mean_df[0][1 : len(X) + 1], std_df[0][1 : len(X) + 1] ** 2
        Y_2, Z_2 = mean_df[1][1 : len(X) + 1], std_df[1][1 : len(X) + 1] ** 2

        A = result_m[result_m["Key"] == key]["A"].values[0]
        B = result_m[result_m["Key"] == key]["B"].values[0]
        c = result_m[result_m["Key"] == key]["c"].values[0]
        sim_m = A + B * (X**c)

        A_1 = result_sigma[result_sigma["Key"] == key]["A1"].values[0]
        A_2 = result_sigma[result_sigma["Key"] == key]["A2"].values[0]
        omega = result_sigma[result_sigma["Key"] == key]["omega"].values[0]
        b_inf = B_inf[key]
        sim_sigma = A_1 / X + A_2 * (X ** (omega)) + b_inf

        def get_label(pbv, m, i):
            return (
                f"{criteria} < {m} (m)" if pbv.iloc[i] < m else f"{criteria} > {m} (m)"
            )

        ax = axes[i]
        ax.plot(
            X,
            Y_1,
            label="m " + get_label(pbv, m, 0),
            color=colormap(0),
            marker="o",
            linestyle="-",
        )

        ax.plot(
            X,
            Y_2,
            label="m " + get_label(pbv, m, 1),
            color=colormap(0.5),
            marker="o",
            linestyle="-",
        )

        ax.plot(
            X,
            Z_1,
            label="sigma " + get_label(pbv, m, 0),
            color=colormap(0),
            marker="x",
            linestyle="--",
        )

        ax.plot(
            X,
            Z_2,
            label="sigma " + get_label(pbv, m, 1),
            color=colormap(0.5),
            marker="x",
            linestyle="--",
        )

        X = np.array(X)
        Y_ratio = Y_1 / Y_2

        # Calculate IQR
        Q1 = np.percentile(Y_ratio, 25)
        Q3 = np.percentile(Y_ratio, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers
        mask = (Y_ratio >= lower_bound) & (Y_ratio <= upper_bound)
        X_filtered = X[mask]
        Y_filtered = Y_ratio[mask]

        # Plot without outliers
        ax.plot(
            X_filtered, Y_filtered, label="lambda_m", color="orange", linestyle="--"
        )
        # ax.plot(X, Y_1 / Y_2, label="lambda_m", color="orange", linestyle="--")
        ax.axhline(np.mean(Y_filtered), color="orange", linestyle="-")
        lambda_m[key] = np.mean(Y_filtered)
        ax.plot(X, Z_1 / Z_2, label="lambda_sigma", color="red", linestyle="--")
        ax.axhline(np.mean(Z_1 / Z_2), color="red", linestyle="-")
        lambda_sigma[key] = np.mean(Z_1 / Z_2)
        ax.plot(X, sim_m, label="Simulation model for m", color="gray", linestyle=":")
        ax.plot(
            X,
            sim_sigma,
            label="Simulation model for sigma",
            color="gray",
            linestyle=":",
        )

        ax.set_title(f"{key}")
        ax.set_xlabel("Inscrits Bins")
        ax.set_xscale("log")
        # ax.set_yscale('log')
        ax.set_ylabel("Values")
        ax.legend(fontsize=8)
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    years = np.array([int(key.split("/")[1]) for key in lambda_m.keys()])
    values = np.array(list(lambda_m.values()))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(years, values, marker="o", linestyle="-")
    plt.title("Lambda m in Function of Year")
    plt.xlabel("Year")
    plt.ylabel("Lambda m")
    plt.ylim(0.8, 1.2)
    plt.grid(True)
    plt.show()

    years_ = np.array([int(key.split("/")[1]) for key in lambda_sigma.keys()])
    values_ = np.array(list(lambda_sigma.values()))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(years_, values_, marker="o", linestyle="-")
    plt.title("Lambda sigma in Function of Year")
    plt.xlabel("Year")
    plt.ylabel("Lambda sigma")
    plt.ylim(0.8, 1.2)
    plt.grid(True)
    plt.show()

    return lambda_m, lambda_sigma


def compare_fit(key, result_m, result_sigma, bin_avg_values, sigma_N, m_N, B_inf):
    N = bin_avg_values[key]
    sigma = sigma_N[key] ** 2
    m = m_N[key]

    if result_m is not None:
        A_ = result_m[result_m["Key"] == key]["A"].values[0]
        B_ = result_m[result_m["Key"] == key]["B"].values[0]
        c = result_m[result_m["Key"] == key]["c"].values[0]
        R2m = result_m[result_m["Key"] == key]["R2"].values[0]
        sim_ = A_ + B_ * (N**c)

    if result_sigma is not None:
        A_1 = result_sigma[result_sigma["Key"] == key]["A1"].values[0]
        A_2 = result_sigma[result_sigma["Key"] == key]["A2"].values[0]
        omega = result_sigma[result_sigma["Key"] == key]["omega"].values[0]
        R2sigma = result_sigma[result_sigma["Key"] == key]["R2"].values[0]
        b_inf = B_inf[key]
        sim__ = A_1 / N + A_2 * (N ** (omega)) + b_inf

    # Création du graphique log-log
    plt.figure(figsize=(8, 6))

    # Plot des données réelles et simulées en échelle logarithmique

    if result_m is not None:
        plt.loglog(N, m, "o-", label="Données réelles (m)", color="r")
        plt.loglog(N, sim_, "x-", label=f"Model_fit (m) : R^2 = {R2m}", color="g")
        plt.xlabel("N")
        plt.ylabel("sigma_N^2")
        plt.title(f"Comparaison des données réelles et simulées pour {key}")
        plt.legend()
    if result_sigma is not None:
        plt.loglog(N, sigma, "o-", label="Données réelle (sigma)", color="b")
        plt.loglog(
            N, sim__, "x-", label=f"Model_fit (sigma) : R^2 = {R2sigma}", color="purple"
        )
        plt.xlabel("N")
        plt.ylabel("sigma_N^2")
        plt.title(f"Comparaison des données réelles et simulées pour {key}")
        plt.legend()

        # Affichage du graphique
        plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()
