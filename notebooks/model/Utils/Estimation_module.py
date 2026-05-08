import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def f_1(m_N, bin_avg_values, plot=False, show_result=False):
    results = []
    election_not_computed = []

    # Define the function to fit
    def model(x, A, B, c):
        return A + B * x**c

    # lower_bounds = [0, -1000, 0]  # Lower bounds for [a, b, c]
    # upper_bounds = [1000, 1000, 1]  # Upper bounds for [a, b, c

    # Loop through each key in the dictionaries
    for key in m_N.keys():
        try:
            x_data = bin_avg_values[key].to_numpy()
            y_data = m_N[key].to_numpy()

            # Fit the curve with increased maxfev
            params, covariance = curve_fit(
                model,
                x_data,
                y_data,
                p0=(y_data[0], 0, 0.5),
                # bounds=([0.0, -1.0, 0.0], [y_data[0] + 1, 0.0, 1.0]),
                # Point de départ : A prendre un meilleur point A = moyenne des données
                # Prendre autre chose que le R^2 car non linéaire
                # bounds=([0.0, -10.0, 0.0], [10.0, 0.0, 1.0]),
                maxfev=100000,
            )

            A_fit, B_fit, c_fit = params

            # Calculate R² score
            y_pred = model(x_data, A_fit, B_fit, c_fit)
            r2 = r2_score(y_data, y_pred)
            mse = np.mean((np.array(y_data) - np.array(y_pred)) ** 2)

            # Store results
            results.append(
                {"Key": key, "A": A_fit, "B": B_fit, "c": c_fit, "R2": r2, "MSE": mse}
            )

            if plot:
                # Plot the data and the fit
                plt.figure(figsize=(8, 6))
                plt.scatter(x_data, y_data, label="Data", color="blue")
                plt.plot(
                    x_data,
                    y_pred,
                    label=f"Fit: A={A_fit:.2f}, B={B_fit:.2f}, c={c_fit:.2f}\nR2={r2:.4f}",
                    color="red",
                )
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.title(f"Fit for {key}")
                plt.grid()
                plt.show()
        except Exception:
            election_not_computed.append(key)

    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)

    # Print the results as a table
    if show_result:
        print(results_df)

    return results_df, election_not_computed


def f_2(m_N):
    """
    Compute the average value of the series for each key in the dictionary and return as a sorted DataFrame.

    Parameters:
    m_N (dict): A dictionary where each key maps to a list or iterable of numeric values.

    Returns:
    pd.DataFrame: A DataFrame with keys as index and average values as a column, sorted by key.
    """
    averages = {}
    for key, values in m_N.items():
        if not values.empty:  # Ensure the list is not empty
            averages[key] = sum(values) / len(values)
        else:
            averages[key] = None  # Handle empty lists if needed

    df = (
        pd.DataFrame(list(averages.items()), columns=["Key", "K"])
        .sort_values(by="K")
        .reset_index(drop=True)
    )
    return df


def f_3(sigma_N_squared_dict, N_dict):
    """
    Estime les paramètres A et B pour chaque série dans les dictionnaires :
    sigma_N^2(N) = A/N + B

    Arguments :
    sigma_N_squared_dict : dict de pandas Series ou listes, avec des valeurs de sigma_N^2(N)
    N_dict : dict de pandas Series ou listes, avec des valeurs de N

    Retourne :
    dict avec les estimations de A et B pour chaque série
    """

    # Dictionnaire pour stocker les résultats
    results = {}

    # Itérer sur chaque série dans les dictionnaires
    for key in sigma_N_squared_dict:
        sigma_N_squared = sigma_N_squared_dict[key]
        N = N_dict[key]

        # Calcul de 1/N
        inv_N = 1 / np.array(N)

        # Régression linéaire
        slope, intercept, r_value, _, _ = stats.linregress(inv_N, sigma_N_squared)

        # Estimation de A et B
        A = slope
        B = intercept
        R2 = r_value**2  # Coefficient de détermination

        # Sauvegarder les résultats dans le dictionnaire
        results[key] = {"Key": key, "A": A, "B": B, "R2": R2}

    df_results = pd.DataFrame.from_dict(results, orient="index")
    return df_results


def f_4(gamma_N, bin_avg_values, compute_h=False, PI={}):
    results = []
    elections_not_computed = []

    def model(N, A1, A2, omega):
        return A1 * N ** (-1) + A2 * N**omega

    for key in gamma_N.keys():
        N = bin_avg_values[key]
        gamma = gamma_N[key]
        pi = PI[key]
        pi_r = 1 / ((1 - pi) * pi)
        # lower_bounds = [0, -1000, -1]  # Lower bounds for [a, b, c]
        # upper_bounds = [1000, 1000, 1]  # Upper bounds for [a, b, c

        try:
            # Fit the model
            params, covariance = curve_fit(
                model,
                N,
                gamma,
                p0=(pi_r, 1.0, 0.0),
                maxfev=1000000,
                bounds=(
                    [pi_r, -100.0, -2.0],
                    [5 * pi_r, 100.0, 1.0],
                ),
            )
            A1, A2, omega = params

            # Compute R²
            gamma_pred = model(N, A1, A2, omega)
            r2 = r2_score(gamma, gamma_pred)
            mse = np.mean((np.array(gamma) - np.array(gamma_pred)) ** 2)

            # Compute h
            if compute_h:
                h = A1 * (1 - pi) * pi
                results.append(
                    {
                        "Key": key,
                        "A1": A1,
                        "h": h,
                        "A2": A2,
                        "omega": omega,
                        "R2": r2,
                        "MSE": mse,
                    }
                )
            else:
                # Store results
                results.append(
                    {
                        "Key": key,
                        "A1": A1,
                        "A2": A2,
                        "omega": omega,
                        "R2": r2,
                        "MSE": mse,
                    }
                )

        except Exception:
            # Handle cases where fitting fails
            results.append(
                {"Key": key, "A1": np.nan, "A2": np.nan, "omega": np.nan, "R2": np.nan}
            )
            elections_not_computed.append(key)

    # Return results as a DataFrame
    return pd.DataFrame(results), elections_not_computed
