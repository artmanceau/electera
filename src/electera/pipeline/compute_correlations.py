import numpy as np
import polars as pl
import polars_distance as pld
from tqdm import tqdm

from electera.pipeline.data_processing_pl import ElectionDataProcessor

# TODO:
# - vote gagnant tour 2 (modify schema)
# - participation tour 2 (modify schema)
# - vote oui/non referundum (modify schema)
# - vote blancs/nuls tour 1 et 2 (tour 1 easy, tour 2 (modify schema))
# - vote droite / gauche tour 1 (easy)
# Générer les plots

S3_BASE = "s3://arthurmanceau/election_modeling_uhcp/data/derived/spatial_correlations/distance_bins"


def get_pairs_for_bin(bin_id: int):
    pairs = pl.scan_parquet(
        f"{S3_BASE}/distance_bin={bin_id}/*.parquet",
        hive_partitioning=True,
    ).collect()
    bounds = bin_edges.filter(pl.col("distance_bin") == bin_id).row(0, named=True)
    return {
        "bounds": (bounds["distance_lower"], bounds["distance_upper"]),
        "pairs": pairs.filter(
            pl.col("codecommune") != pl.col("codecommune_dest")
        ).select("codecommune", "codecommune_dest", "distance_km"),
    }


if __name__ == "__main__":
    processor = ElectionDataProcessor()

    electoral_data, election_catalogs = processor.load_electoral_data()
    election_catalog, election_code_mapping = election_catalogs

    commune_data = processor.load_communes_data()

    var = "pvotepar"

    if var == "pvoteTD":
        electoral_data = electoral_data.with_columns(pl.col(var).fill_null(0.5))

    X = (
        electoral_data.filter(pl.col("inscrits") > 0)
        .with_columns(
            p_alpha=pl.when(pl.col(var) == 1.0)
            .then((pl.col("inscrits") - 0.5) / pl.col("inscrits"))
            .when(pl.col(var) == 0.0)
            .then(0.5 / pl.col("inscrits"))
            .otherwise(pl.col(var))
        )
        .with_columns(tau=(pl.col("p_alpha") / (1 - pl.col("p_alpha"))).log())
        .join(
            commune_data.select("codecommune", "lat", "long"),
            on="codecommune",
            validate="m:1",
            how="left",
        )
        .with_columns(
            pl.col("lat").fill_null(pl.col("lat").mean().over("dep")),
            pl.col("long").fill_null(pl.col("long").mean().over("dep")),
        )
        .select(
            "codecommune",
            "p_alpha",
            var,
            "tau",
            "election_code",
            "inscrits",
            "lat",
            "long",
        )
    )
    assert (
        X.select(
            pl.col("tau").null_count().alias("null_count"),
            pl.col("tau").is_nan().sum().alias("nan_count"),
            pl.col("tau").is_infinite().sum().alias("inf_count"),
        )
        .sum_horizontal()
        .item()
        == 0
    )

    n_communes = 1000

    boundaries = {}
    results = []

    for election_code, subset in X.partition_by("election_code", as_dict=True).items():
        # Step 1: Compute bin boundaries
        n_bins = max(2, len(subset) // n_communes)
        breaks = (
            subset.select(
                pl.col("inscrits")
                .qcut(n_bins, include_breaks=True, allow_duplicates=True)
                .alias("qcut")
            )
            .unnest("qcut")
            .select("breakpoint")
            .unique()
            .sort("breakpoint")
            .filter(pl.col("breakpoint") != float("inf"))
            .with_columns(pl.col("breakpoint").round(0))
            .with_row_index("bin_index")
        )
        boundaries[election_code] = breaks.rows()  # [(0, edge0), (1, edge1), ...]

        # Step 2: Build edges lookup — lower/upper per bin index
        break_values = [b[1] for b in boundaries[election_code]]
        break_labels = [str(b[0]) for b in boundaries[election_code]] + [
            str(len(boundaries[election_code]))
        ]

        bin_edges = pl.DataFrame(
            {
                "N_bin_index": list(range(len(break_values) + 1)),
                "N_bin_lower": [float("-inf")] + break_values,  # lower edge per bin
                "N_bin_upper": break_values + [float("inf")],  # upper edge per bin
            }
        ).with_columns(pl.col("N_bin_index").cast(pl.UInt32))

        # Step 3: Apply cut and join edges
        results.append(
            subset.with_columns(
                pl.col("inscrits")
                .cut(break_values, labels=break_labels)
                .cast(pl.UInt32)
                .alias("N_bin_index")
            ).join(bin_edges, on="N_bin_index", how="left")
        )

    X_binned = (
        pl.concat(results)
        .with_columns(m=pl.col("tau").mean().over("election_code", "N_bin_index"))
        .with_columns(tau_minus_m=pl.col("tau") - pl.col("m"))
    )
    # Compute the mean tau by bin for each election

    # Distances
    df = (
        X.select("codecommune", "lat", "long")
        .unique("codecommune")
        .with_columns(
            pl.struct(latitude=pl.col("lat"), longitude=pl.col("long")).alias("coords")
        )
        .select("codecommune", "coords")
        .lazy()
    )

    # Create distance bins (uniform)
    distance_breaks = [round(x, 2) for x in np.linspace(0, 1000, 301).tolist()[1:-1]]
    break_labels = [str(i) for i in range(len(distance_breaks) + 1)]

    bin_edges = pl.DataFrame(
        {
            "distance_bin": list(range(len(distance_breaks) + 1)),
            "distance_lower": [0.0] + distance_breaks,
            "distance_upper": distance_breaks + [float("inf")],
        }
    ).with_columns(
        pl.col("distance_bin").cast(pl.UInt32),
        ((pl.col("distance_lower") + pl.col("distance_upper")) / 2).alias("mid"),
    )

    # Compute pairs, assign bin, sink to hive-partitioned parquet
    compute_distance = False
    if compute_distance:
        (
            df.join(df, how="cross", suffix="_dest")
            .filter(pl.col("codecommune") < pl.col("codecommune_dest"))
            .with_columns(
                pld.col("coords")
                .dist.haversine("coords_dest", "km")
                .alias("distance_km")
            )
            .with_columns(
                pl.col("distance_km")
                .cut(distance_breaks, labels=break_labels)
                .cast(pl.UInt32)
                .alias("distance_bin")
            )
            .select("distance_bin", "codecommune", "codecommune_dest", "distance_km")
            .sink_parquet(
                pl.PartitionBy(f"{S3_BASE}/", key="distance_bin"),
            )
        )

    results = []
    n_bins = len(distance_breaks) + 1

    election_codes = (
        X_binned.select("election_code").unique().get_column("election_code").to_list()
    )
    for election_code in tqdm(election_codes):
        Z = X_binned.filter(pl.col("election_code") == election_code)
        denum = Z.with_columns(x=pl.col("tau_minus_m") ** 2).select("x").mean().item()
        for r in tqdm(range(n_bins)):
            pairs = get_pairs_for_bin(r)
            C_tau_r = (
                pairs["pairs"]
                .join(
                    Z.select("codecommune", "tau_minus_m"),
                    on="codecommune",
                    how="inner",
                )
                .rename({"tau_minus_m": "tau_minus_m_alpha"})
                .join(
                    Z.select("codecommune", "tau_minus_m").rename(
                        {
                            "codecommune": "codecommune_dest",
                            "tau_minus_m": "tau_minus_m_beta",
                        }
                    ),
                    on="codecommune_dest",
                    how="inner",
                )
                .with_columns(
                    num=pl.col("tau_minus_m_alpha") * pl.col("tau_minus_m_beta"),
                )
                .mean()
                .with_columns(C_tau=pl.col("num") / denum)
                .select("C_tau")
                .item()
            )
            results.append({"election_code": election_code, "bin": r, "C_tau": C_tau_r})

    df_c_tau = pl.DataFrame(results)
    df_c_tau = df_c_tau.join(
        bin_edges, left_on="bin", right_on="distance_bin", validate="m:1"
    )

    df_c_tau.write_parquet(
        f"s3://arthurmanceau/election_modeling_uhcp/data/derived/spatial_correlations/c_tau_{var}_r.parquet"
    )
