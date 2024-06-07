import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ydata_profiling import ProfileReport
from src.DF_penumbra.data_preprocessor import DataPreprocessor
from src.DF_penumbra.data_loader import Neo4jDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run different functions based on input parameters."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="task name:{missing-features, }",
        metavar="",
    )

    args = parser.parse_args()

    data_dir = "src/data"

    if args.task != "handle-features":
        dp = DataPreprocessor(data_dir, include_npi=True, training=True)
        # -- Training data --
        combined_df, df_46, dropped_df = dp.get_training_data(
            skewed_factor=2, size=None
        )
        # Merge ltable and rtable into one
        ltable_cols = [col for col in combined_df.columns if "ltable_" in col]
        rtable_cols = [col for col in combined_df.columns if "rtable_" in col]
        A = combined_df[ltable_cols]
        A.columns = [col.replace("ltable_", "") for col in ltable_cols]
        B = combined_df[rtable_cols]
        B.columns = [col.replace("rtable_", "") for col in rtable_cols]
        training_df = pd.concat([A, B], sort=False)
        # removes duplicate rows based on all columns.
        training_df = training_df.drop_duplicates()
        drop_org_columns = ["og_state", "og_country", "og_specialty"]
        training_df.drop(columns=drop_org_columns, inplace=True, errors="ignore")

        print("Training data:")
        print(training_df)
        # -- Entire data --
        entire_df = dp.get_entire_data()
        print("Entire data:")
        print(entire_df)
        entire_df.drop(columns=drop_org_columns, inplace=True)

    if args.task == "handle-features":
        dl = Neo4jDataLoader(data_dir)
        dl.data_optimization()

    elif args.task == "missing-features":
        # non missing percent
        t_non_missing_per = training_df.notna().mean() * 100
        print("training non_missing_per")
        print(t_non_missing_per.to_string())

        # entire_df.to_csv("all.csv", index=False)
        entire_df = entire_df[training_df.columns]

        # non missing percent
        all_non_missing_per = entire_df.notna().mean() * 100
        print("All data non_missing_per")
        print(all_non_missing_per.to_string())

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(
            list(t_non_missing_per.keys()),
            list(t_non_missing_per.values),
            marker="^",
            label="Training data",
        )
        plt.scatter(
            list(all_non_missing_per.keys()),
            list(all_non_missing_per.values),
            marker="o",
            label="Entire data",
        )

        plt.ylabel("% of Present Values")
        plt.xlabel("Columns")
        plt.title("Percentage of Present Values per Column")
        plt.legend()
        plt.show()

    elif args.task == "missing-alldata":
        def set_address(row):
            return row["addr1"] if row["addr1"] else row["addr2"]
        entire_df["address"] = entire_df.apply(set_address, axis=1)
        entire_df.drop(columns=["state2", "lic_state", "addr1", "addr2"], inplace=True)
        all_non_missing_per = entire_df.notna().mean().round(4) * 100
        print("All data non_missing_per")
        print(all_non_missing_per.to_string())

        # Plotting
        fig, ax = plt.subplots()
        bars = ax.bar(
            list(all_non_missing_per.keys()),
            list(all_non_missing_per.values),
            label="Entire data",
        )

        plt.ylabel("% of Present Values")
        plt.xlabel("Columns")
        plt.title("Percentage of present values")
        plt.xticks(rotation=45)
        plt.legend()
        for container in ax.containers:
            ax.bar_label(container)
        plt.show()

    elif args.task == "categorical":
        categoricals = ["category", "country", "specialty"]
        for c in categoricals:
            entire_counts = (
                entire_df[c]
                .value_counts(
                    dropna=False,
                    normalize=True,
                )
                .round(4)
                * 100
            )
            training_counts = None
            if c in training_df.columns:
                training_counts = (
                    training_df[c].value_counts(dropna=False, normalize=True).round(4)
                    * 100
                )
            # Combine the counts into a single DataFrame
            if training_counts is not None:
                category_comparison = pd.DataFrame(
                    {
                        "Entire Dataset": entire_counts,
                        "Training Dataset": training_counts,
                    }
                ).fillna(0)
            else:
                category_comparison = entire_counts
            category_comparison.to_csv("categorical_{}.csv".format(c))
            # Plotting
            ax = category_comparison.plot(
                kind="barh", figsize=(10, 6), color=["red", "blue"]
            )
            plt.title(
                "Comparison of Data Distribution for {}: Entire Data vs. Training Data".format(
                    c.capitalize()
                )
            )
            plt.ylabel("Unique Value")
            plt.xlabel("Percent")
            plt.xticks(rotation=0)
            plt.legend()
            plt.grid(True)
            for container in ax.containers:
                ax.bar_label(container)
            plt.show()

    elif args.task == "profiling":
        entire_profile = ProfileReport(
            entire_df,
            title="Pandas Profiling Report for Entire dataset",
            correlations={"auto": {"calculate": False}},
        )
        entire_profile.to_file("profiling_entire.html")

        training_profile = ProfileReport(
            training_df,
            title="Pandas Profiling Report for Training dataset",
            correlations={"auto": {"calculate": False}},
        )
        training_profile.to_file("profiling_training.html")

        # entire_profile = ProfileReport(
        #     entire_df,
        #     title="Pandas Profiling Report for Entire dataset",
        #     correlations={"auto": {"calculate": False}},
        # )
        # comparison_report = entire_profile.compare(training_profile)
        # comparison_report.to_file("profiling_comparison.html")
