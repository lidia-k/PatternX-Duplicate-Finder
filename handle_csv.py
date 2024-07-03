# check predictions_RandomForestClassifier_all.csv file

import argparse
import pandas as pd

from src.penumbra.edge_builder import EdgeBuilder
from src.penumbra.data_loader import Neo4jDataLoader
from src.penumbra.utils import process_first_name_synonyms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run different functions based on input parameters."
    )
    parser.add_argument("--task", type=str, default=None, help="task name", metavar="")
    parser.add_argument("--file_path", type=str, default=None, help="task name", metavar="")
    args = parser.parse_args()

    if args.task == "match-firstlast-name":
        df = pd.read_csv("predictions_RandomForestClassifier_all.csv")
        # replace fname with synonyms
        process_first_name_synonyms(df)

    elif args.task == "create-rf-edges":
        df = pd.read_csv("predictions_RandomForestClassifier_all.csv")
        df = df[df["predicted"] == 1]
        print(df)

        # false positive from rf-predict-all
        f_pos = [
            ("po_co_342", "po_co_928"),
            ("po_no_171", "po_no_486"),
            ("po_co_319", "po_no_490"),
            ("po_no_400", "po_no_401"),
            ("po_no_481", "po_no_485"),
            ("po_no_677", "po_no_681"),
            ("po_no_331", "po_no_413"),
            ("po_no_413", "po_no_197"),
            ("po_no_114", "po_no_133"),
            ("po_no_114", "po_no_129"),
            ("po_no_560", "po_co_442")
        ]

        def ignore(row):
            is_ignore = False
            for pair in [
                (row["ltable_uid"], row["rtable_uid"]),
                (row["rtable_uid"], row["ltable_uid"]),
            ]:
                if pair in f_pos:
                    is_ignore = True
            return is_ignore

        df["ignore"] = df.apply(ignore, axis=1)
        print("Ignore: {}".format(len(df[df["ignore"] == 1])))
        df = df[df["ignore"] == 0]
        print(df)
        eb = EdgeBuilder()
        eb.set_relationship(df, "r2_rf")

    elif args.task == "export-matched":
        data_dir = "src/data"
        dl = Neo4jDataLoader(data_dir)
        dl.export_results(filename="master.csv")
    
    elif args.task == "split" and args.file_path is not None:
        df = pd.read_csv(f'{args.file_path}.csv')

        all_columns = df.columns

        # Separate left and right table columns
        left_columns = [col for col in all_columns if col.startswith('ltable_')]
        right_columns = [col for col in all_columns if col.startswith('rtable_')]

        # Create new dataframes for left and right tables
        df_left = df[left_columns].copy()
        df_right = df[right_columns].copy()

        # Rename columns to remove prefixes
        df_left.columns = [col.replace('ltable_', '') for col in df_left.columns]
        df_right.columns = [col.replace('rtable_', '') for col in df_right.columns]

        # Create a new DataFrame to store the result
        df_result = pd.DataFrame()
        for i in range(len(df)):
            df_result = pd.concat([df_result, df_left.iloc[[i]], df_right.iloc[[i]]], ignore_index=True)

        # Save the result to a new CSV file
        df_result.to_csv(f'{args.file_path}_split.csv', index=False)
        print("File has been split and saved")

