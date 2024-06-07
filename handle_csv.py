# check predictions_RandomForestClassifier_all.csv file

import argparse
import pandas as pd

from src.DF_penumbra.edge_builder import EdgeBuilder
from src.DF_penumbra.data_loader import Neo4jDataLoader
from src.DF_penumbra.utils import process_first_name_synonyms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run different functions based on input parameters."
    )
    parser.add_argument("--task", type=str, default=None, help="task name", metavar="")
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
