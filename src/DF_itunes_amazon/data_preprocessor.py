import os
import pandas as pd
import py_entitymatching as em
from sklearn.utils import shuffle


class DataPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def prepare_training_data(self):
        # path_A = os.path.join(self.data_dir, "tableA.csv")
        # path_B = os.path.join(self.data_dir, "tableB.csv")

        # # Load the two tables.
        # A = em.read_csv_metadata(path_A, key="id")
        # B = em.read_csv_metadata(path_B, key="id")

        # # Basic information about the tables.
        # print("Number of tuples in A: " + str(len(A)))
        # print("Number of tuples in B: " + str(len(B)))
        # print(
        #     "Number of tuples in A X B (i.e the cartesian product): "
        #     + str(len(A) * len(B))
        # )

        # # Create an overlap blocker in Magellan and apply it to A and B to get the candidate set K1 which is in the format of
        # # a dataframe. The "l_out_attrs" and "r_out_attrs" parameters indicate the columns that will be included in K1 from A
        # # and B respectively.
        # ob = em.OverlapBlocker()
        # columns = [
        #     "Song_Name",
        #     "Artist_Name",
        #     "Album_Name",
        #     "Genre",
        #     "Price",
        #     "CopyRight",
        #     "Time",
        #     "Released",
        # ]
        # K1 = ob.block_tables(
        #     A,
        #     B,
        #     "Album_Name",
        #     "Album_Name",
        #     l_output_attrs=columns,
        #     r_output_attrs=columns,
        #     overlap_size=2,
        # )
        # # The number of tuple pairs in K1.
        # print("Number of tuples pairs in K1: " + str(len(K1)))
        # # Create a new overlap blocker to remove pairs from K1 that have no common word in "Artist_Name".
        # K2 = ob.block_candset(K1, "Artist_Name", "Artist_Name", overlap_size=1)
        # # The number of tuple pairs in K2.
        # print("Number of tuples pairs in K2: " + str(len(K2)))
        # # Apply the third overlap blocker.
        # K3 = ob.block_candset(K2, "Song_Name", "Song_Name", overlap_size=1)
        # # The number of tuple pairs in K3.
        # print("Number of tuples pairs in K3: " + str(len(K3)))
        # path_K = os.path.join(self.data_dir, "candidate.csv")
        # K3.to_csv(path_K, index=False)

        # The path to the labeled data file.
        path_G = os.path.join(self.data_dir, "e2e-tutorial", "gold.csv")
        data = pd.read_csv(path_G)
        data.rename(columns={"_id": "id"}, inplace=True)
        # path_cp = os.path.join(self.data_dir, "gold_id.csv")
        # cp_gold.to_csv(path_cp, index=False)
        # # Load the labeled data into a dataframe.
        # C = em.read_csv_metadata(
        #     path_cp,
        #     key="id",
        #     ltable=A,
        #     rtable=B,
        #     fk_ltable="ltable_id",
        #     fk_rtable="rtable_id",
        # )
        # print(G.head())
        # G.rename(columns={'_id': 'id'}, inplace=True)
        # print(G.head())
        print("Number of labeled pairs:", len(data))
        return self._load_data(data)

    def _split_tables(self, df):
        """
        Split the data into A, B, and C tables.
        """
        df = shuffle(df, random_state=1).reset_index(drop=True)
        df["id"] = df.index
        df["ltable_id"] = df["id"]
        df["rtable_id"] = df["id"]
        df.to_csv("C.csv", index=False)

        ltable_cols = [col for col in df.columns if "ltable_" in col]
        rtable_cols = [col for col in df.columns if "rtable_" in col]

        A = df[ltable_cols]
        A.columns = [col.replace("ltable_", "") for col in ltable_cols]
        A = A.rename(columns={"id": "ltable_id"})
        A.to_csv("A.csv", index=False)

        B = df[rtable_cols]
        B.columns = [col.replace("rtable_", "") for col in rtable_cols]
        B = B.rename(columns={"id": "rtable_id"})
        B.to_csv("B.csv", index=False)

    def _load_data(self, df):
        self._split_tables(df)

        A = em.read_csv_metadata("A.csv", key="ltable_id")

        B = em.read_csv_metadata("B.csv", key="rtable_id")

        C = em.read_csv_metadata(
            "C.csv",
            key="id",
            ltable=A,
            rtable=B,
            fk_ltable="ltable_id",
            fk_rtable="rtable_id",
        )

        return A, B, C

    def prepare_all_data(self):
        path_A = os.path.join(self.data_dir, "tableA.csv")
        path_B = os.path.join(self.data_dir, "tableB.csv")

        # Load the two tables.
        A = em.read_csv_metadata(path_A, key="id")
        B = em.read_csv_metadata(path_B, key="id")

        # Basic information about the tables.
        print("Number of tuples in A: " + str(len(A)))
        print("Number of tuples in B: " + str(len(B)))
        print(
            "Number of tuples in A X B (i.e the cartesian product): "
            + str(len(A) * len(B))
        )

        # Create an overlap blocker in Magellan and apply it to A and B to get the candidate set K1 which is in the format of
        # a dataframe. The "l_out_attrs" and "r_out_attrs" parameters indicate the columns that will be included in K1 from A
        # and B respectively.
        ob = em.OverlapBlocker()
        columns = [
            "Song_Name",
            "Artist_Name",
            "Album_Name",
            "Genre",
            "Price",
            "CopyRight",
            "Time",
            "Released",
        ]
        K1 = ob.block_tables(
            A,
            B,
            "Album_Name",
            "Album_Name",
            l_output_attrs=columns,
            r_output_attrs=columns,
            overlap_size=2,
        )
        # The number of tuple pairs in K1.
        print("Number of tuples pairs in K1: " + str(len(K1)))
        # Create a new overlap blocker to remove pairs from K1 that have no common word in "Artist_Name".
        K2 = ob.block_candset(K1, "Artist_Name", "Artist_Name", overlap_size=1)
        # The number of tuple pairs in K2.
        print("Number of tuples pairs in K2: " + str(len(K2)))
        # Apply the third overlap blocker.
        K3 = ob.block_candset(K2, "Song_Name", "Song_Name", overlap_size=1)
        # The number of tuple pairs in K3.
        print("Number of tuples pairs in K3: " + str(len(K3)))
        K3.rename(columns={"_id": "id"}, inplace=True)
        return K3

    def prepare_unlabeled_data(self):
        path = os.path.join(self.data_dir, "unlabeled.csv")
        data = pd.read_csv(path)
        data.columns = [col.replace("left_", "ltable_") for col in data.columns]
        data.columns = [col.replace("right_", "rtable_") for col in data.columns]
        return data
