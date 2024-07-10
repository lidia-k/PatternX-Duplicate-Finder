import argparse
import pandas as pd
from sklearn.utils import shuffle
from src.penumbra.training_magellan import MagellanTrainer
from src.penumbra.utils import get_synonyms
from src.penumbra import constants
from src.penumbra.data_preprocessor import DataPreprocessor


def df_columns(df):
    cols = [
        "uid",
        "fname",
        "lname",
        "fullname",
        "npi",
        "country",
        "email",
        "sap_no",
        "qb_id",
        "currency",
        "category",
        "specialty",
        "org",
    ]

    df = df[
        [c for c in ["type", "label"]]
        + ["ltable_" + c for c in cols]
        + ["rtable_" + c for c in cols]
    ]
    return df


def get_matching_pairs():
    edge_types = ["r1_" + et for et in constants.EDGE_TYPES]
    edge_types += ["r2_rf"]
    q = f"""
    UNWIND {edge_types} AS type
    MATCH (a)-[r]->(b)
    WHERE type(r) = type AND id(a) < id(b) AND (a:Provider OR a:Speaker) AND (b:Provider OR b:Speaker)
    RETURN DISTINCT a, b
    """

    result = dp.graph.cypher_transaction(q)
    df = dp._create_df(result, label=1)
    df["id"] = df.index
    # print("All Matching pairs")
    # print(df)
    return df


def matching_pairs_diff():
    """
    Matching entities: duplicates with different sap_no, qb_id or email
    - Get matching pairs with edges r1 and r2
    - find duplicates with different sap_no, qb_id or email. Then put them together
    """
    df = get_matching_pairs()
    diff_cols = ["sap_no", "qb_id", "email"]
    diff_df = []
    for col in diff_cols:
        diff = df[
            df[f"ltable_{col}"].notna()
            & df[f"rtable_{col}"].notna()
            & (df[f"ltable_{col}"] != df[f"rtable_{col}"])
        ]
        print(col)
        print(diff)
        diff["type"] = "matching_diff_" + col
        diff_df.append(diff)

    matching_pairs = pd.concat(diff_df, sort=False)
    matching_pairs = matching_pairs.drop_duplicates(subset=["id"])
    print("matching_pairs_diff")
    print(matching_pairs)
    matching_pairs = df_columns(matching_pairs)

    # matching_pairs.to_csv("matching_pairs_diff.csv", index=False)
    return matching_pairs


def matching_pairs_synonyms():
    """
    Matching entities: create matching pairs with first name synonyms
    - Get matching pairs with edges r1 and r2
    - Find duplicates with same first name and create matching pairs with synonyms
    """
    df = get_matching_pairs()
    same_fname_df = df[df["ltable_fname"] == df["rtable_fname"]]
    synonyms = get_synonyms()
    synonyms_df = same_fname_df[same_fname_df["ltable_fname"].isin(synonyms.keys())]
    synonyms_df["rtable_fname"] = synonyms_df["rtable_fname"].apply(
        lambda x: [s.strip() for s in synonyms[x.strip()].split(",")]
    )
    synonyms_df = synonyms_df.explode("rtable_fname")
    synonyms_df["type"] = "matching_synonyms"
    synonyms_df = df_columns(synonyms_df)
    print("matching_pairs_synonyms")
    print(synonyms_df)
    # synonyms_df = synonyms_df.sample(n=800, random_state=1)
    # synonyms_df.to_csv("matching_pairs_synonyms.csv", index=False)
    return synonyms_df


def non_matching_pairs_same_col():
    """
    Non-matching entities: pairs with same sap_no, qb_id or email
    -
    """

    diff_cols = ["sap_no", "qb_id", "email"]
    same_df = []

    for col in diff_cols:
        q = f"""
        MATCH (a), (b)
        WHERE id(a) < id(b)
        AND (a:Provider OR a:Speaker) AND (b:Provider OR b:Speaker)
        AND a.{col} = b.{col}
        AND NOT (a)-[]-(b)
        RETURN a, b
        """
        print(q)
        result = dp.graph.cypher_transaction(q)
        if len(result) > 0:
            same = dp._create_df(result, label=0)
            same["type"] = "non_matching_same_" + col
            same_df.append(same)

    non_matching_pairs = pd.concat(same_df, sort=False)
    non_matching_pairs = df_columns(non_matching_pairs)
    print("non_matching_pairs_same_col")
    print(non_matching_pairs)
    # non_matching_pairs.to_csv("non_matching_pairs.csv", index=False)
    return non_matching_pairs


def non_matching_synonyms(infile):
    """
    Non-matching entities: similar names
    - Get the pairs of nodes in the attached file.
    - Extensive processing of these pairs with the addition of synonyms
    """
    df = pd.read_csv(infile)
    grouped = df.groupby("id")

    # get data from Neo4J
    uids = list(set(df["uid"].to_list()))
    q = f"""
    MATCH (n)
    WHERE (n:Provider OR n:Speaker) AND n.uid in {uids}
    RETURN n
    """
    result = dp.graph.cypher_transaction(q)

    df = pd.DataFrame([dict(record[0]) for record in result])
    df = df.set_index("uid", drop=False)

    paired_data = []
    for _, df_group in grouped:
        index = 0
        for i, row in df_group.iterrows():
            if index == 0:
                left = df.loc[row["uid"]]
            else:
                right = df.loc[row["uid"]]
            index += 1

        left_dict = {"ltable_" + col: val for col, val in left.items()}
        right_dict = {"rtable_" + col: val for col, val in right.items()}
        paired_data.append({**left_dict, **right_dict})
    paired_df = pd.DataFrame(paired_data)

    synonyms = get_synonyms()
    paired_df["ltable_fname"] = paired_df["ltable_fname"].apply(
        lambda x: (
            [s.strip() for s in synonyms[x.strip()].split(",")]
            if x.strip() in synonyms
            else x
        )
    )
    paired_df["rtable_fname"] = paired_df["rtable_fname"].apply(
        lambda x: (
            [s.strip() for s in synonyms[x.strip()].split(",")]
            if x.strip() in synonyms
            else x
        )
    )
    paired_df = paired_df.explode("ltable_fname")
    paired_df = paired_df.explode("rtable_fname")
    if not ("ltable_sap_no" in paired_df.columns.to_list()):
        paired_df["ltable_sap_no"] = pd.NA
    if not ("rtable_sap_no" in paired_df.columns.to_list()):
        paired_df["rtable_sap_no"] = pd.NA

    paired_df["type"] = "non_matching_synonyms"
    paired_df["label"] = 0
    paired_df = df_columns(paired_df)
    print("non_matching_synonyms")
    print(paired_df)
    # paired_df.to_csv("non_matching_synonyms.csv", index=False)
    return paired_df


def non_matching_diff_name(limit=20000):
    q = f"""
    MATCH (a), (b) 
    WHERE id(a) < id(b)
    AND (a:Provider OR a:Speaker) AND (b:Provider OR b:Speaker)
    AND NOT (a)-[]-(b) AND a.lname <> b.lname AND a.fname <> b.fname
    RETURN a, b
    LIMIT {limit}
    """
    result = dp.graph.cypher_transaction(q)
    df = dp._create_df(result, label=0)
    df["type"] = "non_matching_diff_name"
    df = df_columns(df)
    print(len(df))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run different functions based on input parameters."
    )
    parser.add_argument("--task", type=str, default=None, help="task name", metavar="")
    args = parser.parse_args()

    master_file = "master_edge_full_1.csv"
    # master_file = "master_edge_balance.csv"

    if args.task == "master-edge":
        data_dir = "src/data"
        dp = DataPreprocessor(data_dir, include_npi=True)

        df = pd.concat(
            [
                matching_pairs_diff(),
                matching_pairs_synonyms(),
                non_matching_pairs_same_col(),
                non_matching_synonyms("edge_cases_split.csv"),
                non_matching_diff_name(20000)
            ],
            sort=False,
        )
        df.to_csv(master_file, index=False)
    elif args.task == "training":
        data_dir = "src/data"
        model = "rf"
        dp = DataPreprocessor(data_dir, include_npi=False, training=True)
        df = pd.read_csv(master_file)
        print("Matching pairs: {}".format(len(df[df["label"] == 1])))
        print("Non-matching pairs: {}".format(len(df[df["label"] == 0])))
        df.drop(
            columns=[
                "type",
                "ltable_uid",
                "rtable_uid",
                "ltable_npi",
                "rtable_npi",
                "ltable_fullname",
                "rtable_fullname",
                "ltable_lname",
                "rtable_lname",
            ],
            inplace=True,
        )
        df = shuffle(df, random_state=1).reset_index(drop=True)
        ltable, rtable, data = dp._load_data(df)
        # ltable, rtable, data = dp.prepare_training_data(skewed_factor=2, size=args.size, model=model)

        mt = MagellanTrainer(ltable, rtable, data, model=model, training=True)
        mt.train_model()
        print("Evaluating the model...")
        preds = mt.predict()
        mt.evaluate(preds)

        print("Displaying feature importance...")
        mt.retrieve_feature_importance()
