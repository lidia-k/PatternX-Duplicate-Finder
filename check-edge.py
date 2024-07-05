import argparse
import pandas as pd
from src.penumbra.utils import get_synonyms
from src.penumbra import constants
from src.penumbra.data_preprocessor import DataPreprocessor


def get_matching_pairs():
    edge_types = ["r1_" + et for et in constants.EDGE_TYPES]
    edge_types += ["r2_rf"]
    q = f"""
    UNWIND {edge_types} AS type
    MATCH (a)-[r]->(b)
    WHERE type(r) = type AND id(a) < id(b)  AND (a:Provider OR a:Speaker) AND (b:Provider OR b:Speaker)
    RETURN DISTINCT a, b
    """

    result = dp.graph.cypher_transaction(q)
    df = dp._create_df(result, label=1)
    df["id"] = df.index
    print("All Matching pairs")
    print(df)
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
        diff_df.append(diff)

    matching_pairs = pd.concat(diff_df, sort=False)
    matching_pairs = matching_pairs.drop_duplicates(subset=["id"])
    print(matching_pairs)
    matching_pairs = matching_pairs[
        ["ltable_" + c for c in cols] + ["rtable_" + c for c in cols]
    ]
    matching_pairs.to_csv("matching_pairs_diff.csv", index=False)


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
    synonyms_df = synonyms_df[
        ["ltable_" + c for c in cols] + ["rtable_" + c for c in cols]
    ]
    print("synonyms_df")
    print(synonyms_df)
    synonyms_df.to_csv("matching_pairs_synonyms.csv", index=False)


def non_matching_pairs_same_col():
    """
    Non-matching entities: pairs with same sap_no, qb_id or email
    -
    """
    df, _ = dp._build_non_matching_pairs(100000)
    df = dp._drop_high_missing_features(df, threshold=92)

    df["id"] = df.index
    print("All Non matching pairs")
    print(df)

    diff_cols = ["sap_no", "qb_id", "email"]
    same_df = []
    for col in diff_cols:
        same = df[
            df[f"ltable_{col}"].notna()
            & df[f"rtable_{col}"].notna()
            & (df[f"ltable_{col}"] == df[f"rtable_{col}"])
        ]
        print(col)
        print(same)
        same_df.append(same)

    non_matching_pairs = pd.concat(same_df, sort=False)
    non_matching_pairs = non_matching_pairs.drop_duplicates(subset=["id"])
    non_matching_pairs = non_matching_pairs[
        ["ltable_" + c for c in cols] + ["rtable_" + c for c in cols]
    ]
    print(non_matching_pairs)
    non_matching_pairs.to_csv("non_matching_pairs.csv", index=False)


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
    cols.remove("sap_no")
    paired_df = paired_df[
        ["ltable_" + c for c in cols] + ["rtable_" + c for c in cols]
    ]
    print(paired_df)
    paired_df.to_csv("non_matching_synonyms.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run different functions based on input parameters."
    )
    parser.add_argument("--task", type=str, default=None, help="task name", metavar="")
    args = parser.parse_args()

        
    data_dir = "src/data"
    dp = DataPreprocessor(data_dir, include_npi=True)    
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
        "org",
    ]
    if args.task == "matching_pairs_diff":
        matching_pairs_diff()
    elif args.task == "matching_pairs_synonyms":
        matching_pairs_synonyms()
    elif args.task == "non_matching_pairs_same_col":
        non_matching_pairs_same_col()
    elif args.task == "non_matching_synonyms":
        non_matching_synonyms("edge_cases_split.csv")
