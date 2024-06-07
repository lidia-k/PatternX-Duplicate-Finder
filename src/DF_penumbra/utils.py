import json
import os
import pandas as pd


def process_columns(prop):
    prop = prop.replace("a_", "")
    prop = prop.replace("b_", "")
    prop = prop.replace("t_", "") if prop.startswith("t_") else prop
    return prop


def process_int_cols(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
            )
    return df


def process_bi_emails(df):
    df["email"] = df["email"].apply(lambda x: x.split("; ") if pd.notna(x) else x)
    df = df.explode("email")
    return df


def process_biSAP_number(df):
    df["sap_no"] = df["sap_no"].apply(
        lambda x: str(x).split("\n") if pd.notna(x) else x
    )
    df = df.explode("sap_no")

    if "sap_name" in df.columns:
        df["sap_name"] = df["sap_name"].apply(
            lambda x: x.split("\n") if pd.notna(x) else x
        )
        df = df.explode("sap_name")
    return df


def get_synonyms():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    name_synonyms = {}

    with open(dir_path + "/data/synonyms.txt", "r", encoding="utf-8") as file:
        for line in file:
            columns = line.strip().split("\t")
            if len(columns) == 5:
                name = columns[0]
                synonyms = columns[4]
                name_synonyms[name] = synonyms

    return name_synonyms


def process_first_name_synonyms(df):

    def match_first_last_name(row):
        ltable_lname = row["ltable_lname"].lower()
        rtable_lname = row["rtable_lname"].lower()
        ltable_fname = row["ltable_fname"].lower()
        rtable_fname = row["rtable_fname"].lower()

        if ltable_lname == rtable_lname:
            # fname is same
            if ltable_fname == rtable_fname:
                return 1

            ltable_fname_result = [x.strip() for x in ltable_fname.split(",")]
            rtable_fname_result = [x.strip() for x in rtable_fname.split(",")]
            if ltable_fname_result:
                if rtable_fname in ltable_fname_result:
                    return 1
            if rtable_fname_result:
                if ltable_fname in rtable_fname_result:
                    return 1

        return 0

    name_synonyms = get_synonyms()
    df["ltable_fname"] = df["ltable_fname"].apply(
        lambda x: name_synonyms[x] if x in name_synonyms else x
    )
    df["rtable_fname"] = df["rtable_fname"].apply(
        lambda x: name_synonyms[x] if x in name_synonyms else x
    )

    df["match_f_l_name"] = df.apply(match_first_last_name, axis=1)
    df.to_csv("predictions_RandomForestClassifier_all_result.csv", index=False)


def get_us_states():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + "/data/us_states.json") as f_in:
        return json.load(f_in)
