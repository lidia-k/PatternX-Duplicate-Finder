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


def process_fill_flname_na(df):
    def fill_names(row):
        def get_first_last_name(fullname):
            parts = fullname.split()
            first_name = parts[0]
            if "." in parts[0]:  # like: 'D. Christopher Metzger'
                first_name = parts[1]
                last_name = " ".join(parts[2:])
            else:
                last_name = " ".join(parts[1:])

            return first_name, last_name

        first_name, last_name = get_first_last_name(row["fullname"])
        row["fname"] = first_name if pd.isna(row["fname"]) else row["fname"]
        row["lname"] = last_name if pd.isna(row["lname"]) else row["lname"]
        return row

    df[["fname", "lname"]] = df.apply(fill_names, axis=1)[["fname", "lname"]]

    return df


def process_first_name_synonyms(df):
    name_synonyms = get_synonyms()
    df["fname"] = df["fname"].apply(
        lambda x: name_synonyms[x] if x in name_synonyms else x
    )

    return df
