import glob
import os
import platform
import subprocess

import numpy as np
import pandas as pd

from src.dao.NEO4J_Graph import Graph
from src.DF_penumbra import constants
from src.DF_penumbra.utils import (
    get_synonyms,
    get_us_states,
    process_bi_emails,
    process_biSAP_number,
    process_columns,
    process_int_cols,
)
from src.utils import auto_config as config


class Neo4jDataLoader:
    def __init__(self, data_dir=None):
        self.graph = Graph(
            config.NEO4J_URL,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD,
        )
        self.data_dir = data_dir

    def _process_names(self, file_name):
        file_type = "sp" if "speaker" in file_name else "po"
        renamed_cols = constants.SP_COLS if file_type == "sp" else constants.PO_COLS

        name_str = file_name.split("-")[3 if file_type == "sp" else 2].split(".")[0]
        file_name = f"{self.data_dir}/{file_type}_{name_str}.csv"
        node_type = f"{file_type}_{name_str[:2]}"

        return renamed_cols, name_str, file_name, node_type

    def _prepare_csv_file(self, csv_file):
        if "speaker" not in csv_file and "hcp" not in csv_file:
            print(f"Invalid file: {csv_file}")
            return

        rename, name_str, file_name, node_type = self._process_names(csv_file)
        df = pd.read_csv(csv_file)

        # Add uid column based on the node type
        df["uid"] = [f"{node_type}_{i+2}" for i in range(len(df))]

        """
        # Drop unnecessary columns
        drop_cols = [
            'Prefix', 'Status', 'Contact Type', 'Reportable HCP', 
            'Request Type', 'HCC ID', 't_primary', 'Practice Type'
        ]
        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        """

        if "speaker" in csv_file and not "all" in csv_file:
            df = pd.read_csv(csv_file, header=1)
            df["uid"] = [f"{node_type}_{i+3}" for i in range(len(df))]
            df["franchise"] = [name_str.capitalize() for i in range(len(df))]

        if "vcheck" in csv_file:
            df.columns = [process_columns(col) for col in df.columns]
            rename = constants.PO_VC_COLS
            for col in ["first_name", "last_name"]:
                df[col] = df[col].str.capitalize()

        # Rename columns and convert to lowercase
        df = df.rename(columns=rename)
        df.columns = [col.lower() for col in df.columns]

        if "speaker" in csv_file:
            df[["fname", "lname"]] = df["fullname"].str.split(" ", n=1, expand=True)

        # if 'speaker' in csv_file:
        #    df[['fname', 'lname']] = df['fullname'].str.split(n=1, expand=True)

        # Split a row with two sap numbers into two rows
        if "sap_no" in df.columns:
            df = process_biSAP_number(df)

        # Split a row with two emails into two rows
        if "email" in df.columns:
            df = process_bi_emails(df)

        # Convert columns to int
        df = process_int_cols(df, constants.INT_COLS)

        # Replace null values with np.nan
        null_val = [0, "0", "N/A", "#N/A", "N/A ", "n/a (ask Carson Milner)", "unknown"]
        df.replace(null_val, np.nan, inplace=True)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]

        df.to_csv(f"./{file_name}", index=False)
        print(f"Updated file: {file_name}")
        return file_name, df

    def _load_data_from_cypher(self, file_path):
        if "sp" in file_path:
            cypher_file = (
                f"{self.data_dir}/sp_all.cypher"
                if "all" in file_path
                else f"{self.data_dir}/sp.cypher"
            )
        else:
            cypher_file = (
                f"{self.data_dir}/po_vcheck.cypher"
                if "vcheck" in file_path
                else f"{self.data_dir}/po.cypher"
            )

        with open(cypher_file, "r") as f:
            query = f.read()

        query = query.format(file_path=f"file:///{file_path}")
        self.graph.cypher_transaction(query)
        print(f"Loaded data from {file_path} to Neo4j")

    def _find_common_columns(self, df_dict):
        common_cols = set()
        for key, df in df_dict.items():
            if "vcheck" in key:
                continue
            (
                common_cols.intersection_update(df.columns)
                if common_cols
                else common_cols.update(df.columns)
            )
        return common_cols

    def load_csv_to_neo4j(self):
        self.graph.wipe_database()

        # On Linux, docker exec chown and chmod the data directory
        if platform.system() == "Linux":
            cmd = "docker exec neo4j /bin/bash -c 'chown -R 777:777 import/data && chmod -R 777 import/data'"
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        # Remove existing data files
        f_types = ["sp_*.csv", "po_*.csv"]
        for f_type in f_types:
            for f in glob.glob(f"{self.data_dir}/{f_type}"):
                os.remove(f)

        # Update csv files and load data to Neo4j
        data_bundles = glob.glob(f"{self.data_dir}/*.csv")
        df_dict = {}
        for f in data_bundles:
            file_name, df = self._prepare_csv_file(f)
            self._load_data_from_cypher(file_name)

    def create_synonym_nodes(self):
        synonym_dict = get_synonyms()
        fnames = list(synonym_dict.keys())

        q = f"MATCH (n) WHERE n.fname IN {fnames} RETURN n, id(n) AS node_id"
        result = self.graph.cypher_transaction(q)
        df = pd.DataFrame([dict(record[0], node_id=record[1]) for record in result])
        print(df)

        count = 0
        for i, row in df.iterrows():
            fname = row["fname"]
            synonyms = [s.strip() for s in synonym_dict[fname].split(",")]

            # Create synonym nodes and copy properties and relationships
            for synonym in synonyms:
                count += 1
                # Create the new node
                create_node_query = """
                MATCH (n)
                WHERE ID(n) = $original_id
                CREATE (copy:Synonym)
                SET copy = n
                SET copy.fname = $synonym
                SET copy.fullname = $fullname
                RETURN id(copy)"""
                result = self.graph.cypher_transaction(
                    create_node_query,
                    {
                        "original_id": row["node_id"],
                        "synonym": synonym,
                        "fullname": "{} {}".format(synonym, row["lname"]),
                    },
                )
                synonym_node_id = result[0][0]

                # Create the 'clone' relationship
                create_clone_rel_query = """
                MATCH (original), (synonym)
                WHERE id(original) = $original_node_id AND id(synonym) = $synonym_node_id
                CREATE (original)-[:r_clone]->(synonym)
                """
                self.graph.cypher_transaction(
                    create_clone_rel_query,
                    {
                        "original_node_id": row["node_id"],
                        "synonym_node_id": synonym_node_id,
                    },
                )

                # Copy relationships from original node to new synonym node
                self._copy_relationships(row["node_id"], synonym_node_id)

        print(f"Added node {count}")

    def _copy_relationships(self, original_node_id, synonym_node_id):
        copy_outgoing_rels_query = """
        MATCH (n)-[r]->(m)
        WHERE id(n) = $original_node_id AND type(r) <> 'r_clone'
        WITH type(r) AS relType, m
        MATCH (synonym)
        WHERE id(synonym) = $synonym_node_id
        CALL apoc.create.relationship(synonym, relType, {}, m) YIELD rel
        RETURN synonym, m
        """
        self.graph.cypher_transaction(
            copy_outgoing_rels_query,
            {"original_node_id": original_node_id, "synonym_node_id": synonym_node_id},
        )

        copy_incoming_rels_query = """
        MATCH (n)<-[r]-(m)
        WHERE id(n) = $original_node_id AND type(r) <> 'r_clone'
        WITH type(r) AS relType, m
        MATCH (synonym)
        WHERE id(synonym) = $synonym_node_id
        CALL apoc.create.relationship(m, relType, {}, synonym) YIELD rel
        RETURN synonym, m
        """
        self.graph.cypher_transaction(
            copy_incoming_rels_query,
            {"original_node_id": original_node_id, "synonym_node_id": synonym_node_id},
        )

    def create_synoname_nodes(self):
        add = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))

        with open(dir_path + "/data/synonyms.txt", "r", encoding="utf-8") as file:
            for line in file:
                columns = line.strip().split("\t")
                properties = {
                    "name": columns[0]
                    + (", {}".format(columns[4]) if len(columns) == 5 else ""),
                    "origin": columns[1],
                    "gender": columns[2],
                    "meaning": columns[3] if len(columns) > 4 else None,
                }
                q = """
                CREATE (synonym:Synoname $properties)
                RETURN synonym
                """
                self.graph.cypher_transaction(q, {"properties": properties})
                add += 1
        print("Added {} Synoname nodes.".format(add))
        print(
            "Note - use this cypher command to delete all Synoname nodes: MATCH (n:Synoname) DELETE n"
        )

    def _clone_original_values(self, properties=[]):
        for p in properties:
            q = f"""
            MATCH (n)
            WHERE n.{p} IS NOT NULL and n.og_{p} IS NULL
            SET n.og_{p} = n.{p}
            """
            self.graph.cypher_transaction(q)

    def _update_country(self):
        # set country = "United States" if country = "US"
        q_country = """
        MATCH (n)
        WHERE n.country IS NULL AND n.country_code = "US"
        SET n.country = "United States"
        """
        self.graph.cypher_transaction(q_country)

        # remove country2
        self.graph.cypher_transaction(
            "MATCH (n) WHERE n.country2 IS NOT NULL REMOVE n.country2"
        )

        # check country
        # q = "MATCH (n) WHERE n.country is NOT NULL RETURN n"
        # result = self.graph.cypher_transaction(q)
        # df = pd.DataFrame([dict(record[0]) for record in result])
        # df["country"].value_counts(dropna=False).to_csv("check_country.csv")

        # merge country
        replace_country = {
            "Austria": "Australia",
            "POLAND": "Poland",
        }
        for old_value, new_value in replace_country.items():
            q = f"""
            MATCH (n) WHERE n.country = "{old_value}"
            SET n.country = "{new_value}"
            """
            self.graph.cypher_transaction(q)

    def _update_state(self):
        # check states
        # q = "MATCH (n) WHERE n.state is NOT NULL RETURN n"
        # result = self.graph.cypher_transaction(q)
        # df = pd.DataFrame([dict(record[0]) for record in result])

        # df[["country", "state"]].value_counts(dropna=False).to_csv("check_state.csv")
        # df["state"].value_counts(dropna=False).to_csv("check_state_only.csv")

        # set state = state2 or lic_state if state = null
        for s in ["state2", "lic_state"]:
            self.graph.cypher_transaction(
                f"""
                MATCH (n)
                WHERE n.state IS NULL AND n.{s} IS NOT NULL
                SET n.state = n.{s}
                """
            )

        # update to correct state
        update_states = {
            "United States": get_us_states(),
            "Australia": {"WA": "Western Australia"},
            "Brazil": {"Espírito Santo": "Espiritu Santo", "San Paolo": "Sao Paolo"},
            "Italy": {"LI": "Livorno", "PR": "Parma"},
            "United Kingdom": {"London ": "London"},
        }
        for country, states in update_states.items():
            for old_value, new_value in states.items():
                self.graph.cypher_transaction(
                    f"""
                    MATCH (n)
                    WHERE n.country = "{country}" AND n.state = "{old_value}"
                    SET n.state = "{new_value}"
                    """
                )

    def _update_category(self):
        nurses = ["Nurse (OUS)", "Registered Nurse (RN)", "Nurse Practitioner (NP)"]
        self.graph.cypher_transaction(
            f"""
            MATCH (n)
            WHERE n.category in {nurses}
            SET n.category = "Nurse"
            """
        )

    def _update_specialty(self):
        # trim string
        self.graph.cypher_transaction(
            """
            MATCH (n)
            WHERE n.specialty IS NOT NULL
            SET n.specialty = trim(n.specialty)
            """
        )

        # merge specialty
        update_specialty = {
            "Interventional Cardiology (IC)": "Cardiology",
            "Cardiology, interventional cardiology secondary": "Cardiology",
            "Cardiologist, interventional cardiology as secondary": "Cardiology",
            "Cardiology, internal medicine secondary": "Cardiology",
            "Cardiology, interventional cardiology/internal medicine secondary": "Cardiology",
            "Cardiology, no secondary": "Cardiology",
            "Pediatric medicine, cardiology as a secondary": "Cardiology",
            "Pediatric medicine, cardiology secondary": "Cardiology",
            'Listed as "Cardiologist" in specialty with Interventional Cardiology as secondary': "Cardiology",
            'Listed as "Cardiologist" in specialty with internal medicine as secondary': "Cardiology",
            "Interventional Neuroradiology (INR)": "Neuro Radiology",
            "Neuro Radiologist ": "Neuro Radiology",
            "Interventional Neuroradiology": "Neuro Radiology",
            "Hematologist": "Hematology",
            "Emergency medicine, no secondary": "Emergency Medicine (EM)",
            "Physical Therapy (PT)": "Physical Therapy",
            "Pulmonary disease, critical care/internal medicine secondary": "Pulmonology",
        }
        for old, new in update_specialty.items():
            self.graph.cypher_transaction(
                f"""
                MATCH (n)
                WHERE n.specialty = '{old}'
                SET n.specialty = "{new}"
                """
            )

        # q = "MATCH (n) WHERE n.specialty is NOT NULL RETURN n"
        # result = self.graph.cypher_transaction(q)
        # df = pd.DataFrame([dict(record[0]) for record in result])

        # df["specialty"].value_counts(dropna=False).to_csv("check_specialty.csv")

    def data_optimization(self):
        # clone the property to keep the original value
        self._clone_original_values(["state", "country", "specialty"])

        # update country
        self._update_country()

        # update state
        self._update_state()

        # update category
        self._update_category()

        # update specialty
        self._update_specialty()
