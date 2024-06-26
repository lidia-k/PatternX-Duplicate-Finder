import glob
import os
import platform
import subprocess

import numpy as np
import pandas as pd

from src.dao.NEO4J_Graph import Graph
from src.DF_penumbra import constants
from src.DF_penumbra.utils import process_bi_emails, process_biSAP_number, process_columns, process_int_cols
from src.utils import auto_config as config


class Neo4jDataLoader:
    def __init__(self, data_dir):
        self.graph = Graph(
            config.NEO4J_URL,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD 
        )
        self.data_dir = data_dir
    
    def _process_names(self, file_name):
        file_type = 'sp' if 'speaker' in file_name else 'po'
        renamed_cols = constants.SP_COLS if file_type == 'sp' else constants.PO_COLS
        name_str = file_name.split('-')[3 if file_type == 'sp' else 2].split('.')[0]
        file_name = f'{self.data_dir}/{file_type}_{name_str}.csv'
        node_type = f'{file_type}_{name_str[:2]}'
        return renamed_cols, name_str, file_name, node_type

    def _prepare_csv_file(self, csv_file):
        if 'speaker' not in csv_file and 'hcp' not in csv_file:
            print(f'Invalid file: {csv_file}')
            return

        rename, name_str, file_name, node_type = self._process_names(csv_file)
        df = pd.read_csv(csv_file)        

        # Add uid column based on the node type        
        df['uid'] = [f'{node_type}_{i+2}' for i in range(len(df))]
        
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

        if 'speaker' in csv_file and not 'all' in csv_file:      
            df = pd.read_csv(csv_file, header=1)
            df['uid'] = [f'{node_type}_{i+3}' for i in range(len(df))]
            df['franchise'] = [name_str.capitalize() for i in range(len(df))]
        
        if 'vcheck' in csv_file:
            df.columns = [process_columns(col) for col in df.columns]
            rename = constants.PO_VC_COLS
            for col in ['first_name', 'last_name']:
                df[col] = df[col].str.capitalize()

        # Renmae columns and convert to lowercase
        df = df.rename(columns=rename)
        df.columns = [col.lower() for col in df.columns]

        # Split a row with two sap numbers into two rows
        if 'sap_no' in df.columns:
            df = process_biSAP_number(df)

        # Split a row with two emails into two rows
        if 'email' in df.columns:
            df = process_bi_emails(df)

        # Convert columns to int
        df = process_int_cols(df, constants.INT_COLS)
        
        # Replace null values with np.nan
        null_val = [0, '0', 'N/A', '#N/A', 'N/A ', 'n/a (ask Carson Milner)', 'unknown']
        df.replace(null_val, np.nan, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]

        df.to_csv(f'./{file_name}', index=False)
        print(f'Updated file: {file_name}')
        return file_name

    def _load_data_from_cypher(self, file_path):
        if 'sp' in file_path:
            cypher_file = f'{self.data_dir}/sp_all.cypher' if 'all' in file_path else f'{self.data_dir}/sp.cypher'
        else: 
            cypher_file = f'{self.data_dir}/po_vcheck.cypher' if 'vcheck' in file_path else f'{self.data_dir}/po.cypher'

        with open(cypher_file, 'r') as f:
            query = f.read()

        query = query.format(file_path=f'file:///{file_path}')
        self.graph.cypher_transaction(query)
        print(f'Loaded data from {file_path} to Neo4j')

    def load_csv_to_neo4j(self):
        self.graph.wipe_database()

        # On Linux, docker exec chown and chmod the data directory
        if platform.system() == 'Linux':
            cmd = "docker exec neo4j /bin/bash -c 'chown -R 777:777 import/data && chmod -R 777 import/data'"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)    

        # Remove existing data files
        f_types = ['sp_*.csv', 'po_*.csv']
        for f_type in f_types:
            for f in glob.glob(f'{self.data_dir}/{f_type}'):
                os.remove(f)

        # Update csv files and load data to Neo4j
        data_bundles = glob.glob(f'{self.data_dir}/*.csv')
        for f in data_bundles:
            fname = self._prepare_csv_file(f)
            self._load_data_from_cypher(fname)

    def import_human_labeled(self, infile):
        df = pd.read_csv(infile)
        df.replace("-", np.nan, inplace=True)
        node_df = df.drop(columns=["match", "group"])
        relation_df = df[df["group"].notna()]
        driver = self.graph.get_driver()
        with driver.session() as session:
            # create node if not exists
            result = session.run(
                "MATCH (n) WHERE (n:Provider OR n:Speaker) RETURN n.uid"
            )
            exists_uids = [record[0] for record in result]
            all_uids = df["uid"].to_list()
            non_exists_id = list(set(all_uids) - set(exists_uids))
            if non_exists_id:
                non_exists_df = node_df[node_df["uid"].isin(non_exists_id)]

                def create_query(record):
                    node_label = "Speaker" if "sp_" in record["uid"] else "Provider"
                    query = f"CREATE (n:{node_label} {{"
                    query += ", ".join([f"{key}: ${key}" for key in record.keys()])
                    query += "})"
                    return query

                for index, row in non_exists_df.iterrows():
                    record = row.to_dict()
                    filtered_record = {k: v for k, v in record.items() if pd.notna(v)}
                    # insert node
                    session.run(
                        create_query(filtered_record), parameters=filtered_record
                    )
                print(f"Added nodes: {non_exists_df}")

            # remove r0_other
            session.run("MATCH ()-[r:r0_other]->() DELETE r")
            # label 1 (x) pair
            label1_pairs = []
            grouped = relation_df.groupby("group")
            for group, group_df in grouped:
                uids = group_df["uid"].dropna().unique()
                new_group_df = group_df.set_index("uid", drop=False)
                for i in range(len(uids)):
                    for j in range(i + 1, len(uids)):
                        left = new_group_df.loc[uids[i]]
                        right = new_group_df.loc[uids[j]]
                        if left["match"] == "x" and right["match"] == "x":
                            result = session.run(
                                f"""MATCH (a), (b)
                                WHERE (a:Provider OR a:Speaker) AND (b:Provider OR b:Speaker)
                                    AND a.uid='{uids[i]}' AND b.uid='{uids[j]}'
                                CREATE (a)-[:r0_other]->(b)
                                RETURN a, b"""
                            )
                            label1_pairs.append([left.to_dict(), right.to_dict()])
        driver.close()
        return node_df, label1_pairs
