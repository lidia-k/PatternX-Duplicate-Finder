import glob
import os
import platform
import subprocess

import numpy as np
import pandas as pd

from src.dao.NEO4J_Graph import Graph
from src.DF_penumbra import constants
from src.DF_penumbra.utils import (get_synonyms, process_bi_emails, 
                                   process_biSAP_number, process_columns, 
                                   process_int_cols)
from src.utils import auto_config as config


class Neo4jDataLoader:
    def __init__(self, data_dir=None):
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

        # Rename columns and convert to lowercase
        df = df.rename(columns=rename)
        df.columns = [col.lower() for col in df.columns]
        
        if 'speaker' in csv_file:
            df[['fname', 'lname']] = df['fullname'].str.split(' ', n=1, expand=True)

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
