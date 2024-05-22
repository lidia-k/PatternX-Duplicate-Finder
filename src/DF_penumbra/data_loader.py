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

        #if 'speaker' in csv_file:
        #    df[['fname', 'lname']] = df['fullname'].str.split(n=1, expand=True)

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
        return file_name, df

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

    def _find_common_columns(self, df_dict):
        common_cols = set()
        for key, df in df_dict.items():
            if 'vcheck' in key:
                continue
            common_cols.intersection_update(df.columns) if common_cols else common_cols.update(df.columns)
        return common_cols

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
        df_dict = {}
        for f in data_bundles:
            fname, df = self._prepare_csv_file(f)
            self._load_data_from_cypher(fname)
            
            df_dict[fname] = df

