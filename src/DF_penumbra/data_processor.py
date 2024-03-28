import glob
import numpy as np
import pandas as pd
import platform
import requests
import subprocess

from fuzzywuzzy import fuzz

from dao.NEO4J_Graph import Graph
from utils import auto_config as config

SP_COLS = {
    'Request Type': 'type',
    'HCP Full Name': 'fullname',
    'NPI Number': 'npi',
    'HCP Category': 'category',
    'HCP Specialty': 'specialty',
    'HCP Institution / Customer Name': 'org',
    'HCP Institution': 'org',
    'SAP Customer ID': 'sap_no',
    'Institution City': 'city',
    'Institution State': 'state',
    'HCP Country': 'country',
    'HCP Email ': 'email',
    'HCC ID': 'hid',
    'HCP NPI#': 'npi',
    'SAP Supplier ID': 'sap_no',
    'Presentation Title': 'title',
    'Country': 'country2',
}
PO_COLS = {
    'First Name': 'fname',
    'Last Name': 'lname',
    'Full Name': 'fullname',
    'National Physician ID': 'npi',
    'National Physician ID/RPPS ID': 'npi',
    'Email Address': 'email',
    'Quickbase Record ID#':  'qb_id',
    'HCP Category': 'category',
    'Payments Made To:': 'payments_to',
    'SAP Number': 'sap_no',
    'SAP Entity Name': 'sap_name',
    'State/Region/Province': 'state1',
    'State/Region': 'state2',
    'State License #': 'license',
    'License State (US)': 'lic_state',
    'Focus Area': 'fc_area',
    'Taxonomy Code': 'tax_code',
    'Payment Currency': 'currency',
    'Primary Address': 'addr1',
    'Mailing Address': 'addr2',
    'Primary Organization': 'org',
    'Primary Organization Type': 'org_type',
}
PO_VC_COLS = {
    'Full Name': 'fullname',
    'b_first_name': 'fname',
    'b_last_name': 'lname',
    'a_country_code': 'country'
}

class DataProcessor:
    def __init__(self):
        self.graph = Graph(
            config.NEO4J_URL,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD
        )

    def _convert_row_to_text(self, df):
        """
        We're converting the row to text so that we can vectorize it and use it for similarity search.
        """
        for i, row in df.iterrows():
            text = 'The following is the information of the health care provider.\n'
            
            for col, val in row.items():
                if col in ['text', 'id', 'nppes_data']:
                    continue
                if pd.isnull(val):
                    continue 
                if isinstance(val, float):
                   val = int(val)

                col = col.replace('HCP ', '')
                col = col.replace('#', ' Number')
                col = col.replace('a_', '')
                col = col.replace('b_', '') 
                col = col.replace('t_', '') if col.startswith('t_') else col 
                col = col.replace('_', ' ')
                col = col.replace('SAP', 'System Applications and Products in Data Processing(SAP)')
                col = 'description' if col == 'desc' else col 
                col = 'license' if col == 'lisc' else col 
                
                if col in ['National Physician ID', 'npi', 'NPI Number']:
                    col = 'National Provider Identifier(NPI)' 
                if col == 'Payments Made To:':
                    text += f"The provider's payments are paid to {val}.\n"
                else: 
                    text += f'The {col.lower()} of the provider is {val}.\n'

            df.at[i, 'text'] = text
        return df 
    
    def _update_csv_file(self, csv_file):
        if 'speaker' not in csv_file and 'hcp' not in csv_file:
            print(f'Invalid file: {csv_file}')
            return

        df = pd.read_csv(csv_file) 
        file_type = 'sp' if 'speaker' in csv_file else 'po'
        name_str = csv_file.split('-')[3 if file_type == 'sp' else 2].split('.')[0]
        file_name = f'/data/{file_type}_{name_str}.csv'
        rename = SP_COLS if file_type == 'sp' else PO_COLS

        # Add id column based on the node type        
        node_type = f'{file_type}_{name_str[:2]}'
        df['id'] = [f'{node_type}_{i+2}' for i in range(len(df))]

        # Drop unnecessary columns
        drop_cols = [
            'Prefix', 'Status', 'Contact Type', 'Reportable HCP', 
            'Request Type', 'HCC ID', 't_primary', 'Practice Type'
        ]
        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
    
        if 'speaker' in csv_file and not 'all' in csv_file:      
            df = pd.read_csv(csv_file, header=1)
            df['id'] = [f'{node_type}_{i+3}' for i in range(len(df))]
            df['franchise'] = [name_str.capitalize() for i in range(len(df))]
        
        if 'vcheck' in csv_file:
            rename = PO_VC_COLS
            for col in ['b_first_name', 'b_last_name']:
                df[col] = df[col].str.capitalize()

        # Replace null values with np.nan
        null_val = [0, '0', 'N/A', '#N/A', 'N/A ', 'n/a (ask Carson Milner)', 'unknown']
        df.replace(null_val, np.nan, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]

        # Convert row to text
        df = self._convert_row_to_text(df)
        # Renmae columns and convert to lowercase
        df = df.rename(columns=rename)
        df.columns = [col.lower() for col in df.columns]

        # Convert columns to int
        int_cols = ['npi', 'qb_id', 'sap_no', 'license']
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        df.to_csv(f'./{file_name}', index=False)
        print(f'Updated file: {file_name}')
        return file_name

    def _load_data_from_cypher(self, file_path):
        if 'sp' in file_path:
            cypher_file = './data/sp_all.cypher' if 'all' in file_path else './data/sp.cypher'
        else: 
            cypher_file = './data/po_vcheck.cypher' if 'vcheck' in file_path else './data/po.cypher'

        with open(cypher_file, 'r') as f:
            query = f.read()

        query = query.format(file_path=f'file:///{file_path}')
        self.graph.cypher_transaction(query)
        print(f'Loaded data from {file_path} to Neo4j')

    def import_csv_to_neo4j(self):
        self.graph.wipe_database()

        # On Linux, docker exec chown and chmod the data directory
        if platform.system() == 'Linux':
            cmd = "docker exec neo4j /bin/bash -c 'chown -R 777:777 import/data && chmod -R 777 import/data'"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)    

        data_bundles = glob.glob('./data/*.csv')
        for f in data_bundles:
            fname = self._update_csv_file(f)
            self._load_data_from_cypher(fname)
    
