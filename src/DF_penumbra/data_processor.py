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
        for i, row in df.iterrows():
            text = 'The following is the information of the health care provider.\n'
            
            for col in df.columns:
                if col == 'text':
                    continue
                if col == 'id':
                    continue 
                if col == 'nppes_data':
                    continue

                val = row[col]
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
                if pd.isnull(val) or val in ['N/A', '#N/A', 'N/A ', 'n/a (ask Carson Milner)']:
                    continue 
                if isinstance(val, float):
                   val = int(val)

                stc = f'The {col.lower()} of the provider is {val}.\n'
                if col == 'Payments Made To:':
                    stc = f"The provider's payments are paid to {val}.\n"
                text += stc

            df.at[i, 'text'] = text
        return df 
    
    def _update_csv_file(self, csv_file):
        df = pd.read_csv(csv_file)

        if 'speaker' in csv_file:
            name_str = csv_file.split('-')[3].split('.')[0]  
            file_name = f'/data/sp_{name_str}.csv'
            node_type = f'sp_{name_str[:2]}'
            rename = SP_COLS
            
            if 'all' in csv_file:
                df['id'] = [f'{node_type}_{i+2}' for i in range(len(df))]
                df.drop(columns=['Request Type', 'HCC ID'], inplace=True)
            else:
                df['id'] = [f'{node_type}_{i+3}' for i in range(len(df))]
                df['franchise'] = [name_str.capitalize() for i in range(len(df))]
                #df.drop(columns=['Practice Type'], inplace=True)
                df.replace(0, np.nan, inplace=True) 
                df.replace('0', np.nan, inplace=True) 
        
        elif 'hcp' in csv_file:
            name_str = csv_file.split('-')[2].split('.')[0]  
            file_name = f'/data/po_{name_str}.csv'
            node_type = f'po_{name_str[:2]}'
            df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]
            df['id'] = [f'{node_type}_{i+2}' for i in range(len(df))]
            rename = PO_COLS

            if 'vcheck' in csv_file:
                rename = PO_VC_COLS
                df.drop(columns=['t_primary'], inplace=True)
                for col in ['b_first_name', 'b_last_name']:
                    df[col] = df[col].str.capitalize()
            else:
                df.drop(columns=['Prefix', 'Status', 'Contact Type'], inplace=True)
                if 'Reportable HCP' in df.columns:
                    df.drop(columns=['Reportable HCP'], inplace=True)   
        
        else: 
            print(f'File {csv_file} not recognized')
        
        df = self._convert_row_to_text(df)
        df = df.rename(columns=rename)
        df.columns = [col.lower() for col in df.columns]

        int_cols = ['npi', 'qb_id', 'sap_no', 'license']
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        df.to_csv(f'./{file_name}', index=False)
        print(f'Updated file: {file_name}')
        return file_name

    def _load_data_from_cypher(self, file_path):
        if 'sp' in file_path:
            cypher_file = './data/sp.cypher'
            if 'all' in file_path:
                cypher_file = './data/sp_all.cypher'
        else: 
            cypher_file = './data/po.cypher'
            if 'vcheck' in file_path:
                cypher_file = './data/po_vcheck.cypher'

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
    
