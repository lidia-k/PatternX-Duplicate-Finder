import glob
import numpy as np
import pandas as pd

from dao.NEO4J_Graph import Graph
from utils import auto_config as config


class DataProcessor:
    def __init__(self):
        self.graph = Graph(
            config.NEO4J_URL,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD
        )

    def _update_csv_files(self, csv_file):
        df = pd.read_csv(csv_file)

        if 'speaker' in csv_file:
            name_str = csv_file.split('-')[3].split('.')[0]  
            file_name = f'/data/sp_{name_str}.csv'
            node_type = f'sp_{name_str[:2]}'
            rename = {
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
            if 'all' not in csv_file:
                df['franchise'] = [name_str.capitalize() for i in range(len(df))]
                df.drop(columns=['Practice Type'], inplace=True)
                df.replace(0, np.nan, inplace=True)

        if 'hcp' in csv_file:
            name_str = csv_file.split('-')[2].split('.')[0]  
            file_name = f'/data/po_{name_str}.csv'
            node_type = f'po_{name_str[:2]}'
            rename = {
                'First Name': 'fname',
                'Last Name': 'lname',
                'Full Name': 'fullname',
                'National Physician ID': 'npi',
                'Email Address': 'email',
                'Quickbase Record ID#':  'qb_id',
                'Contact Type': 'ctype',
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
            df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]
    
            if 'vcheck' in csv_file:
                rename = {
                    'Full Name': 'fullname',
                    'b_first_name': 'fname',
                    'b_last_name': 'lname',
                }
                for col in ['b_first_name', 'b_last_name']:
                    df[col] = df[col].str.capitalize()

        df['id'] = [f'{node_type}_{i+1}' for i in range(len(df))]
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
        data_bundles = glob.glob('./data/*.csv')
        for f in data_bundles:
            fname = self._update_csv_files(f)
            self._load_data_from_cypher(fname)
