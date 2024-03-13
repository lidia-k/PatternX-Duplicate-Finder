import glob
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
        node_type = f'po_{csv_file.split("-")[2].split(".")[0][:2]}'
        df['id'] = [f'{node_type}_{i+1}' for i in range(len(df))]
        
        if 'hcp' in csv_file:
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

        df = df.rename(columns=rename)
        df.columns = [col.lower() for col in df.columns]

        int_cols = ['npi', 'qb_id', 'sap_no', 'license']
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        new_name = csv_file.replace('.csv', '_updated.csv')
        df.to_csv(new_name, index=False)
        print(f'Updated file: {new_name}')
        return new_name

    def _load_data_from_cypher(self, file_path):
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
