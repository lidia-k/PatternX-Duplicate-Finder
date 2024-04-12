import glob
import numpy as np
import os
import pandas as pd
import platform
import subprocess

from src.dao.NEO4J_Graph import Graph
from src.utils import auto_config as config

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

    def _update_csv_file(self, csv_file):
        if 'speaker' not in csv_file and 'hcp' not in csv_file:
            print(f'Invalid file: {csv_file}')
            return

        df = pd.read_csv(csv_file) 
        file_type = 'sp' if 'speaker' in csv_file else 'po'
        name_str = csv_file.split('-')[3 if file_type == 'sp' else 2].split('.')[0]
        file_name = f'/data/{file_type}_{name_str}.csv'
        rename = SP_COLS if file_type == 'sp' else PO_COLS

        # Add uid column based on the node type        
        node_type = f'{file_type}_{name_str[:2]}'
        df['uid'] = [f'{node_type}_{i+2}' for i in range(len(df))]

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
            df['uid'] = [f'{node_type}_{i+3}' for i in range(len(df))]
            df['franchise'] = [name_str.capitalize() for i in range(len(df))]
        
        if 'vcheck' in csv_file:
            rename = PO_VC_COLS
            for col in ['b_first_name', 'b_last_name']:
                df[col] = df[col].str.capitalize()

        # Replace null values with np.nan
        null_val = [0, '0', 'N/A', '#N/A', 'N/A ', 'n/a (ask Carson Milner)', 'unknown']
        df.replace(null_val, np.nan, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]

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

        # Remove existing data files
        data_dir = './data'
        f_types = ['sp_*.csv', 'po_*.csv']
        for f_type in f_types:
            for f in glob.glob(f'{data_dir}/{f_type}'):
                os.remove(f)

        # Update csv files and load data to Neo4j
        data_bundles = glob.glob(f'{data_dir}/*.csv')
        for f in data_bundles:
            fname = self._update_csv_file(f)
            self._load_data_from_cypher(fname)
    
    @classmethod
    def _build_text(self, node):
        text = 'The following is the information of the health care provider.\n'
        for prop, val in node.items():
            if prop in [
                'uid', 'text', 'embedding', 'nppes_data', 'fname', 'lname', 
                'payments_to', 'license', 'lic_state', 'title', 'taxonomy', 
                'tax_code', 'addr1', 'addr2', 'qb_id', 'b_credential', 'b_middle_name',
                'b_last_updated', 'b_status', 'a_country_code', 't_code',
                't_primary', 'franchise', 'country2', 'state2'
            ]:
                continue
            if val is None:
                continue 

            prop = prop.replace('a_', '')
            prop = prop.replace('b_', '') 
            prop = prop.replace('t_', '') if prop.startswith('t_') else prop 
            prop = prop.replace('_', ' ') if '_' in prop else prop
            prop = prop.replace('sap', 'System Applications and Products in Data Processing(SAP)')
            prop = 'description' if prop == 'desc' else prop 
            prop = 'license' if prop == 'lisc' else prop 
            prop = 'National Provider Identifier(NPI)' if prop == 'npi' else prop
            prop = f'payment {prop}' if prop == 'currency' else prop
            prop = prop.replace('no', 'number')
            prop = prop.replace('fc', 'focus')
            prop = prop.replace('org', 'organization')

            text += f'The {prop.lower()} of the provider is {val}.\n'
        
        q = '''
            MATCH (n) WHERE n.uid = $uid
            SET n.text = $text
            '''
        return text, q

    def add_text_props(self):
        """
        Add text property to all nodes in the graph.
        We will vectorize the text property for similarity search.
        """
        driver = self.graph.get_driver()
        with driver.session() as session:
            q = 'MATCH (n) RETURN n'
            result = session.run(q).data()
            for node in result:
                node = node['n']
                text, update_q = self._build_text(node)
                session.run(update_q, uid=node['uid'], text=text)
        
