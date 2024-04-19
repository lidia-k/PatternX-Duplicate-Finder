import glob
import numpy as np
import os
import pandas as pd
import platform
import subprocess

from src.dao.NEO4J_Graph import Graph
from src.DF_penumbra.duplicate_finder import EDGE_TYPES
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
    'first_name': 'fname',
    'middle_name': 'mname',
    'last_name': 'lname',
    'country_code': 'country',
    'address_1': 'addr1',
    'address_2': 'addr2',
    'lisc': 'license'
}

class DataProcessor:
    def __init__(self, data_dir):
        self.graph = Graph(
            config.NEO4J_URL,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD
        )
        self.data_dir = data_dir
        
    def _process_bi_emails(self, df):
        df['email'] = df['email'].astype(str).str.split('; ')
        df = df.explode('email')
        return df
    
    def _process_biSAP_number(self, df):
        df['sap_no'] = df['sap_no'].astype(str).str.split('\n')
        df = df.explode('sap_no')

        if 'sap_name' in df.columns:
            df['sap_name'] = df['sap_name'].astype(str).str.split('\n')
            df = df.explode('sap_name')
        return df 

    def _process_columns(self, prop):
        prop = prop.replace('a_', '')
        prop = prop.replace('b_', '') 
        prop = prop.replace('t_', '') if prop.startswith('t_') else prop 
        return prop
    
    def _process_names(self, file_name):
        file_type = 'sp' if 'speaker' in file_name else 'po'
        renamed_cols = SP_COLS if file_type == 'sp' else PO_COLS
        name_str = file_name.split('-')[3 if file_type == 'sp' else 2].split('.')[0]
        file_name = f'{self.data_dir}/{file_type}_{name_str}.csv'
        node_type = f'{file_type}_{name_str[:2]}'
        return renamed_cols, name_str, file_name, node_type
            
    def _update_csv_file(self, csv_file):
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
            df.columns = [self._process_columns(col) for col in df.columns]
            rename = PO_VC_COLS
            for col in ['first_name', 'last_name']:
                df[col] = df[col].str.capitalize()

        # Renmae columns and convert to lowercase
        df = df.rename(columns=rename)
        df.columns = [col.lower() for col in df.columns]

        # Split a row with two sap numbers into two rows
        if 'sap_no' in df.columns:
            df = self._process_biSAP_number(df)

        # Split a row with two emails into two rows
        if 'email' in df.columns:
            df = self._process_bi_emails(df)

        # Convert columns to int
        int_cols = ['npi', 'qb_id', 'sap_no', 'license']
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int).astype('Int64')
        
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

    def import_csv_to_neo4j(self):
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
            fname = self._update_csv_file(f)
            self._load_data_from_cypher(fname)
    
    def detect_high_missing_features(self, missing_percentage_threshold=59.2):
        q = '''
        // Collect all unique property keys from all nodes
        MATCH (n)
        WITH COLLECT(n) AS nodes
        UNWIND nodes AS node
        UNWIND keys(node) AS key
        WITH key, COUNT(DISTINCT node) AS nonNullCount, SIZE(nodes) AS totalNodes
        RETURN key, totalNodes, nonNullCount, totalNodes - nonNullCount AS nullCount
        '''
        result = self.graph.cypher_transaction(q)
        total = result[0][1]
        for prop in result:
            prop_name = prop[0]
            null_count = prop[3]
            missing_percentage = (null_count / total) * 100
            if missing_percentage > missing_percentage_threshold:
                print(f'{prop_name} has {missing_percentage}% missing values')

    def build_matching_pairs(self):
        edge_types = ['r1_' + et for et in EDGE_TYPES]
        q = f'''
        UNWIND {edge_types} AS type
        MATCH (a)-[r]->(b)
        WHERE type(r) = type AND a.uid <> b.uid
        RETURN DISTINCT a, b
        '''
        result = self.graph.cypher_transaction(q)
        node_pairs = [(record[0], record[1]) for record in result]

        all_props = set()
        for node_a, node_b in node_pairs:
            all_props.update(dict(node_a).keys())
            all_props.update(dict(node_b).keys())

        rows = []
        for node_a, node_b in node_pairs:
            row = {}
            for prop in all_props:
                row[f'ltable_{prop}'] = node_a.get(prop, np.nan)
            for prop in all_props:
                row[f'rtable_{prop}'] = node_b.get(prop, np.nan)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(f'{self.data_dir}/matching_pairs.csv', index=False)

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

            prop = self._process_columns(prop)
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
        
