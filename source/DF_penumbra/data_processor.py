import glob
import numpy as np
import pandas as pd
import platform
import requests
import subprocess

from fuzzywuzzy import fuzz

from dao.NEO4J_Graph import Graph
from utils import auto_config as config


class DataProcessor:
    def __init__(self):
        self.graph = Graph(
            config.NEO4J_URL,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD
        )

    def _update_csv_file(self, csv_file):
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
                #df.drop(columns=['Practice Type'], inplace=True)
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
        #self.graph.wipe_database()

        # On Linux, docker exec chown and chmod the data directory
        if platform.system() == 'Linux':
            cmd = "docker exec neo4j /bin/bash -c 'chown -R 777:777 import/data && chmod -R 777 import/data'"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)    

        data_bundles = glob.glob('./data/*.csv')
        for f in data_bundles:
            fname = self._update_csv_file(f)
            self._load_data_from_cypher(fname)
    
    def _check_against_gov_registry(self, npi):
        api_url = "https://npiregistry.cms.hhs.gov/api/"
        params = {
                    "number": npi,
                    "version": 2.1
                }
        try: 
            res = requests.get(api_url, params=params)
        except Exception as e:
            raise e
        return res

    def validate_NPIs(self):        
        driver = self.graph.get_driver()
        with driver.session() as session:
            # Check for NPIs that are not 10 digits
            query = '''
                MATCH (n) 
                WHERE n.npi IS NOT NULL AND size(toString(n.npi)) <> 10 
                RETURN n.npi, n.id
                '''
            result = session.run(query).data()
            print(f'NPIs larger than 10 digits: {result}')   

            # Return NPIs that are 10 digits
            query = '''
                MATCH (n) 
                WHERE n.npi IS NOT NULL AND size(toString(n.npi)) = 10 
                RETURN n.npi, n.id, n.fname, n.lname, n.fullname
                '''
            results = session.run(query).data()
            output_f = open('npi_val.txt', 'w')
            for result in results:
                id = result['n.id']
                npi = result['n.npi']

                fullname = result.get('n.fullname', None)
                if not fullname:
                    fname = result.get('n.fname', None)
                    lname = result.get('n.lname', None)
                    if not fname or not lname:
                        print(f"Missing name for NPI {npi}, {id}")
                        continue
                    fullname = f'{fname} {lname}'

                # Check NPI and name against government registry
                res = self._check_against_gov_registry()
                reg_output = res.json()["results"]
                if not reg_output:
                    print(f'No data returned for NPI {npi}, {id}')
                elif len(reg_output) > 1:
                    print(f"More than one NPI returned for {npi}, {id}")
                else: 
                    basic = reg_output[0]['basic']
                    fname = reg_output[0]['basic'].get('first_name', None)
                    lname = reg_output[0]['basic'].get('last_name', None)
                    if not fname and not lname:
                        keys = list(basic.keys())
                        fname_key = [key for key in keys if 'first_name' in key]
                        lname_key = [key for key in keys if 'last_name' in key]
                        if not fname_key and not lname_key:
                            print(f"Missing name for NPI {npi}, {id} in the registry")
                            continue
                        else:
                            fname = basic[fname_key[0]]
                            lname = basic[lname_key[0]]
                    
                    name = f"{fname} {lname}"
                    ratio = fuzz.ratio(fullname.lower(), name.lower())
                    if ratio != 100:
                        output_f.write(f'{fullname} vs {name} : {ratio} for NPI {npi}, {id}\n')  
  
            output_f.close()    
            

