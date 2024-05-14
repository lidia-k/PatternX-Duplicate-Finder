import os
from itertools import combinations

import py_entitymatching as em
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from src.dao.NEO4J_Graph import Graph
from src.DF_penumbra import constants
from src.DF_penumbra.utils import process_int_cols
from src.utils import auto_config as config


class DataPreprocessor:
    def __init__(self, data_dir, include_npi):
        self.graph = Graph(
            config.NEO4J_URL,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD
        )
        self.data_dir = data_dir
        self.include_npi = include_npi
    
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

    def _handle_int_cols(self, df):
        int_cols = ['ltable_' + col for col in constants.INT_COLS] + ['rtable_' + col for col in constants.INT_COLS]
        df = process_int_cols(df, int_cols)
        df.replace(0, np.nan, inplace=True)
        return df

    def _create_df(self, result, label=0):
        node_pairs = [(record[0], record[1]) for record in result]

        all_props = set()
        all_props.update(['lic_state', 'title', 'license'])
        for node_a, node_b in node_pairs:
            all_props.update(dict(node_a).keys())
            all_props.update(dict(node_b).keys())
        
        if not self.include_npi:
            all_props.remove('npi')

        rows = []
        for node_a, node_b in node_pairs:
            row = {}
            for prop in all_props:
                row[f'ltable_{prop}'] = node_a.get(prop, np.nan)
            for prop in all_props:
                row[f'rtable_{prop}'] = node_b.get(prop, np.nan)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = self._handle_int_cols(df)
        df['label'] = label

        #df = df.loc[:, ['label', 'ltable_uid', 'rtable_uid', 'ltable_npi', 'rtable_npi', 'ltable_fullname', 'rtable_fullname']]
        return df

    def _build_non_matching_pairs(self, limit):
        """
        For nodes that have different npis, create a non-matching pair.
        """
        q = f'''
        MATCH (n)
        WHERE NOT n:Master AND n.npi IS NOT NULL AND NOT EXISTS ((n)-[:r1_npi]-())
        RETURN n
        '''
        result = self.graph.cypher_transaction(q)

        df = pd.DataFrame([dict(record[0]) for record in result])
        if not self.include_npi:
            df.drop(columns=['npi'], inplace=True)
        
        pairs = [(df.iloc[i], df.iloc[j]) for i, j in combinations(range(len(df)), 2)]
        paired_data = []
        for left, right in pairs:
            left_dict = {
                'ltable_' + col: val for col, val in left.items()}
            right_dict = {'rtable_' + col: val for col, val in right.items()}
            paired_data.append({**left_dict, **right_dict})

        paired_df = pd.DataFrame(paired_data)
        self._handle_int_cols(paired_df)

        sample_df = paired_df.sample(n=limit, random_state=1)
        sample_df['label'] = 0
        
        dropped_df = paired_df.drop(sample_df.index)
        dropped_df.to_csv('dropped.csv', index=False)

        print(f'The number of non-matching pairs:', len(sample_df))
        return sample_df, dropped_df
       
    def _build_matching_pairs(self, size):
        """
        Retrieves pairs of nodes connected by specified types of edges.

        It uses a Cypher query to find all node pairs (a, b) such that:
        - There is a specified type of edge from node a to node b.
        - The 'uid' of node a is not equal to the 'uid' . of node b.
        - The query returns distinct node pairs to ensure no duplicates.
       
        Returns:
            pandas.DataFrame: 
            A DataFrame containing the distinct pairs of nodes that match the criteria, with a column labeled '1'.
        """
        edge_types = ['r1_' + et for et in constants.EDGE_TYPES]
        q = f'''
        UNWIND {edge_types} AS type
        MATCH (a)-[r]->(b)
        WHERE type(r) = type AND a.uid <> b.uid
        RETURN DISTINCT a, b
        '''
        if size is not None:
            q += f'LIMIT {size}'

        result = self.graph.cypher_transaction(q)
        df = self._create_df(result, label=1)
        df = shuffle(df, random_state=1).reset_index(drop=True)

        # Split the first 46 rows for the test case 2
        df_46 = df.loc[:46, :].to_csv('46.csv', index=False)
        matching_df = df.loc[46:, :]
        print(f'The number of matching pairs:', len(matching_df))
        return matching_df
        
    def _split_tables(self, df):
        """
        Split the data into A, B, and C tables.
        """
        df = shuffle(df, random_state=1).reset_index(drop=True)
        df['id'] = df.index
        df['ltable_id'] = df['id']
        df['rtable_id'] = df['id']
        #combined_df = combined_df.loc[:, ['label', 'ltable_uid', 'rtable_uid', 'ltable_npi', 'rtable_npi', 'ltable_fullname', 'rtable_fullname']]
        df.to_csv('C.csv', index=False)

        ltable_cols = [col for col in df.columns if 'ltable_' in col]
        rtable_cols = [col for col in df.columns if 'rtable_' in col]
        
        A = df[ltable_cols]
        A.columns = [col.replace('ltable_', '') for col in ltable_cols]
        A = A.rename(columns={'id': 'ltable_id'})
        A.to_csv('A.csv', index=False)

        B = df[rtable_cols]
        B.columns = [col.replace('rtable_', '') for col in rtable_cols]
        B = B.rename(columns={'id': 'rtable_id'})
        B.to_csv('B.csv', index=False)
    
    def _load_data(self, df):
        self._split_tables(df)

        A = em.read_csv_metadata('A.csv', key='ltable_id')
        B = em.read_csv_metadata('B.csv', key='rtable_id')
        C = em.read_csv_metadata(
            'C.csv', key='id', ltable=A, rtable=B,
            fk_ltable='ltable_id', fk_rtable='rtable_id'
        )
        return A, B, C

    def prepare_training_data(self, skewed_factor, size):
        """
        Label the pairs as matching or non-matching and prepare the training data. 
        """
        matching_df = self._build_matching_pairs(size)
        
        limit = len(matching_df) * skewed_factor
        non_matching_df, _ = self._build_non_matching_pairs(limit=limit)
        matching_df = matching_df[non_matching_df.columns]
        
        combined_df = pd.concat([matching_df, non_matching_df], sort=False)
        return self._load_data(combined_df)
    
    def prepare_test_data(self):
        result = []
        props = ['email', 'sap_no']
        for prop in props:
            matching_q = f'''
            MATCH (a), (b)
            WHERE a.{prop} = b.{prop} AND a <> b AND NOT (a)-[]-(b)
            RETURN DISTINCT a, b
            LIMIT 20
            '''
            matching_result = self.graph.cypher_transaction(matching_q)
            result.extend(matching_result)
        
        non_matching_q = '''
        MATCH (a), (b)
        WHERE a.email <> b.email AND a.sap_no = b.sap_no AND a <> b AND NOT (a)-[]-(b)
        RETURN DISTINCT a, b
        LIMIT 14
        '''
        non_matching_result = self.graph.cypher_transaction(non_matching_q)
        result.extend(non_matching_result)

        df = self._create_df(result)

        # Add the label 1 training data to the test data
        df_46 = pd.read_csv('46.csv')
        df = pd.concat([df, df_46], sort=False)

        df = self._handle_int_cols(df)
        df.drop(columns=['label'], inplace=True)
        #df[['ltable_fullname', 'rtable_fullname',
        #    'ltable_email', 'rtable_email', 'ltable_sap_no', 'rtable_sap_no']].to_csv('test.csv', index=False)
        return df

    def prepare_alldata_exclude_traindata(self):
        # exclude the data used for training
        ltable, rtable, data = self.prepare_training_data(skewed_factor=2, size=None)
        exclude_uids = ltable["uid"].unique().tolist()
        exclude_uids.extend(rtable["uid"].unique().tolist())
        exclude_uids = list(set(exclude_uids))

        # fetch all data from Neo (exclude exclude_uids)
        q = f'''
        MATCH (n)
        WHERE NOT n.uid in {exclude_uids}
        RETURN n
        '''
        result = self.graph.cypher_transaction(q)

        df = pd.DataFrame([dict(record[0]) for record in result])
        
        pairs = [(df.iloc[i], df.iloc[j]) for i, j in combinations(range(len(df)), 2)]
        paired_data = []
        for left, right in pairs:
            left_dict = {'ltable_' + col: val for col, val in left.items()}
            right_dict = {'rtable_' + col: val for col, val in right.items()}
            paired_data.append({**left_dict, **right_dict})

        paired_df = pd.DataFrame(paired_data)
        df = self._handle_int_cols(paired_df)
        return df

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
        
