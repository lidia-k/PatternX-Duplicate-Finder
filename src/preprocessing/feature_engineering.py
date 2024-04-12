import typing as tp 
import py_entitymatching as em
import pandas as pd 
from itertools import permutations

class FeatureEngineer:
    def __init__(self, strategy):
        self.strategy = strategy

    def execute_strategy(self, data):
        return self.strategy.execute(data)
    

class PenumbraFeatureEnginner(FeatureEngineer):
    def __init__(self, strategy):
        super().__init__(strategy)
        self.blocking_config = {
            "level_0": "National Physician ID",       
            "level_1": "Full Name", 
        }

    def execute_strategy(self, A, B, blocking_config: tp.Dict):
        # retrieve metadata for Magenlla
        A = A.reset_index(drop=True).reset_index().rename(columns = {"index": "id"}).drop_duplicates()
        A['id'] = 'ltable_' + A['id'].astype(str)
        B = B.reset_index(drop=True).reset_index().rename(columns = {"index": "id"}).drop_duplicates()
        B['id'] = 'rtable_' + B['id'].astype(str)

        em.set_key(A, 'id')
        em.set_key(B, 'id')

        # level 0 blocking
        creteria = blocking_config.pop('level_0')
        if creteria is not None:
            cols = list(set(B.columns) & set(A.columns))
            ob = em.OverlapBlocker()
            K = ob.block_tables(A, B, creteria, creteria, l_output_attrs=cols, r_output_attrs=cols, overlap_size=1)

            # level 1,2,... blocking
            levels = [f"level_{i}" for i in range(1, len(blocking_config))]
            for level in levels:
                creteria = blocking_config[level]
                K = ob.block_candset(K, creteria, creteria, overlap_size=1)


            ltable_columns = [col for col in K.columns if col.startswith("ltable")]
            rtable_columns = [col for col in K.columns if col.startswith("rtable")]

            new_A = K[ltable_columns]
            new_B = K[rtable_columns]
            columns = [col.replace("ltable_", "") for col in ltable_columns]
            new_A.columns = columns
            new_B.columns = columns
            new_A  = new_A.drop_duplicates()
            new_B  = new_B.drop_duplicates()
        else:
            new_A = A.copy()
            new_B = B.copy()
    
        df = pd.concat([new_A, new_B])
        df, _, _, _, _, _ = self.label_duplicated_data(df, skewed_factor=2)
        del df['id']

        return df 
    
    def process_group(self, df:pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df.reset_index(drop=True)
        cols1 = df.columns
        cols2 = cols1.copy()
        cols1 = ['ltable_' + c for c in cols1]
        cols2 = ['rtable_' + c for c in cols2]
        cols = cols1 + cols2
        
        if n == 1:
            new_df = pd.DataFrame(columns=cols)
            return new_df

        iter = permutations(range(n), 2)
        data = []
        for pair in iter:
            row = df.loc[pair[0]].tolist() + df.loc[pair[1]].tolist()
            data.append(row)
        new_df = pd.DataFrame(data= data, columns=cols)
        new_df['label'] = 1
        return new_df

    def label_duplicated_data(self, df: pd.DataFrame, skewed_factor: int = 5) -> pd.DataFrame:
        """Try to label data by some conditions. The conditions we implement in this function are:
        1) Full name match
        2) National Physician ID  
        3) Email Address
        
        whenever a pair satisfies one of those condition, its label will be assinged to 1. Otherwise, its label will be assigned to 0.
        Args:
            df DataFrame: dataframe with label
        """
        duplicated_dfs = []
        full_name_duplicated_df = df[df.duplicated(keep=False, subset=['Full Name'])]
        full_name_duplicated_df = full_name_duplicated_df.groupby('Full Name').apply(self.process_group).reset_index(drop=True)
        if not full_name_duplicated_df.empty:
            duplicated_dfs.append(full_name_duplicated_df)

        NPI_duplicated_df = df[(df['National Physician ID']!= 'UNKNOWN') & (df.duplicated(keep=False, subset=['National Physician ID'])) & (df['National Physician ID']!='OUS')]
        NPI_duplicated_df = NPI_duplicated_df.groupby('National Physician ID').apply(self.process_group).reset_index(drop=True)
        if not NPI_duplicated_df.empty:
            duplicated_dfs.append(NPI_duplicated_df)

        email_duplicated_df = df[(df['Email Address']!= 'UNKNOWN') & (df.duplicated(keep=False, subset=['Email Address']))]
        email_duplicated_df = email_duplicated_df.groupby('Email Address').apply(self.process_group).reset_index(drop=True)
        if not email_duplicated_df.empty:
            duplicated_dfs.append(email_duplicated_df)

        label_1_df = pd.concat(duplicated_dfs)
        
        label_1_df = label_1_df.rename(columns={'ltable_id': 'id'}).drop(columns=['rtable_id'])
        label_1_df['label'] = 1

        label_0_df = df[~df.id.isin(label_1_df.id.unique())]

        label_0_df = self.process_group(label_0_df)
        label_0_df = label_0_df.rename(columns={'ltable_id': 'id'}).drop(columns=['rtable_id'])
        label_0_df['label'] = 0
        
        # sampling with skewed factor
        n_1 = label_1_df.shape[0]
        n_0 = len(label_0_df)
        
        if n_0 < n_1 :
            n = n_0 * skewed_factor
            n = n if n < n_1 else len(label_0_df)
            label_1_df = label_1_df.sample(n, random_state=1)
        else:
            n = n_1 * skewed_factor
            n = n if n < n_0 else len(label_0_df)
            label_0_df = label_0_df.sample(n, random_state=1)

        data_df = pd.concat([label_0_df, label_1_df]).drop_duplicates()
        return data_df, label_1_df, label_0_df, full_name_duplicated_df, NPI_duplicated_df, email_duplicated_df