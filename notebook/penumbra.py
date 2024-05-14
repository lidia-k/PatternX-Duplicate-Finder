#===============================================================================
# intent:  define functions for running Provider matching for customer Penumbra.
#          these functions are called from match-penumbra.py
#===============================================================================

import pdb, io, sys, os, pandas as pd, numpy as np, matplotlib.pyplot as plt
import typing as tp, seaborn as sns
from itertools import combinations, permutations
from fastai.tabular.all import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import src.utils.auto_config as config 
import msoffcrypto, py_entitymatching as em

def build_data_pair( items_in_A: tp.List[ tp.Union[str, pd.DataFrame]], items_in_B: \
    tp.List[ tp.Union[str, pd.DataFrame]]) -> tp.Tuple[pd.DataFrame]:
    def process_items(items:tp.Union[str, pd.DataFrame]) -> tp.List[pd.DataFrame]:
        item_dfs = []
        for item in items:
            if isinstance(item, str):
                item_df = pd.read_csv(item)
                item_dfs.append(item_df)
            elif  isinstance(item, pd.DataFrame):  item_dfs.append(item)
            else: raise Exception("Data type is not supported!")
        return item_dfs
    A_items = process_items(items_in_A)
    B_items = process_items(items_in_B)
    return (pd.concat(A_items), pd.concat(B_items))

#----------------- clean data -------------------

# clean SAP, npi, qbid.  add "UNKNOWN" where relevant

def normalize_NPI(df: pd.DataFrame):
    col = 'National Physician ID'
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    df[col] = df[col].fillna(-1)
    df[col] = df[col].apply(lambda x: int(x) if x != -1 else -1)
    df[col] = df[col].apply(lambda x: str(x) if x != -1 else 'UNKNOWN')
    return df 

def normalize_qbid(df: pd.DataFrame):
    col = 'Quickbase Record ID#'
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    df[col] = df[col].fillna(-1)
    df[col] = df[col].apply(lambda x: int(x) if x != -1 else -1)
    df[col] = df[col].apply(lambda x: str(x) if x != -1 else 'UNKNOWN')
    return df 

def process_biSAP_number(df):    # for weird SAP number: "123 \n abc".  split it into 2 rows
    df_1 = df.copy()
    df[  'SAP Number'     ] = df[  'SAP Number'     ].map(lambda x: x.split("\n")[0])
    df_1['SAP Number'     ] = df_1['SAP Number'     ].map(lambda x: x.split("\n")[1])
    df[  'SAP Entity Name'] = df[  'SAP Entity Name'].map(lambda x: x.split("\n")[0])
    df_1['SAP Entity Name'] = df_1['SAP Entity Name'].map(lambda x: x.split("\n")[1])
    return pd.concat([df, df_1]) 

def detect_high_missing_features(df: pd.DataFrame, missing_percentage_threshold: float = 30) -> tp.List[str]:
    """
    Identify features in dataframe that have a lots of missing values. 
    We use a percentage threshold to capture feature that has more than that and 
    list all of them. 
    Args:
        df (pd.DataFrame): input data as DataFrame
        missing_percentage_threshold (float, optional): threshold to identify a significant missing value feature. Defaults to 30.
    Returns:
        List: a list of significant missing value features
    """
    missing_percentage = df.isna().sum()/df.shape[0] * 100
    removed_missed_features = missing_percentage[missing_percentage > missing_percentage_threshold].index
    return removed_missed_features

def fill_missing_data(df, column):
    default = None
    type = dtype_dict[column]
    if type == 'datetime64[ns]':
        df[column] = pd.to_datetime(df[column])
        return df 
    if (type == str):        default = "UNKNOWN" 
    elif type == np.int32:        default = -1
    elif type == np.float64:        default = -1
    elif type == object:        default = "UNKNOWN"
    else:         default = "UNKNOWN"
    df[column] = df[column].fillna(default).astype(type)
    return df 

dtype_dict = {
    'Status': str,
    'Contact Type': str,
    'First Name': str,
    'Last Name': str,
    'Full Name': str,
    'HCP Category': str,
    'Payments Made To:': str,
    'SAP Number': str,
    'SAP Entity Name': str,
    'State/Region/Province': str,
    'Country': str,
    'National Physician ID': str,
    'Unnamed': np.float64,
    'Specialty': str,
    'Payment Currency':str,
    'Email Address': str,
    'Quickbase Record ID#': np.float64,
    'Reportable HCP': 'str',
    'National Physician ID/RPPS ID': str,
    'Primary Organization': str,
    'Primary Organization Type': str
}

#----------------- blocking --------------------

# blocking.  input = NPI, fullname, ...
def process_data(table, blocking_config: tp.Dict):
    # retrieve metadata for Magellan
    A = table.reset_index(drop=True).reset_index().rename(columns={"index": "id"}).drop_duplicates()
    B = A.copy()
    A['id'] = 'ltable_' + A['id'].astype(str)
    B['id'] = 'rtable_' + B['id'].astype(str)

    em.set_key(A, 'id')
    em.set_key(B, 'id')
    print(f'Number of rows {len(A)}')

    # level 0 blocking
    criteria = blocking_config.pop('level_0')
    cols = list(set(A.columns))
    ob = em.OverlapBlocker()

    K = ob.block_tables( A, B, criteria, criteria, l_output_attrs=cols, r_output_attrs=cols, 
        overlap_size=1, allow_missing=True
    )
    print(f'Number of pairs after level 0 blocking: {len(K)}')
    
    # level 1,2,... blocking
    levels = [f"level_{i}" for i in range(1, len(blocking_config))]
    for level in levels:
        criteria = blocking_config[level]
        K = ob.block_candset(K, criteria, criteria, overlap_size=1)
        print(f'Number of pairs after {level} blocking: {len(K)}')

    blocked_ids = pd.concat( [ K['ltable_id'], K['rtable_id'] ] ).unique()
    blocked_ids = [x.split('_')[1] for x in blocked_ids]
    final_table = table.loc[~table.index.isin(blocked_ids)]
    print(f'Final table size: {final_table.shape}')
    return final_table

#------------------- old school -----------------
def build_tabular_dataloader(df,  dep_var = 'n_trips'):
    cont,cat  = cont_cat_split(df, 20, dep_var=dep_var)
    np.random.seed(0)
    valid_idx = np.random.choice(df.shape[0], len(df)//10*3, replace=False)
    train_idx = range(valid_idx[-1], df.shape[0])
    splits    = (list(train_idx),list(valid_idx))
    procs     = [Categorify, FillMissing]
    to        = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)
    return to, cont, cat

#------------------- labeling -----------------

def process_group(df:pd.DataFrame) -> pd.DataFrame:
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
    new_df = pd.DataFrame(data=data, columns=cols)
    new_df['label'] = 1
    return new_df

def label_duplicate_data(df: pd.DataFrame, skewed_factor: int = 5) -> pd.DataFrame:
    """
    Try to label data by some conditions. The conditions we implement in this function are
       1) Full Name + Email Address
       2) National Physician ID 
       3) Full Name + SAP Number
       4) Full Name + Quickbase ID 
    Whenever a pair satisfies one of those conditions, its label will be assinged to 1. Otherwise, its label will be assigned to 0.
    Args:    df DataFrame: dataframe with label
    """
    df.rename(columns={'index': 'id'}, inplace=True)

    duplicate_dfs = []
    fullname_duplicate_df = df[(df['Email Address']!= 'UNKNOWN') & (df.duplicated(keep=False, subset=['Full Name', 'Email Address']))]
    fullname_duplicate_df = fullname_duplicate_df.groupby('Full Name').apply(process_group).reset_index(drop=True)
    if not fullname_duplicate_df.empty:  duplicate_dfs.append(fullname_duplicate_df)

    NPI_duplicate_df = df[(df['National Physician ID']!= 'UNKNOWN') & (df.duplicated(keep=False, subset=['National Physician ID']))]
    NPI_duplicate_df = NPI_duplicate_df.groupby('National Physician ID').apply(process_group).reset_index(drop=True)
    if not NPI_duplicate_df.empty:       duplicate_dfs.append(NPI_duplicate_df)
    
    SAP_duplicate_df = df[(df['SAP Number']!= 'UNKNOWN') & (df.duplicated(keep=False, subset=['Full Name', 'SAP Number']))]
    SAP_duplicate_df = SAP_duplicate_df.groupby('Full Name').apply(process_group).reset_index(drop=True)
    if not SAP_duplicate_df.empty:       duplicate_dfs.append(SAP_duplicate_df)
    
    QB_duplicate_df = df[(df['Quickbase Record ID#']!= 'UNKNOWN') & (df.duplicated(keep=False, subset=['Full Name', 'Quickbase Record ID#']))]
    QB_duplicate_df = QB_duplicate_df.groupby('Full Name').apply(process_group).reset_index(drop=True)
    if not QB_duplicate_df.empty:        duplicate_dfs.append(QB_duplicate_df)
    
    label_1_df = pd.concat(duplicate_dfs).drop_duplicates()
    label_1_df = label_1_df.rename(columns={'ltable_id': 'id'}).drop(columns=['rtable_id'])
    label_1_df['label'] = 1
    label_0_df = df[~df.id.isin(label_1_df.id.unique())]
    label_0_df = process_group(label_0_df)
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
    return data_df, label_1_df, label_0_df, fullname_duplicate_df, NPI_duplicate_df, SAP_duplicate_df, QB_duplicate_df
