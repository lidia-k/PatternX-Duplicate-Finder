# usage:  python39 run.py --project penumbra --task m_training
# pip39 install langchain==0.1.15 fuzzywuzzy neo4j
# .env file:
#   NEO4J_URL='bolt://localhost:7687'
#   NEO4J_USER='neo4j'
#   NEO4J_PASSWORD='password'
# error: pandas has no attribute 'np'
# fix  : /home/snguyen/big/app/python3.9/lib/python3.9/site-packages/py_entitymatching/matcher/matcherutils.py
#        imp.statistics_[ pd.np.isnan(imp.statistics_) ] to
#        imp.statistics_[    np.isnan(imp.statistics_) ] 


import argparse, joblib, pandas as pd, sys, pdb, os  #from DF_adventureworks.duplicate_finder import DuplicateFinder
sys.path.append( '/Users/lidia/PatternX/ditto' )
# ? sys.path.append( "/home/snguyen/norm/dupsie/deepmatcher" )
sys.path.insert(0, '/Users/lidia/PatternX/ditto/apex') 
import numpy as np, pandas as pd, math               # math for floor() function
from sklearn.utils import shuffle
from src.DF_penumbra import constants
from src.DF_penumbra.utils import process_int_cols
from src.utils import auto_config as config          # for neo4j login
from neo4j import GraphDatabase
from itertools import combinations                   # for build_non_matching_pairs()
import ditto_light.dataset as dida, ditto_light.ditto as didi
dadi = "src/data"  #sm

class Graph:                 # taken from dupsie/git/df-calvin-240415/src/dao/NEO4J_Graph -- Tu
    def __init__(self, url, username, password):
        self._url = url; self._username = username; self._password = password
    def cypher_transaction(self, cypher):             # Helper function that runs cypher transaction on local database
        driver = GraphDatabase.driver(self._url, auth=(self._username, self._password)); values = []
        with driver.session() as session:
            res = session.run(cypher)
            for record in res:  values.append(record.values())
        driver.close();  return values

class DataPreprocessor:      # taken from ~/norm/dupsie/git/df-calvin-240415/src/DF_penumbra/data_preprocessor.py  -- lidia
    def __init__(self, data_dir):
        self.graph = Graph( config.NEO4J_URL, config.NEO4J_USER, config.NEO4J_PASSWORD )
        self.data_dir = data_dir
        #sn self.include_npi = include_npi                     
    """def _create_df_expire(self, result, label):
        node_pairs = [(record[0], record[1]) for record in result];  all_props = set();
        rows = []
        for node_a, node_b in node_pairs:
            all_props.update(dict(node_a).keys())
            all_props.update(dict(node_b).keys())
        for node_a, node_b in node_pairs:
            row = {}
            for prop in all_props:  row[f'ltable_{prop}'] = node_a.get(prop, np.nan)
            for prop in all_props:  row[f'rtable_{prop}'] = node_b.get(prop, np.nan)
            rows.append(row)
        df = pd.DataFrame(rows)
        int_cols = ['ltable_' + col for col in constants.INT_COLS] + ['rtable_' + col for col in constants.INT_COLS]
        df = process_int_cols(df, int_cols)
        df.replace(0, np.nan, inplace=True)
        df['label'] = label
        return df """
    def _create_df(self, result, label):
        # each record of "result" is a node pair.  that is result[1] is a node pair.
        node_pairs = [(record[0], record[1]) for record in result];  all_props = set();
        rows = []
        for node_a, node_b in node_pairs:
            all_props.update(dict(node_a).keys())
            all_props.update(dict(node_b).keys())
        for node_a, node_b in node_pairs:
            row = {}
            for prop in all_props:  row[f'ltable_{prop}'] = node_a.get(prop, np.nan)
            for prop in all_props:  row[f'rtable_{prop}'] = node_b.get(prop, np.nan)
            rows.append(row)
        df = pd.DataFrame(rows)
        int_cols = ['ltable_' + col for col in constants.INT_COLS] + ['rtable_' + col for col in constants.INT_COLS]
        df = process_int_cols(df, int_cols)
        df.replace(0, np.nan, inplace=True)
        df['label'] = label
        return df
    def _create_df_1s(self, result, label, side ):  # sides = (l) left, (r) right, (b) both
        #a int_cols = ['ltable_npi', 'ltable_qb_id', 'ltable_sap_no', 'ltable_license', 'rtable_npi', 'rtable_qb_id', 'rtable_sap_no', 'rtable_license']
        node_pairs = [(record[0], record[1]) for record in result];  all_props = set(); rows = []
        prefix = 'ltable' if side=='l' else 'rtable'
        for node_a, node_b in node_pairs:           # loop rows to form a Union of all column names
            all_props.update(dict(node_a).keys())
            all_props.update(dict(node_b).keys())
        for node_a, node_b in node_pairs:           # create table w/ "ltable" or "rtable" column names
            row = {}
            for prop in all_props:  row[f'{prefix}_{prop}'] = node_a.get(prop, np.nan)
            rows.append(row)
        df = pd.DataFrame(rows)
        int_cols = [prefix + '_' + col for col in constants.INT_COLS] # + ['rtable_' + col for col in constants.INT_COLS] #a

        df = process_int_cols( df, int_cols )
        df.replace(0, np.nan, inplace=True)
        df['label'] = label
        return df
    # build_*() functions adds column prefix "xtable_"
    def prepare_ditto_data(self, skewed_factor=5):  # Label the pairs as matching or non-matching and prepare the training data. 
        matchDaf    = self._build_matching_pairs()
        limit       = len( matchDaf ) * skewed_factor
        mismatchDaf,_ = self._build_non_matching_pairs( limit=limit )
        matchDaf    = matchDaf[ mismatchDaf.columns ]
        combinedDaf = pd.concat( [matchDaf, mismatchDaf], sort=False )
        A, B, combinedDaf = self._split_tables( combinedDaf )  # combinedDaf gets 3 id columns in split_table()
        return A, B, combinedDaf    
    def _build_matching_pairs(self):
        edge_types = ['r1_' + et for et in constants.EDGE_TYPES]
        query = f''' UNWIND {edge_types} AS type MATCH (a)-[r]->(b) WHERE type(r) = type AND a.uid <> b.uid RETURN DISTINCT a, b '''
        result = self.graph.cypher_transaction( query )
        df = self._create_df( result, label=1 )  #sn try _create_df_1s( result, label=1, side='l' )
        print(f'The number of matching pairs:', len(df))
        return df
    def _build_non_matching_pairs(self, limit):  # For nodes that have different npis, create a non-matching pair.
        query = f'''MATCH (n) WHERE NOT n:Master AND n.npi IS NOT NULL AND NOT EXISTS ((n)-[:r1_npi]-())  RETURN n'''
        result = self.graph.cypher_transaction( query )
        df = pd.DataFrame([dict(record[0]) for record in result])
        df.drop( columns=['npi'], inplace=True )  #sn was:  if not self.include_npi: df.drop(...)
        pairs = [(df.iloc[i], df.iloc[j]) for i, j in combinations(range(len(df)), 2)]
        paired_data = []
        for left, right in pairs:
            left_dict = {'ltable_' + col: val for col, val in left.items()}
            right_dict = {'rtable_' + col: val for col, val in right.items()}
            paired_data.append({**left_dict, **right_dict})
        paired_df = pd.DataFrame(paired_data)
        int_cols = ['ltable_' + col for col in constants.INT_COLS] + ['rtable_' + col for col in constants.INT_COLS]
        paired_df = process_int_cols(paired_df, int_cols)
        paired_df.replace(0, np.nan, inplace=True)
        sample_df = paired_df.sample(n=limit, random_state=1)
        sample_df['label'] = 0
        dropped_df = paired_df.drop(sample_df.index)
        dropped_df.to_csv('dropped.csv', index=False)
        print(f'The number of non-matching pairs:', len(sample_df))
        return sample_df, dropped_df

    # input : dataframe like this.  
    #               ltable_c1    ltable_c2    rtable_c1    rtable_c2
    #          1          bob        jones         mary       smith
    #          2          
    # C is the same as input, except: shuffled rows, 3 new columns (id, rtable_id, ltable_id) all of which is numbering the row. 
    # A
    def _split_tables(self, df):                 # Split the data into A, B, and C tables.
        df = shuffle(df, random_state=1).reset_index(drop=True)
        df['id'] = df.index
        df['ltable_id'] = df['id']
        df['rtable_id'] = df['id']
        #combined_df = combined_df.loc[:, ['label', 'ltable_uid', 'rtable_uid', 'ltable_npi', 'rtable_npi', 'ltable_fullname', 'rtable_fullname']]
        df.to_csv('C.csv', index=False)
        ltable_cols = [col for col in df.columns if 'ltable_' in col]
        rtable_cols = [col for col in df.columns if 'rtable_' in col]
        A = df[ltable_cols]
        B = df[rtable_cols]
        A.columns = [col.replace('ltable_', '') for col in ltable_cols]
        B.columns = [col.replace('rtable_', '') for col in rtable_cols]
        A = A.rename(columns={'id': 'ltable_id'})
        B = B.rename(columns={'id': 'rtable_id'})
        A.to_csv('A.csv', index=False)
        B.to_csv('B.csv', index=False)
        return A, B, df

def splitDset( los, ratio ):               # split into train-valon-test
   chunk = math.floor( len(los) / ( ratio[0] + ratio[1] + ratio[2] ) )
   seg1b = 0      ; seg1e = seg1b + ratio[0] * chunk;  train = los[ seg1b : seg1e ]
   seg2b = seg1e+1; seg2e = seg2b + ratio[1] * chunk;  valid = los[ seg2b : seg2e ]
   seg3b = seg2e+1; seg3e = seg3b + ratio[2] * chunk;  test  = los[ seg3b : seg3e ]
   return train, valid, test

#a list columns names of l/r table (exclude lid)
def gel2ditto( ltable: pd.DataFrame, rtable: pd.DataFrame, data: pd.DataFrame,
    lfokn, rfokn, c_data_id_name="id", c_label_name="label", ) -> list:
    #
    def formatted_string(row):      # formatted_string column
      lvalue = []; rvalue = []
      lvalue0 = [ f"COL {col} VAL {ltable.loc[ltable[lfokn] == row[lfokn], col].iloc[0]}" for col in left_columns  ]
      rvalue0 = [ f"COL {col} VAL {rtable.loc[rtable[rfokn] == row[rfokn], col].iloc[0]}" for col in right_columns ]
      col_id =   f"COL id VAL {row[c_data_id_name]}"
      for segment in lvalue0:  lvalue.append( segment.replace('\t','').replace('\n','') )
      for segment in rvalue0:  rvalue.append( segment.replace('\t','').replace('\n','') )
      return f"{' '.join(lvalue)} {col_id} \t {' '.join(rvalue)} {col_id} \t {row[c_label_name]}"
    #

    left_columns  = list(ltable);   left_columns.remove(lfokn)  #a
    right_columns = list(rtable);  right_columns.remove(rfokn)
    df_combined   = data.copy()
    df_combined[ "formatted_string"] = df_combined.apply( formatted_string, axis=1 )
    return df_combined["formatted_string"].tolist()

#----------------------  main -------------------------

if __name__ == '__main__':
    choices = ['adventureworks', 'penumbra']
    parser = argparse.ArgumentParser(description='Run different functions based on input parameters.')
    parser.add_argument('--project', choices=choices, type=str, help='The project to run')
    parser.add_argument("--task", type=str, default=None, help="task name:{train, predict, online_train}",  metavar='')
#    parser.add_argument("--npi", action="store_true", help="Include NPIs for training (default: exclude NPIs)")
    parser.add_argument("--model", type=str, default=None, help="model name",  metavar='')
    parser.add_argument("--data", type=str, default=None, help="data file name",  metavar='')
    # ditto arguments:
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    args = parser.parse_args()

# ditto must run under "conda activate py37; python37 run.py --project penumbra --task train-ditto"
if args.project == 'penumbra' and args.task == 'train-ditto':    #sn  added elif-ditto section
    trainpath = dadi + '/train.csv'; validpath = dadi + '/valid.csv'; testpath = dadi + '/test.csv'
    if not( os.path.isfile ( trainpath ) and  os.path.isfile ( testpath ) and os.path.isfile ( validpath ) ):
        print( 'training ditto ...' )
        def writelist( path, los ):
            with open( path, "w" ) as fh:
                for line in los: fh.write( f"{line}\n" )
        dp = DataPreprocessor( dadi )              # lidia's src/DF_penumbra/data_preprocessor.py
        ltable, rtable, data = dp.prepare_ditto_data( skewed_factor=2 )  # fetch from database into dataframe
        lostring             = gel2ditto( ltable, rtable, data, 'ltable_id', 'rtable_id' )         # 
        train, valid, test   = splitDset( lostring, [3,1,1] )
        writelist( trainpath, train ); writelist( validpath, valid ); writelist( testpath, test)

    # copy code from train_ditto.py:
    # sn name changes:      trainset --> trainpath.        train_dataset --> trainds

    else:  # file exist.  open, read, train
        runtag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % ( args.task, args.lm, args.da,
            args.dk, args.summarize, str(args.size), args.run_id )
        runtag = runtag.replace('/', '_')

        trainds = dida.DittoDataset( trainpath, runtag, args ) #c, ditto_light/dataset.py
        validds = dida.DittoDataset( validpath, lm=args.lm)
        testds  = dida.DittoDataset( testpath , lm=args.lm)
        didi.train( trainds, validds, testds, runtag, args )                                         # ditto/ditto.py





