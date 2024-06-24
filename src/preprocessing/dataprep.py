#==============================================================================
# intent: convert neo4j --> text files of pairs for training.
#         this clones and changes Lidia's DataPreprocessor's class.  Her version 
#         does not work in the Ditto environment due to the imports not working 
#         with Ditto's required library versions.
# notes : the main functions taken from Lidia's versions are: 
#         prepare_ditto_data(), build_matching_pairs(), build_nonmatching_pairs()
# vocab : daf = dataframe
#
# prepare_ditto_data()
# |--> build_matching_pairs()
# |    |--> pairs2daf()
# |    |    |--> process_int_cols()
# |--> build_nonmatching_pairs()
# |    |-- same as matching pairs ...
#==============================================================================

import pandas as pd, pdb, math, numpy as np
from neo4j import GraphDatabase;
from src.utils import auto_config as config          # for neo4j login
from src.DF_penumbra.utils import process_int_cols
from src.DF_penumbra import constants
from itertools import combinations                   # for build_non_matching_pairs()

# taken from dupsie/git/df-calvin-240415/src/dao/NEO4J_Graph -- Tu
class Graph:
    def __init__(self, url, username, password):
        self._url = url; self._username = username; self._password = password
    def cypher_transaction(self, cypher):            # Helper function that runs cypher transaction on local database
        driver = GraphDatabase.driver(self._url, auth=(self._username, self._password)); values = []
        with driver.session() as session:
            res = session.run(cypher)
            for record in res:  values.append(record.values())
        driver.close();  return values

# taken from ~/norm/dupsie/git/df-calvin-240415/src/DF_penumbra/data_preprocessor.py  -- lidia
class DataPreprocessor:
    def __init__(self, data_dir, args):
        self.graph = Graph( config.NEO4J_URL, config.NEO4J_USER, config.NEO4J_PASSWORD )
        self.data_dir = data_dir;  self.args = args
        #sn self.include_npi = include_npi                     

    def create_synoname_nodes( self ):
        add = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + "/synonyms.txt", "r", encoding="utf-8") as file:
            for line in file:
                columns = line.strip().split("\t")
                properties = {
                    "name"   : columns[0] + (", {}".format(columns[4]) if len(columns) == 5 else ""),
                    "origin" : columns[1],  "gender" : columns[2],
                    "meaning": columns[3] if len(columns) > 4 else None }
                q = """ CREATE (synonym:Synoname $properties) RETURN synonym """
                self.graph.cypher_transaction(q, {"properties": properties})
                add += 1
                print("Added {} Synoname nodes.".format(add))
        print("Note - use this cypher command to delete all Synoname nodes: MATCH (n:Synoname) DELETE n")

    #--------
    # intent: split node pairs list from a neo4j query into 3 tables, represented as dataframes.
    #         2d-list of neo4j nodes        ldaf       rdaf           bdaf
    #         +-  [a,b]  -+                a:---      b:---      a.uid, b.uid, 1
    #         |   [c,d]   |       --->     c:---      d:---      c.uid, d.uid, 1
    #         +-  [e,f]  -+                e:---      f:---      e.uid, f.uid, 1
    #         the output dafs is intended to feed into gel2ditto()                  
    # usage : A,B,C = pairs2daf( pairs, 1 ).  1 is the label for matching pairs
    # input : (pairs, labels).  pairs is the output of graph.cypher_transaction( '...match...' ).
    #         pairs is a 2d list.  ex -- shape of pairs = 5700 x 2.  each element is neo4j node.
    #         ex:  pairs = [ [a,b] [c,d] [e,f] [g,h] ], where each letter is a neo4j Node 
    #         (Pdb) type( pairs[0][0] )  --> <class 'neo4j.graph.Node'>
    #         this is a pair [a,b]:
    #         [<Node '4:3a1..f0a:658' labels=..'Provider' properties={'country': 'United States', 'uid': 'po_no_660', 'fname': 'Robert', 'lname': 'Ryan', 'npi': 1720366024 ,... 'qb_id': 5274}>
    #         ,<Node '4:3a1..f0a:660' labels=..'Provider' properties={'uid': 'po_no_662', 'fname': 'Roby', ... 'qb_id': 893}>]
    #
    # output: 3 dataframes.  left, right, and combined.  combined has columns (luid, ruid, label)
    #         ldaf = table (dataframe) for nodes a,c,e,g.  rdaf contains b,d,f,h.
    # vocabs: r = right.  l = left.  b = both r and l 
    # method: after converting nodes to dataframe, we still need to do some cleanup:
    #         convert char to numerics for id columns, replace 0 with nan, delete "embedding" column.
    #--------

    def _pairs2daf( self, pairs: list, label: int ):
        # drop columns:  text, embeddings.  convert int columns
        lrows = [ record[0] for record in pairs ];     ldaf = pd.DataFrame( lrows );
        rrows = [ record[1] for record in pairs ];     rdaf = pd.DataFrame( rrows );
        ldaf  = ldaf.rename( columns={'uid':'luid'} ) 
        rdaf  = rdaf.rename( columns={'uid':'ruid'} )
        # drop columns
        ldaf  = ldaf.drop( 'text', axis=1 ); ldaf = ldaf.drop( 'embedding', axis=1 );
        rdaf  = rdaf.drop( 'text', axis=1 ); rdaf = rdaf.drop( 'embedding', axis=1 );
        ldaf  = ldaf.drop(  'npi', axis=1 )  if self.args.npi == False else ldaf
        rdaf  = rdaf.drop(  'npi', axis=1 )  if self.args.npi == False else rdaf
        # int columns.  uid's, label
        ldaf  = process_int_cols( ldaf, constants.INT_COLS);  ldaf = ldaf.replace( 0, np.nan );
        rdaf  = process_int_cols( rdaf, constants.INT_COLS);  rdaf = rdaf.replace( 0, np.nan );
        brows = [ [ record[0]['uid'], record[1]['uid'], label ] for record in pairs ]; 
        bdaf  = pd.DataFrame( brows, columns=['luid','ruid','label'] )
        return ldaf, rdaf, bdaf

    # Label the pairs as matching or non-matching and prepare the training data. 
    # build_*() functions adds column prefix "xtable_"
    def prepare_ditto_data(self, skewed_factor):
        A1,B1,C1   = self._build_matching_pairs();
        limit      = 200 #sn math.floor( len(C1) * skewed_factor )
        A0,B0,C0   = self._build_non_matching_pairs( limit=limit )
        A1 = A1[ A0.columns ]; A = pd.concat( [A1, A0] );   # matchDaf    = matchDaf[ mismatchDaf.columns ]
        B1 = B1[ B0.columns ]; B = pd.concat( [B1, B0] );   # combinedDaf = pd.concat( [matchDaf, mismatchDaf], sort=False )
        C1 = C1[ C0.columns ]; C = pd.concat( [C1, C0] );  C['id'] = C.index 
        A.rename( inplace=True, columns=constants.RENAME_COLS );
        B.rename( inplace=True, columns=constants.RENAME_COLS );
        return A, B, C    

    def _build_matching_pairs(self):
        edge_types = ['r1_' + et for et in constants.EDGE_TYPES]
        query = f''' UNWIND {edge_types} AS type MATCH (a)-[r]->(b) WHERE type(r) = type AND a.uid <> b.uid RETURN DISTINCT a, b '''
        result = self.graph.cypher_transaction( query )
        A,B,C = self._pairs2daf( result, 1 )                                  #sn try _create_df_1s( result, label=1, side='l' )
        print(f'The number of match pairs:', len(C))
        return A,B,C

    #--------
    # intent:  create a list of mismatches
    # method:  get a list of names of known distinct people, and pair them up
    #          #1 get a list of names from distinct NPIs.  distinct NPI means distinct people.
    #          #2 create pairs from upper triangular coordinates:
    #          #3 plug database results into upper triangular matrix:
    #f miss-matches are nodes that have npi's but don't have the r1_npi relationship.  
    #  not having that relationship guarantees that their npi's are different.
    #e combinations() creates matrix element coordinates that marks a triangular 
    #  matrix -- no diagonals (node matches with itself), removes commutative pairs.
    #  these are the exact pairs we want.
    #  ex:  (Pdb) [ [i,j] for i,j in combinations( range(5), 2 ) ].  
    #       --> [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    #       "2" indicates pairs, instead of triplets, or quartets
    #g it's odd that we have to access the Node as result[i][0] insteatd of 
    #  result[i], but result[i] is a list of 1 element, the node itself.
    #--------

    def _build_non_matching_pairs(self, limit):  # For nodes that have different npis, create a non-matching pair.
        #1
        query  = f'''MATCH (n) WHERE NOT n:Master AND n.npi IS NOT NULL AND NOT EXISTS ((n)-[:r1_npi]-())  RETURN n limit {limit}''' #f
        result = self.graph.cypher_transaction( query )                       # result is a 2d-array with 2nd coord always 0
        result = [ result[i][0] for i in range(len(result)) ]                 # change to a 1d-array  #g
        #2
        coord  = [ [i,j] for i,j in combinations( range(len(result)), 2 ) ]   # coordinates of upper triangular matrix
        for k in range( len(coord) ):                                         # swap left & right coordinate, in case it matters.
            if k % 2 == 0:  coord[k][0], coord[k][1] = coord[k][1], coord[k][0]
        #3
        pairs  = [ [ result[ k[0] ], result[ k[1] ] ]   for k in coord ]
        A, B, C = self._pairs2daf( pairs, 0 )
        print(f'The number of mismatch pairs:', len(C))
        return A,B,C

    def splitDset( self, los, ratio ):                                        # split into train-valon-test
       chunk = math.floor( len(los) / ( ratio[0] + ratio[1] + ratio[2] ) )
       seg1b = 0      ; seg1e = seg1b + ratio[0] * chunk;  train = los[ seg1b : seg1e ]
       seg2b = seg1e+1; seg2e = seg2b + ratio[1] * chunk;  valid = los[ seg2b : seg2e ]
       seg3b = seg2e+1; seg3e = seg3b + ratio[2] * chunk;  test  = los[ seg3b : seg3e ]
       return train, valid, test

    #--------
    # intent:  convert data from Magellan format to Ditto
    # input :  3 table dataframes like this.  ("l" = left, "r" = right)
    #          ltable           |  rtable            |  data
    #            lc1  lc2  lid  |    rc1  rc2   rid  |    lc1  lc2   rc1  rc2   label id lid rid
    #          0  aa   bb    2  |  0  dd   ee     2  |  0  aa   bb    dd   ee       1  0   2   2
    #          1  gg   hh    3  |  1  jj   kk     3  |  1  gg   hh    jj   kk       0  1   3   3
    #          2  mm   nn    4  |  2  pp   qq     4  |  2  mm   nn    pp   qq       1  2   4   4
    # output:  output list of string like this                                                                  label
    #          .                               data.id                                            data.id         |
    #          .                             |----------|                                       |----------|      V
    #          col lc1 val aa col lc2 val bb col id val 0   \t   col rc1 val dd col rc2 val ee  col id val 0  \t  1
    #          col lc1 val gg col lc2 val hh col id val 1   \t   col rc1 val jj col rc2 val kk  col id val 1  \t  0
    #          col lc1 val mm col lc2 val mm col id val 2   \t   col rc1 val pp col rc2 val qq  col id val 2  \t  1
    # vocab : "gel" is Magellan, which is what the A,B,C dataframe format was based on.    
    #         dok, kod = domain knowledge.  one is the reverse format of the other. see #c
    #a list columns names of l/r table (exclude lid)
    #b fokn = foreign key name
    #--------
    #c from dok:  {'PERSON': ['fname', 'lname', 'fname'], 'ID': ['npi', 'sap_no', 'qb_id']}
    #  to   kod:  {'fname': 'PERSON', 'lname': 'PERSON', 'npi': 'ID', 'sap_no': 'ID', 'qb_id': 'ID'}
    def gel2ditto( self, ltable: pd.DataFrame, rtable: pd.DataFrame, data: pd.DataFrame,
        lfokn, rfokn, dok={}, c_data_id_name="id", c_label_name="label") -> list:       #b
        new_ltable = ltable.set_index(lfokn)
        new_rtable = rtable.set_index(rfokn)
        new_dk = {value: key.upper() for key, values in dok.items() for value in values} #c
        # ----
        def surround( col, value, kod ):      # surround "value" with tag given by "kod", eg [ID] 1234 [/ID].
            return "[" + kod[col] + "] " + value + " [/" + kod[col] + "]" if (col in kod) else value
        def formatted_string(row):      # formatted_string column
          lvalue  = []; rvalue = []
          l_rows  = new_ltable.loc[row[lfokn]]
          r_rows  = new_rtable.loc[row[rfokn]]
          l_row   = l_rows.iloc[0] if isinstance(l_rows, pd.DataFrame) else l_rows
          r_row   = r_rows.iloc[0] if isinstance(r_rows, pd.DataFrame) else r_rows
          lvalue0 = ["COL {} VAL {}".format( col, surround( col, l_row[col], new_dk) ) for col in  left_columns]
          rvalue0 = ["COL {} VAL {}".format( col, surround( col, r_row[col], new_dk) ) for col in right_columns]
          col_id  = f"COL id VAL {row[c_data_id_name]}"
          for segment in lvalue0:  lvalue.append( segment.replace('\t','').replace('\n','') )
          for segment in rvalue0:  rvalue.append( segment.replace('\t','').replace('\n','') )
          return f"{' '.join(lvalue)} {col_id} \t {' '.join(rvalue)} {col_id} \t {row[c_label_name]}"
        # ----
        left_columns  = list(ltable);   left_columns.remove(lfokn)  #a
        right_columns = list(rtable);  right_columns.remove(rfokn)
        df_combined   = data.copy()
        df_combined[ "formatted_string" ] = df_combined.apply( formatted_string, axis=1 )
        df_combined[ "formatted_string" ].to_csv("formatted_string.csv")
        return df_combined["formatted_string"].tolist()

"""
    I need to add an parameter "dk" 
      dk = { person: [ 'fname', 'lname', 'fname'], id = ['npi', 'sap_no', 'qb_id'], product = ['col1', 'col2',..] }
    the keys person/id/product can change and the contents of their list might change with each run.  
    With the dk paramete, gel2ditto() should behave as follows:
    
    for each element in dk['PERSON'] add the start and end tag as follows:  
        [PERSON] john [/PERSON]
    similarly for dk['ID']
    
    here is a more complete example:`
        col fname val john col lname val smith col fname john smith col country val usa col npi val 1234567890 col sap_no 38293839 col qbid val 3829
    add tas as follows:
        col fname val [PERSON] john [/PERSON] col lname val [PERSON] smith [/PERSON] col fname [PERSON] john smith [/PERSON] col country val usa 
        col npi val [ID] 1234567890 [/ID] col sap_no val [ID] 38293839 [/ID] col qbid val [ID] 3829 [/ID] 

# this is the version w/o --dok, eg. no surrounding [PERSON] tag
def gel2ditto( ltable: pd.DataFrame, rtable: pd.DataFrame, data: pd.DataFrame,
    lfokn, rfokn, c_data_id_name="id", c_label_name="label", ) -> list:       #b
    new_ltable = ltable.set_index(lfokn); new_rtable = rtable.set_index(rfokn)
    # ----
    def formatted_string(row):      # formatted_string column
      lvalue  = []; rvalue = []
      l_rows  = new_ltable.loc[row[lfokn]]
      r_rows  = new_rtable.loc[row[rfokn]]
      l_row   = l_rows.iloc[0] if isinstance(l_rows, pd.DataFrame) else l_rows
      r_row   = r_rows.iloc[0] if isinstance(r_rows, pd.DataFrame) else r_rows

      lvalue0 = [f"COL {col} VAL {l_row[col]}" for col in left_columns]
      rvalue0 = [f"COL {col} VAL {r_row[col]}" for col in right_columns]

      col_id  =  f"COL id VAL {row[c_data_id_name]}"
      for segment in lvalue0:  lvalue.append( segment.replace('\t','').replace('\n','') )
      for segment in rvalue0:  rvalue.append( segment.replace('\t','').replace('\n','') )
      return f"{' '.join(lvalue)} {col_id} \t {' '.join(rvalue)} {col_id} \t {row[c_label_name]}"
    # ----
    left_columns  = list(ltable);   left_columns.remove(lfokn)  #a
    right_columns = list(rtable);  right_columns.remove(rfokn)
    df_combined   = data.copy()
    df_combined[ "formatted_string"] = df_combined.apply( formatted_string, axis=1 )
    return df_combined["formatted_string"].tolist()
"""
