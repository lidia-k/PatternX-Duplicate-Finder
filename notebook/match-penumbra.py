#===============================================================================
# intent:  identify duplicate people in penumbra's spreadsheets
# input :  datafile = hard coded path to csv file
# output:  outfile  = hard coded path to output file, which contains a list of 
#          edges as a 3 tuple (xx,yy,##).  xx and yy are row id's from the spreadsheet
#          and ## is the confidence level that xx and yy are the same person
# assume:  we have a directory tree of code file as follows
#          repo_root
#          |-- src
#          |   |-- utils/:  penumbra.py, ...
#          |-- notebook/:   this-file.py
# method:  here are the steps:  read in data, explore data, clean npi/sap/qbid
#          block data, label data, build model, run model
#          Matching methods include Magellen, deepmatcher, old school (regression, 
#          decision tree, xgboost ... yawn)
# vocab :
#   qb    = quickbase, npi = national provider id, sap = penumbra's erp system
#   erp   = enterprise resource planning
#   block = to exclude pairs of rows that are unquestionably different people, 
#   .       so they don't need to undergo fuzzy matching
#   survive = pairs that passes through the blocking process.  these pairs 
#   .       MIGHT refer to the same people.  They will go through fuzzy matching
#===============================================================================

import pdb, io, sys, os, pandas as pd, numpy as np, matplotlib.pyplot as plt
sys.path.append('../'); #  sys.path.append('../src/lib/'); sys.path.append('../src/utils/penumbra/')
import typing as tp, seaborn as sns, msoffcrypto, py_entitymatching as em
from   itertools import combinations, permutations
from   fastai.tabular.all import *
from   sklearn.linear_model import LogisticRegression
from   sklearn.ensemble import GradientBoostingClassifier
from   sklearn.metrics import accuracy_score
from   sklearn.neighbors import KNeighborsClassifier
from   sklearn.tree import DecisionTreeClassifier
from   xgboost import XGBClassifier
import src.utils.auto_config as config, src.utils.penumbra as bra
# import deepmatcher as dm
pd.options.display.max_columns = None
"""
try:     import msoffcrypto
except:  !pip install msoffcrypto-tool
finally: import msoffcrypto

try:     import py_entitymatching as em
except:
   !pip  install setuptools wheel Cython fasttext-wheel
   !pip  install git+https://github.com/anhaidgroup/py_stringsimjoin.git@master
   !pip  install git+https://github.com/anhaidgroup/py_entitymatching.git@master
finally: import py_entitymatching as em
"""

#----------- read data --------------

data_dir = '/home/snguyen/norm/dupsie/df-calvin-data/'  # f'{os.path.dirname(os.getcwd())}/data/'
datafile = data_dir + "hcp-manz-sn.xlsx"
outfile  = data_dir + "edges.dat"
decrypted_workbook = io.BytesIO()
with open(datafile, 'rb') as file:
    office_file = msoffcrypto.OfficeFile(file)
    office_file.load_key(password=config.DATA_PASSWORD)
    office_file.decrypt(decrypted_workbook)
df_dict = {}
for sheet_name in pd.read_excel(decrypted_workbook, None).keys():
  df = pd.read_excel(decrypted_workbook, sheet_name=sheet_name)
  if sheet_name == '(20) France HCPs':
     df.rename(columns={'National Physician ID/RPPS ID': 'National Physician ID'}, inplace=True)
  name = sheet_name.split(') ')[1][:2] 
  df['uid'] = [f'{name}_{i+2}' for i in range(len(df))]
  df_dict[sheet_name] = df

#--------- explore data --------------

A, B = bra.build_data_pair(
    items_in_A=[ df_dict['(1000) Contacts'] ], 
    items_in_B=[ df_dict['(800) No SAP Number and Export '], df_dict['(340) US HCPs']
    ,            df_dict['(320) OUS HCPs'], df_dict['(20) France HCPs'] ] )
# print( A.shape, B.shape, A.head(2), B.head(2) )

def activate_debugging(A, B):            # hard code only 4 rows 
    A = A[A['Full Name'].isin(["Chirag Gandhi", "Chirag Gandi", "Aaron Bress", "John McGrath"])]
    B = B[B['Full Name'].isin(["Chirag Gandhi", "Chirag Gandi", "Aaron Bress", "Abdullah Shaikh"])]
    return A, B 
# A, B = activate_debugging(A, B)

#------------- clean npi, qbid ---------------

A = A.rename(columns={'Unnamed: 14': "Unnamed"})
B = B.rename(columns={'Unnamed: 15': "Unnamed"})       # print( A.head(), B.head() )
A = bra.normalize_qbid(A); B = bra.normalize_qbid(B)   # float-to-int.  eg.  2309.0 --> 2309
A = bra.normalize_NPI( A); B = bra.normalize_NPI( B)   # NaN --> "UNKNOWN"

removed_features_for_A = bra.detect_high_missing_features(A, missing_percentage_threshold=59.2)
removed_features_for_B = bra.detect_high_missing_features(B, missing_percentage_threshold=59.2)
removed_features = set(removed_features_for_A) & set(removed_features_for_B)  # print(removed_features)
A = A[ [ col for col in A.columns if col not in removed_features ] ]
B = B[ [ col for col in B.columns if col not in removed_features ] ]
common_columns = [ col for col in A.columns if col in B.columns ]
A = A[ common_columns ];  B = B[ common_columns ];     # print(common_columns)

missing_columns_for_A = bra.detect_high_missing_features( A, missing_percentage_threshold=0 )
missing_columns_for_B = bra.detect_high_missing_features( B, missing_percentage_threshold=0 )
for col in missing_columns_for_A:    A = bra.fill_missing_data(A, col)
for col in missing_columns_for_B:    B = bra.fill_missing_data(B, col)

#--------------- fix SAP ----------------

#sn:  added prefix "A_" and "B_" to "special_SAP_df.  this allows shuffling the next 6 lines of code for better readability

A_special_SAP_df = A[A['SAP Number'].astype(str).str.contains('\n')].groupby("SAP Number").apply(bra.process_biSAP_number).reset_index(drop = True)
B_special_SAP_df = B[B['SAP Number'].astype(str).str.contains('\n')].groupby("SAP Number").apply(bra.process_biSAP_number).reset_index(drop = True)
A = A[~A['SAP Number'].astype(str).str.contains('\n')]
B = B[~B['SAP Number'].astype(str).str.contains('\n')]
A = pd.concat([A, A_special_SAP_df])
B = pd.concat([B, B_special_SAP_df])
A["SAP Number"] = A["SAP Number"].astype(str) 
B["SAP Number"] = B["SAP Number"].astype(str) 
concat_table = pd.concat([A, B], ignore_index=True)

#----------------- block --------------------

pd.set_option('display.max_columns', None)  # This line ensures all columns are displayed
pd.set_option('display.width', 1000)  # Increase the width of each row to display more
pd.options.mode.copy_on_write = True
blocking_config = { "level_0": "National Physician ID", }   # "level_1": "Full Name"
new_A = bra.process_data(concat_table, blocking_config)     # print( new_A.head() )
new_A = new_A.reset_index()                                 # print( new_A.columns )

# new_A.columns are now:  Index(['index', 'Status', 'Contact Type', 'First Name', 'Last Name'
# ,   'Full Name', 'HCP Category', 'Payments Made To:', 'SAP Number', 'SAP Entity Name'
# ,   'State/Region/Province', 'Country', 'National Physician ID', 'Specialty', 'Payment Currency'
# ,   'Email Address', 'Quickbase Record ID#', 'uid'], dtype='object') 

#-------------- label ------------------

from datetime import datetime
print( 'start labeling:  ', datetime.now().strftime("%H:%M:%S") )
df, label_1_df, label_0_df, full_name_duplicate_df, NPI_duplicate_df, \
    SAP_duplicate_df, QB_duplicate_df =  bra.label_duplicate_data( new_A, skewed_factor=2 ) 
print( 'end   labeling:  ', datetime.now().strftime("%H:%M:%S") )

selected_columns = ['id','ltable_Full Name', 'rtable_Full Name',
    'ltable_National Physician ID', 'rtable_National Physician ID',
    'ltable_Email Address', 'rtable_Email Address', 'label']
sns.countplot(x = 'label', data = df)

# check the labels.
# print( df[df['ltable_Full Name'].isin(["Chirag Gandhi", "Chirag Gandi", "Aaron Bress"])][selected_columns] )
# print( df[selected_columns].head(10).reset_index(drop = True) )
# print( label_0_df[selected_columns].head(3).reset_index(drop = True) )
# print( label_1_df[selected_columns].head(10).reset_index(drop = True) )

#-------------- build model ------------------

del df['id']
pos_neg_ratio = np.sum(df['label'] == 1)/ np.sum(df['label'] == 0)
dm.data.split(df, data_dir, 'train.csv', 'valid.csv', 'test.csv',[3, 1, 1])
train, validation, test = dm.data.process( path=data_dir, cache='train_cache0.pth',
    train='train.csv', validation='valid.csv', test='test.csv', use_magellan_convention=True )
model = dm.MatchingModel(attr_summarizer='hybrid')
model.run_train(train, validation, epochs=3, batch_size=16, best_save_path=None, pos_neg_ratio=pos_neg_ratio)
model.run_eval(test)

#------------- use case --------------------

candidate = dm.data.process_unlabeled( path=os.path.join(data_dir, 'new_data.csv'),
    trained_model=model, ignore_columns=('ltable_id', 'rtable_id', 'label') )
predictions = model.run_prediction(candidate, output_attributes=list(candidate.get_raw_table().columns))
predictions = predictions.rename(columns={"match_score":"confident"})
selected_columns = [ 'ltable_Full Name', 'rtable_Full Name', 'ltable_National Physician ID',
 'rtable_National Physician ID', 'ltable_Email Address', 'rtable_Email Address',
 'confident', 'is_matched', 'label']
threshold = 0.8
predictions['is_matched'] = np.where(predictions['confident'] > threshold, 1, 0)
predictions.query("(is_matched == 0) & (label == 1) ")[selected_columns].tail(20).reset_index()

print( predictions.query("(is_matched == 1) & (label == 0) ")[selected_columns].head(20) )
print( predictions[selected_columns].head(20) )

#-------------- old school ------------------

def rf_feat_importance(m, features):
    return pd.DataFrame({'cols':features, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)

selected_features =[ f for f in  df.columns.tolist() if f!="id"]
regression_dataloader, cont, cat = bra.build_tabular_dataloader(df[selected_features], dep_var='label')
xs,y = regression_dataloader.train.xs, regression_dataloader.train.y
valid_xs,valid_y = regression_dataloader.valid.xs, regression_dataloader.valid.y

models = {
    'LogisticRegression': LogisticRegression(), 
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'XGBClassifier': XGBClassifier()
}

validation_results = {}
for model_name, model in models.items():
    model.fit(xs, y)
    y_pred = model.predict(valid_xs)
    score =accuracy_score(valid_y,y_pred)
    validation_results[model_name] = score

validation_results = {k:v for k,v in sorted(validation_results.items(), key = lambda item: item[1])}
ax = sns.barplot(x = list(validation_results.keys()), y = list(validation_results.values()))
ax.set(xlabel='Model', ylabel='accuracy')
for item in ax.get_xticklabels():
    item.set_rotation(90)

fi = rf_feat_importance(models['XGBClassifier'], models['XGBClassifier'].get_booster().feature_names)
fi.plot.barh(x='cols', y='imp', rot=0)
plt.show()


