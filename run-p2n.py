#==============================================================================
# usage :  > conda activate py37;  
#          > python37 thisfile.py --task ditto-pairs --project penumbra
#          > python37 thisfile.py --task train-ditto --load_ckp --save_model ---n_epoch 4 --size 32 --project penumbra
#          > python37 thisfile.py --task forward-L   --load_ckp                                     --project penumbra
#          > python37 thisfile.py --task forward-noL --load_ckp                                     --project penumbra
#          parameters to use in case of memory deficit
#          --maxlen    512.  the # of tokens in a pair
#          --batch_size 40.  the # of pairs to use in a single Forward and Backward pass.
#          --clipdaet 1000.  use only 1000 pairs in the epochs, even if dataset is much larger.  
#            this is especially relevant in Forward mode, when it has to store the entire dataset result w/o batching.
#          
# change:  dittoSN.py vs ditto.py.  load_state_dict(), don't use test.csv, print f1 properly
#          dataset0L vs dataset.py. run forward() on Lidia's test set.
# vocab :  yorf := yhat OR f1.  ditto.py::forwardSN() returns yhat if no labels are given, or f1 if they are
#          daet := data set
# to-do :  tokenize email, nan --> unk, state1 --> state
#          add tokens:  "npi", "quickbase", "sap"
#          move hard-coded dok dictionary to constants.py 
#          turn dok on/off by command line arg --dok.  currently --dok has no action.
# bitchy:  stuff that's wrong with Ditto.
#          confusing names: 
#          . <xx_set> means xx_path, 'size' means clip the data set, 'iter' means loader
#          . validation is called Test when printed
#          train & evaluate:
#          . best_f1 should not be decreasing, should not use Test set
#          hard to save & load model -- compare original train() with my 
#          . new ditto.py functions save_checkpoint() and load_model()
# changed:  run.py, 
# files     src/data/data_prepocessor.py moved DataPreprocessor into run-xxx.py
#           src/DG_penumbra/constants.py (RENAME_COLS), 
#           ditto_light/ditto.py (huggingface model/user_agent/token)
#           ditto_light/dataset.py.    size --> clipdaet, __init
#
# done  :  why does the f1 change with each run of the same prams???
#          learns wayyy to fast -- fishy.  len(trainset) is 16 !?!
#          tokenization of IDs ??
#          add domain knowledge
#          
# pip39 install langchain==0.1.15 fuzzywuzzy neo4j
# .env file:
#   NEO4J_URL='bolt://localhost:7687'
#   NEO4J_USER='neo4j'
#   NEO4J_PASSWORD='password'
# error: pandas has no attribute 'np'
# fix  : /home/snguyen/big/app/python3.9/lib/python3.9/site-packages/py_entitymatching/matcher/matcherutils.py
#        imp.statistics_[ pd.np.isnan(imp.statistics_) ] to
#        imp.statistics_[    np.isnan(imp.statistics_) ] 
#
# cpu vs gpu (for ditto):  in ditto_light/ditto.py "def train()"  change 
#        model = model.cuda() to
#        if device == 'cpu':  model = model.cpu()
#        else              :  model = model.gpu()
# "args" cannot be inspected directly.  don't know why.  to debug args:  (Pdb) parser.parse_args()
#==============================================================================

import argparse, joblib, sys, pdb, os, time, torch
sys.path.append(   '/home/snguyen/norm/dupsie/ditto' )
sys.path.insert(0, "/home/snguyen/norm/dupsie/apex") 
import numpy as np, pandas as pd, math               # math for floor() function
from sklearn.utils import shuffle
from src.DF_penumbra import constants;
import ditto_light.dataset   as dida     # normal training.  use labels
import ditto_light.ditto     as didi     # good load_state_dict(), skips test.txt, best_f1 is monotonic
import src.preprocessing.dataprep as dataprep
dadi = "src/data"  #sn

#==============================================================================
#                           helper functions
#==============================================================================

# intent: add a label to the end of each line of a file so we can use forwardSN()  
#         to iterate through the dataset and push x & y forward through the model.
#         this allows us to use forwardSN() with or without labels.
#         when label = False, the dummy lable will simply be ignored.
def add_dummy_label( infile=None, outfile=None, label=-1 ):
    with open( infile       ) as f:  lines = f.read().splitlines()
    with open( outfile, "w" ) as f:  f.writelines([ f"{i.rstrip()} \t {label}\n" for i in lines ])
    return outfile

# intent:  prepare a model and dataloader to be used in a forward pass
# input :  path to datafile to be converted into a pytorch DataLoader
def preForward( path ):
    model, optimizer, scheduler, epoch = didi.load_model( args, 10 )
    inDaet  = dida.DittoDataset( path=path, lm=args.lm, clipdaet=args.clipdaet )      #sn daet.pairs[1] is plain text
    dloader = torch.utils.data.DataLoader( dataset=inDaet , batch_size=args.batch_size  #sn was batch_size*16
    ,         shuffle=False, num_workers=0, collate_fn=inDaet.pad ) 
    return  model, dloader

# intent:  break a line containing a pair into 2 lines for easier comparison by a human
# input :  ex:  col fname val john ... \t col fname val johnathon ... \t 1   (line #4 in file)
# output:       4 1 col fname val john      ...
#               4 1 col fname val johnathan ...
def pair2single( infile, outfile ):
    label = ''; segments = []; lines = []; lostr = ''; i = 0
    with open( infile       ) as f:  lines = f.read().splitlines()
    for line in lines:
        segments = line.split('/t')
        label = segments[2] if len(segments) == 3 else ''
        lostr.append( str(i) + label + ' ' + segments[0].strip() ) 
        lostr.append( str(i) + label + ' ' + segments[1].strip() )
        i = i + 1
    with open( outfile, "w" ) as f:  f.writelines( [ s + '\n' for s in lostr ] )
    return outfile

# intent: convert a single node (not pair) in ditto format to csv
# input : file with lines like this
#         COL fname VAL [PERSON] Evans [/PERSON]  ... COL sap_no VAL [ID] 90358207 [/ID] COL id VAL 94
# output: csv file with lines like this
#         fname,.., sap_no, id
#         Evans,.., 90358207, 94
#a COL fname VAL john COL ... --> xxx = [ fname VAL john, lname VAL smith,..]
#b pop: remove "i -1" from list
def single2csv( infile, outfile ):
    cnames = []; values = []; xxx = []; body = []
    with open( infile ) as f:  lines = f.read().splitlines()

    xxx    = lines[0].split('COL ')                    #a
    cnames = [ s.split(' VAL ')[0]  for s in xxx ];
    cnames.pop(0);  body[0] = ",".join( cnames )       #b                              # header line

    for line in lines:
        xxx    = line.split('COL ')                         # xxx = [ fname VAL john, lname VAL smith,..]
        xxx.pop(0)                                          # remove "# -1"
        values = [ s.split(' VAL ')[1]  for s in xxx ];  cnames.pop(0);
        body.append( ','.join( values ) )
    with open("outfile", "w") as f:  f.writelines([ f"{i}\n" for i in body ])

#==============================================================================
#                                 args
#==============================================================================

if __name__ == '__main__':
    choices = ['adventureworks', 'penumbra']
    parser = argparse.ArgumentParser(description='Run different functions based on input parameters.')
    parser.add_argument('--project'   , choices=choices, type=str, help='The project to run')
    parser.add_argument("--task"      , type=str, default=None, help="task name:{train, predict, online_train}",  metavar='')
    parser.add_argument("--npi"       , action="store_true", help="Include NPIs for training (default: exclude NPIs)")
    parser.add_argument("--model"     , type=str, default=None, help="model name",  metavar='')
    parser.add_argument("--data"      , type=str, default=None, help="data file name",  metavar='')
    # ditto arguments:
    parser.add_argument("--run_id"    , type=int, default=0)
    parser.add_argument("--max_len"   , type=int, default=512)     #sn max_len is the length of a pair.  eg 256 integers/pair.
    parser.add_argument("--lr"        , type=float, default=3e-5)  #sn was 3e-5
    parser.add_argument("--n_epochs"  , type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir"    , type=str, default="checkpoints/")
    parser.add_argument("--lm"        , type=str, default='distilbert')
    parser.add_argument("--fp16"      , dest="fp16", action="store_true")
    parser.add_argument("--da"        , type=str, default=None)
    parser.add_argument("--alpha_aug" , type=float, default=0.8)
    parser.add_argument("--dok"       , type=str, default=None)    #sn was --dk
    parser.add_argument("--summarize" , dest="summarize", action="store_true")
    parser.add_argument("--size"      , type=int, default=256)     #sn superceded in some places by clipdaet.  i haven't found all occurences of "size" to replace with "clipdaet"
    parser.add_argument("--batch_size", type=int, default=40)      #sn was 512
    parser.add_argument("--clipdaet"  , type=int, default=1000)
    parser.add_argument("--load_ckp"  , type=str, default='checkpoint/model.pt', nargs='?', help="provide the path to xxx.pt file")
    parser.add_argument('--ckfile'    , type=str, default='model.pt', help='path to the model.pt file')
    args = parser.parse_args()

device    = 'cuda' if torch.cuda.is_available() else 'cpu'
trainpath = dadi + '/train.txt'; validpath = dadi + '/valid.txt'; 
testpath  = dadi +  '/test.txt'

#==============================================================================
#                              IF-ELSE switch
#==============================================================================

#m without labels, yorf means yhat
if args.task == 'forward-noL' and args.project == 'penumbra':                                # forward pass on unlabeled data
    inPath = add_dummy_label( dadi + '/manual-t2-ditto.txt', dadi + '/manual-t2-dummyL.txt'   ) 
    model  , dloader = preForward( inPath )
    yorf   , all_probs, all_y, yhat = didi.forwardSN( model, dloader, .95, label=False )    # yorf is yhat

#n with labels, yorf means f1
elif args.task == 'forward-L' and args.project == 'penumbra':                               # test on LABELED data
    model, dloader = preForward( dadi + '/van1k.txt' )
    yorf , all_probs, all_y, yhat = didi.forwardSN( model, dloader, .95, label=True )       #n
    print( f'run.py yorf = {yorf} \nyhat = {yhat}' )
    # count the # of 1's in yorf:  xx =[i for i in yorf if i == 1];  len(xx)

elif args.project == 'penumbra' and args.task == 'synoname': dp.create_synoname_nodes()

#o create ditto pairs
elif args.project == 'penumbra' and args.task == 'ditto-pairs':           #sn  added elif-ditto section
    if not( os.path.isfile ( trainpath ) and  os.path.isfile ( testpath ) and os.path.isfile ( validpath ) ):
        print( 'making vatt.txt data files ...' )
        def writelist( path, los ):
            with open( path, "w" ) as fh:
                for line in los: fh.write( f"{line}\n" )
        dp = dataprep.DataPreprocessor( dadi, args )                               # lidia's src/DF_penumbra/data_preprocessor.py
        ltable, rtable, data = dp.prepare_ditto_data( skewed_factor=.5 )  # fetch from database into dataframe
        start_time = time.strftime("%Y%m%d-%H%M%S");   print( "gel2ditto()  start = "  + start_time );
        dok                  = { "PERSON" : [ 'first name', 'last name', 'fname']
        ,                        "ID"     : ['national provider id', 'sap_no', 'Quickbase id'] }
        lostring             = dp.gel2ditto( ltable, rtable, data, 'luid', 'ruid', dok, 'id', 'label' )
        print( "gel2ditto()  end   = "  + time.strftime("%Y%m%d-%H%M%S") )
        lostring             = shuffle( lostring, random_state = 1 )
        train, valid, test   = dp.splitDset( lostring, [3,1,1] )
        writelist( trainpath, train ); writelist( validpath, valid ); writelist( testpath, test)

# copied code from train_ditto.py:
# sn name changes:      trainset --> trainpath.        train_dataset --> traindaet
elif args.project == 'penumbra' and args.task == 'train-ditto':    #sn  added elif-ditto section
    runtag    = '%s_lm=%s_da=%s_dk=%s_su=%s_clipdaet=%s_id=%d' % ( args.task, args.lm, 
        args.da, args.dk, args.summarize, str(args.clipdaet), args.run_id )
    runtag    = runtag.replace('/', '_')
    traindaet = dida.DittoDataset( trainpath, lm=args.lm, max_len=args.max_len, clipdaet=args.clipdaet, da=args.da )
    validdaet = dida.DittoDataset( validpath, lm=args.lm, clipdaet=args.clipdaet )   #sn validdaet.pairs[1] is plain text
    testdaet  = dida.DittoDataset( testpath , lm=args.lm, clipdaet=args.clipdaet )
    start_time = time.strftime("%Y%m%d-%H%M%S")
    print( "train start = "  + start_time )
    didi.train( traindaet, validdaet, testdaet, runtag, args )
    print( "train end   = "  + time.strftime("%Y%m%d-%H%M%S") )
else: print( 'did nothing' )

"""
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


#                              experiment log
# maxlen  size  batchsize epocs  lrate
#  256     256        512   100   3e-5  step: 0, loss: 0.6949014067649841   epoch 100: dev_f1=0.11764705882352941, f1=0.47619047619047616, best_f1=0.47619047619047616
#                     512         3e-4  step: 0, loss: 0.7937085628509521   epoch 100: dev_f1=0.18181818181818182, f1=0.5555555555555556, best_f1=0.5555555555555556
#                                 3e-3  step: 0, loss: 0.6904394030570984   epoch 100: dev_f1=0.11764705882352941, f1=0.47619047619047616, best_f1=0.47619047619047616
#                      64   4     3e-3  step: 0, loss: 0.6402472853660583   epoch   4: dev_f1=0.5116279069767442 , f1=0.5116279069767442 , best_f1=0.5116279069767442
#  512      16              8     3e-4  step: 0, loss: 0.6179225444793701     epoch 8: dev_f1=0.8571428571428571, f1=0.8571428571428571, best_f1=0.8571428571428571
#                                       python37 run-p2n.py --project penumbra --task train-ditto --n_epoch 8 --size 16 --lr 1e-4
#                                       .857 not reproducible -- f1 changes with each run.

"""
