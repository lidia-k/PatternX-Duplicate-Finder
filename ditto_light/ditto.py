# this version:       classifier by unfreezing last bert layer and 2-neuron head
# https://www.kaggle.com/code/masayakondo/tutorial-bert-classifier-using-pytorch#Declare-Classifier-Model
# get max sentence length:
#   length = datasets['text'].map(tokenizer.encode).map(len)
#   print(max(length))
#   sns.distplot(length)
# initialize weight:
#   self.linear = nn.Linear(768, 2)
#   nn.init.normal_(self.linear.weight, std=0.02)
#   nn.init.normal_(self.linear.bias, 0)
# fine tune settings:  see this file.  set_finetune_layers()

#   train_ditto.py
#   |
#   |-- train()
#   |   |-- loop epochs
#   |   |   |-- train_step( det )
#   |   |   |    |-- loop batches:
#   |   |   |    |   |      model(x)
#   |   |   |    |   |      forward()      xEntropy()        backward()
#   |   |   |    |   |   x ----------> y^ -----------> loss -----------> gradient
#   |   |   |    |   |
#   |   |   |-- evaluate( vandet );  evaluate( testdet )
#   |   |   |    |-- loop batches:
#   |   |   |    |   |                                                        y--+
#   |   |   |    |   |      forwardSN()         smax()           p > th          |
#   |   |   |    |   |   x ----------> logits ---------> probs ---------> 0/1 ------> f1
#   |   |   |    |   |       model()                           concat()        txr(),
#   |   |   |-- new best?  mark best_f1, save checkpoint                       f1s()
#   <---------------+
#
# show weights:  (pdb) model.bert.transformer.layer[0].attention.q_lin.weight
#
# vocab:
# yora = y or accuracy   acc = accuracy
# yorf = y or f1         txr = true (+),(-) rate
#
#l must convert to int16, or the default float will error "Expected tensor 
#  for argument #1 'indices' to have one of the following scalar types: Long, Int"
#
# tried to use microsoft phi-3
#   tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
#   model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)

import os, pdb, sys, pandas, torch, random, numpy as np, argparse
import torch.backends
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import sklearn.metrics as metrics
from .dataset import DittoDataset  #sn was DittoDataset
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from tensorboardX import SummaryWriter
from apex import amp

lm_mp = {'roberta': 'roberta-base', 'phi3':'microsoft/Phi-3-mini-4k-instruct',   #sn added Phi-3
         'distilbert': 'distilbert-base-uncased'}

#---------------------------------------
class DittoModel(nn.Module):
#---------------------------------------
    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:  self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:            self.bert = AutoModel.from_pretrained(lm)
        self.alpha_aug = alpha_aug
        self.device  = device
        hidden_size  = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(0.2)
        self.fc1     = torch.nn.Linear( hidden_size, 20 )
        self.fc      = torch.nn.Linear( 20, 2 )

#g https://buomsoo-kim.github.io/attention/2020/04/22/Attention-mechanism-20.md/
#  "the output of encoder has to match the target [fc]" 
#     enc.view[-1,768], or maybe enc = enc.reshape(x.shape[1], -1)
#  depending on memory block contiguity, reshape() MAY copy then shape while 
#  view() does not copy.
    def forward( self, x1, x2=None ):
        x1 = x1.to(self.device)     # (batch_size, seq_len)
        if x2 is not None:          # MixDA (Data Augmentation)
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)
            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :] #sn classification uses only the first vector, hence [:, 0, :] 
        # enc = enc.view[-1,768]          #g #sn new.  not sure if this is necessary
        xx  = torch.nn.functional.relu( self.dropout( self.fc1(enc) ) )
        out  = self.fc( xx )
        return out       #sn was:    return self.fc(enc) # .squeeze() # .sigmoid()

    def set_finetune_layers( self ):
      bertLL = self.bert.transformer.layer[-1]  # bertLL = self.bert.encoder.layer[-1]  # bert last layer
      for p in     self.parameters():  p.requires_grad = False # turn off all gradients
      for p in self.fc1.parameters():  p.requires_grad = True  # turn on fc1
      for p in  self.fc.parameters():  p.requires_grad = True  # turn on fc
      for p in   bertLL.parameters():  p.requires_grad = True  # turn on near last bert layer

# intent: calculate f1 over a dataset
# input : iterator  = the van or test dataset
#         threshold = the probability mark at which we accept a label.
#            for example, for label 0, if threshold = .6 and prob = .55, 
#            then we call the prediction "1"
# output; best_f1 for all batches in the dataset, and the best threshold among 0/.5/1
#a dim=1 is horizontal softmax.  [:,1] means take the 2nd column -- coordinates ( __ ,1)
#b list1 += list2 means concatenate the lists.

def best_f1_threshold( all_probs, all_y ):              # try a bunch of thresholds & take the f1-best one of the bunch
  f1 = 0.0
  for th in np.arange(0, 1, .05):
    yhat      = [1 if p > th else 0 for p in all_probs]
    new_f1    = metrics.f1_score( all_y, yhat )
    if new_f1 > f1:
      f1      = new_f1
      best_th = th
  return f1, best_th, yhat

# intent: check accuracy of net being trained so far
def evaluate( model, dloader, threshold=None ):
    all_p = [];  all_y = [];  all_probs = [];  best_th = 0.5; f1 = 0.0
    model.eval()
    all_probs, all_y, yhat, yora = forwardSN( model, dloader, threshold, label=True )
    if threshold is     None: f1, threshold, yhat = best_f1_threshold( all_probs, all_y )  #sn functionized best_f1_threshold
    if threshold is not None: 
        yhat       = [1 if p > threshold else 0 for p in all_probs] # yhat is self-accumulating
        rates = txr( all_y, yhat )                                  # was (f1):  tp, fp, tn, fn, f1 = f1s( all_y, yhat )
    print( f'ditto.py evaluate(). \n ytru = {all_y} \n yhat = {yhat} ' )
    return rates, threshold, yhat                                   # was (f1):  return f1, threshold, yhat

# intent: calculate true/false (+) (-)
def f1s(ytru, yhat):                                        # calculate accuracies
    tp = 0; fp = 0; tn = 0; fn = 0
    for i in range(len(yhat)): 
        if ytru[i] == yhat[i] == 1            : tp += 1
        if ytru[i] == yhat[i] == 0            : tn += 1
        if yhat[i] == 0 and ytru[i] != yhat[i]: fn += 1
        if yhat[i] == 1 and ytru[i] != yhat[i]: fp += 1
    f1 = 2*tp / ( 2*tp + fp +fn )
    return tp, fp, tn, fn, f1

# intent: calculate true (+),(-) rates
#
#             actual (+)                          actual (-)
# |----------------------------------| |----------------------------------| 
# |--------------------||------------| |------------||--------------------| 
#       guessed (+)       guessed (-)    guessed (-)       guessed (+)
#          true (+)         false (-)       true (-)         false (+)

def txr(ytru, yhat):                                        # calculate tpr and tnr
    tp = 0; tn = 0; tpL = []; tnL = []; fpL = []; fnL = [];
    for i in range(len(ytru)):
        xx = fpL.append(i) if (ytru[i] == 0 and yhat[i] == 1) else None
        yy = fnL.append(i) if (ytru[i] == 1 and yhat[i] == 0) else None
    for i in range(len(ytru)):
        if ytru[i] == yhat[i] and yhat[i] == 1: tpL.append(1)           # failed: tpL = [ i for i in ytru if (ytru[i] == yhat[i] == 1) ]; tp   = len(tpL)
        if ytru[i] == yhat[i] and yhat[i] == 0: tnL.append(0)           # failed: tnL = [ i for i in ytru if (ytru[i] == yhat[i] == 0) ]; tn   = len(tnL)
    tp = len(tpL);  tn = len(tnL);
    pos = [ i for i in ytru if i == 1 ];  posK = len(pos);  tpr = tp / posK if posK >= 1 else -1
    neg = [ i for i in ytru if i == 0 ];  negK = len(neg);  tnr = tn / negK if negK >= 1 else -1
    rates = { 'tpr':tpr, 'tp':tp, 'posK':posK, 'tnr':tnr, 'tn':tn, 'negK':negK, 'fpL':fpL, 'fnL':fnL }
    return rates

# intent: push data through a forward pass
#b threshold provided so we can calculate yhat. yhat is self-accumulating
#c label     provided so we can calculate f1.   yora means yhat or accuracy
def forwardSN( model, train_iter, threshold=None, label=True ):  #sn: new section.  calculate f1 for train set
    all_y = [];  all_probs = [];  yhat = []
    with torch.no_grad():
      for i, batch in enumerate( train_iter ):
        x, y       = batch
        all_y     += y.tolist()
        logits     = model(x)
        probs      = logits.softmax(dim=1)[:, 1]        #a        probs      = model(x).softmax( dim=1 )[:,1]
        all_probs += probs.tolist()
    if threshold == None:  return all_probs, all_y  # return to evaluate().  cannot calculate yhat and f1
    yhat = [1 if p > threshold else 0 for p in all_probs] #b
    yora = txr( all_y, yhat ) if label == True else yhat  #c  # was (f1):   yorf = f1s( all_y, yhat ) if label == True else yhat  #c
    print( f'ditto.py forward(). yora = {yora}' )                       # was (f1):   print( f'ditto.py. yorf = {yorf}' )
    return all_probs, all_y, yhat, yora

# intent: train 1 epoch
# inputs: train_iter (Iterator): training dataset.
#         model (DMModel): the model
#         optimizer (Optimizer): the optimizer (Adam or AdamW)
#         scheduler (LRScheduler): learning rate scheduler
#         hp (Namespace): other hyper-parameters (e.g., fp16)
# output: none.  changes made to the input "model"

def train_step(train_iter, model, optimizer, scheduler, hp ):  #sn added threshold for debugging.  delete threshold when done
    criterion = nn.CrossEntropyLoss()       # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):  # loop batch through the dataset
        model.zero_grad()                   # this was missing from Ditto!
        optimizer.zero_grad()
        if len(batch) == 2:
            x, y = batch
            prediction = model(x)
        else:
            x1, x2, y = batch
            prediction = model(x1, x2)
        loss = criterion(prediction, y.to(model.device))
        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:   loss.backward()
        optimizer.step()                    # nudge weights
        scheduler.step()
        print(f"ditto.py.  batch: {i}, loss: {round(loss.item(),5)}")  #sn  was:   if i % 10 == 0:    print(f"step: {i}, loss: {loss.item()}")  # monitoring
        del loss

# intent: loop each epoch and train
# input : 3 vatt datasets,  run_tag (str): the tag of the run
#        hp (Namespace): Hyper-parameters (e.g., batch_size,  learning rate, fp16)
# output:  None
def train(trainset, validset, testset, run_tag, hp):
    best_van_acc = 0.0; th = .5
    train_iter = data.DataLoader(dataset=trainset, batch_size=hp.batch_size   , shuffle=True , num_workers=0, collate_fn=trainset.pad)
    valid_iter = data.DataLoader(dataset=validset, batch_size=hp.batch_size*16, shuffle=False, num_workers=0, collate_fn=trainset.pad)
    test_iter  = data.DataLoader(dataset=testset , batch_size=hp.batch_size*16, shuffle=False, num_workers=0, collate_fn=trainset.pad)
    writer     = SummaryWriter( log_dir=hp.logdir )           # log with tensorboardX
    num_steps  = (len(trainset) // hp.batch_size) * hp.n_epochs
    print("num_steps", num_steps)
    model, optimizer, scheduler, epoch = load_model( hp, num_steps )  #sn:  the body of load_model() was here.  i removed and functionized it.

#   if hp.train-tokens == 1:
#   # The pre-learned sections should have a smaller learning rate, and the last total combined layer should be larger.
#import torch.optim as optim
#optimizer = optim.Adam([
#    {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
#    {'params': classifier.linear.parameters(), 'lr': 1e-4} ])

    #e before f1-->acc change:  van_f1,th,yhat = evaluate();  if van_f1 > best_van_f1: best_van_f1 = van_f1
    #f in original ditto.py using f1, we mark the dev f1, but print best_test_f1 even though it's worse.
    for epoch in range( epoch+1, epoch + hp.n_epochs + 1):
        model.train()
        train_step( train_iter, model, optimizer, scheduler, hp )
        model.eval()
        acc, th, yhat = evaluate( model, valid_iter, .95 )     #e
        van_acc = ( acc['tpr'] + acc['tnr'] ) / 2              #e
        if van_acc        > best_van_acc:                      #e #f
            best_van_acc  = van_acc                            #e
            if hp.save_model:  save_checkpoint( hp, model, optimizer, scheduler, epoch )
        print( f"epoch {epoch}: van_acc={round(van_acc,4)} threshold={round(th,2)}, best_van_acc={round(best_van_acc,4)}\n" )
        writer.add_scalars( run_tag, {'f1': van_acc}, epoch )  # log
    writer.close()

def save_checkpoint( hp, model, optimizer, scheduler, epoch ):
  if not os.path.exists( hp.logdir ):   os.makedirs( hp.logdir )
  ckpt_path = os.path.join(hp.logdir, hp.ckfile)
  ckpt = { 'model': model.state_dict(),     'optimizer': optimizer.state_dict(),
       'scheduler': scheduler.state_dict(), 'epoch'    : epoch }
  torch.save(ckpt, ckpt_path)

#j https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
#  must have strict=False or else error:
#  *** RuntimeError: Error(s) in loading state_dict for DittoModel:
#  Missing key(s) in state_dict: "bert.embeddings.word_embeddings.weight", "bert.embeddings.position_embeddings.weight", "bert.embeddings.LayerNorm.weight", "bert.embeddings.LayerNorm.bias", "bert.transformer.layer.0.attention.q_lin.weight", "bert.transformer.layer.0.attention.q_lin.bias", "bert.transformer.layer.0.attention.k_lin.weight", "bert.transformer.layer.0.attention.k_lin.bias", "bert.transformer.layer.0.attention.v_lin.weight", "bert.transformer.layer.0.attention.v_lin.bias", "bert.transformer.layer.0.attention.out_lin.weight", "bert.transformer.layer.0.attention.out_lin.bias", "bert.transformer.layer.0.sa_layer_norm.weight", "bert.transformer.layer.0.sa_layer_norm.bias", "bert.transformer.layer.0.ffn.lin1.weight", "bert.transformer.layer.0.ffn.lin1.bias", "bert.transformer.layer.0.ffn.lin2.weight", "bert.transformer.layer.0.ffn.lin2.bias", "bert.transformer.layer.0.output_layer_norm.weight", "bert.transformer.layer.0.output_layer_norm.bias", "bert.transformer.layer.1.attention.q_lin.weight", "bert.transformer.layer.1.attention.q_lin.bias", "bert.transformer.layer.1.attention.k_lin.weight", "bert.transformer.layer.1.attention.k_lin.bias", "bert.transformer.layer.1.attention.v_lin.weight", "bert.transformer.layer.1.attention.v_lin.bias", "bert.transformer.layer.1.attention.out_lin.weight", "bert.transformer.layer.1.attention.out_lin.bias", "bert.transformer.layer.1.sa_layer_norm.weight", "bert.transformer.layer.1.sa_layer_norm.bias", "bert.transformer.layer.1.ffn.lin1.weight", "bert.transformer.layer.1.ffn.lin1.bias", "bert.transformer.layer.1.ffn.lin2.weight", "bert.transformer.layer.1.ffn.lin2.bias", "bert.transformer.layer.1.output_layer_norm.weight", "bert.transformer.layer.1.output_layer_norm.bias", "bert.transformer.layer.2.attention.q_lin.weight", "bert.transformer.layer.2.attention.q_lin.bias", "bert.transformer.layer.2.attention.k_lin.weight", "bert.transformer.layer.2.attention.k_lin.bias", "bert.transformer.layer.2.attention.v_lin.weight", "bert.transformer.layer.2.attention.v_lin.bias", "bert.transformer.layer.2.attention.out_lin.weight", "bert.transformer.layer.2.attention.out_lin.bias", "bert.transformer.layer.2.sa_layer_norm.weight", "bert.transformer.layer.2.sa_layer_norm.bias", "bert.transformer.layer.2.ffn.lin1.weight", "bert.transformer.layer.2.ffn.lin1.bias", "bert.transformer.layer.2.ffn.lin2.weight", "bert.transformer.layer.2.ffn.lin2.bias", "bert.transformer.layer.2.output_layer_norm.weight", "bert.transformer.layer.2.output_layer_norm.bias", "bert.transformer.layer.3.attention.q_lin.weight", "bert.transformer.layer.3.attention.q_lin.bias", "bert.transformer.layer.3.attention.k_lin.weight", "bert.transformer.layer.3.attention.k_lin.bias", "bert.transformer.layer.3.attention.v_lin.weight", "bert.transformer.layer.3.attention.v_lin.bias", "bert.transformer.layer.3.attention.out_lin.weight", "bert.transformer.layer.3.attention.out_lin.bias", "bert.transformer.layer.3.sa_layer_norm.weight", "bert.transformer.layer.3.sa_layer_norm.bias", "bert.transformer.layer.3.ffn.lin1.weight", "bert.transformer.layer.3.ffn.lin1.bias", "bert.transformer.layer.3.ffn.lin2.weight", "bert.transformer.layer.3.ffn.lin2.bias", "bert.transformer.layer.3.output_layer_norm.weight", "bert.transformer.layer.3.output_layer_norm.bias", "bert.transformer.layer.4.attention.q_lin.weight", "bert.transformer.layer.4.attention.q_lin.bias", "bert.transformer.layer.4.attention.k_lin.weight", "bert.transformer.layer.4.attention.k_lin.bias", "bert.transformer.layer.4.attention.v_lin.weight", "bert.transformer.layer.4.attention.v_lin.bias", "bert.transformer.layer.4.attention.out_lin.weight", "bert.transformer.layer.4.attention.out_lin.bias", "bert.transformer.layer.4.sa_layer_norm.weight", "bert.transformer.layer.4.sa_layer_norm.bias", "bert.transformer.layer.4.ffn.lin1.weight", "bert.transformer.layer.4.ffn.lin1.bias", "bert.transformer.layer.4.ffn.lin2.weight", "bert.transformer.layer.4.ffn.lin2.bias", "bert.transformer.layer.4.output_layer_norm.weight", "bert.transformer.layer.4.output_layer_norm.bias", "bert.transformer.layer.5.attention.q_lin.weight", "bert.transformer.layer.5.attention.q_lin.bias", "bert.transformer.layer.5.attention.k_lin.weight", "bert.transformer.layer.5.attention.k_lin.bias", "bert.transformer.layer.5.attention.v_lin.weight", "bert.transformer.layer.5.attention.v_lin.bias", "bert.transformer.layer.5.attention.out_lin.weight", "bert.transformer.layer.5.attention.out_lin.bias", "bert.transformer.layer.5.sa_layer_norm.weight", "bert.transformer.layer.5.sa_layer_norm.bias", "bert.transformer.layer.5.ffn.lin1.weight", "bert.transformer.layer.5.ffn.lin1.bias", "bert.transformer.layer.5.ffn.lin2.weight", "bert.transformer.layer.5.ffn.lin2.bias", "bert.transformer.layer.5.output_layer_norm.weight", "bert.transformer.layer.5.output_layer_norm.bias", "fc.weight", "fc.bias". 
#  Unexpected key(s) in state_dict: "model", "optimizer", "scheduler", "epoch".

def load_model( hp, num_steps ):  # num_steps used by learning rate scheduler, not needed for evaluate()
    # initialize model, optimizer, and LR scheduler
    # print(f"Default float dtype: {torch.get_default_dtype()}")  # Mặc định là float32
    # print(f"TF32 enabled for matmul: {torch.backends.cuda.matmul.allow_tf32}")
    # print(f"TF32 enabled for cuDNN: {torch.backends.cudnn.allow_tf32}")
    # print(hp.fp16)
    # exit()
    epoch      = 0
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    model      = DittoModel(device=device, lm=hp.lm, alpha_aug=hp.alpha_aug)
    model      = model.cuda() if torch.cuda.is_available() else model.cpu()
    optimizer  = AdamW( model.parameters(), lr=hp.lr )

    if hp.fp16:  model, optimizer = amp.initialize( model, optimizer, opt_level='O2' )
    #sn was:   num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=0, num_training_steps=num_steps )
    writer    = SummaryWriter( log_dir=hp.logdir )  # log with tensorboardX
    #h https://brsoff.github.io/tutorials/beginner/saving_loading_models.html
    #  documentation is bad?  checkpoint['model_state_dict'] --> KeyError
    ckpath = os.path.join( hp.logdir, hp.ckfile ) #'checkpoints/Structured/Beer/model.pt'
    if  ( hp.load_ckp is not None ) and ( os.path.isfile( ckpath ) ):
        checkpoint = torch.load(   ckpath )
        model.load_state_dict(     checkpoint['model'] )  #j
        optimizer.load_state_dict( checkpoint['optimizer'] )
        scheduler.load_state_dict( checkpoint['scheduler'] )  # need this line, otherwise model restart with bad f1
        epoch                    = checkpoint['epoch']

    tokenizer = AutoTokenizer.from_pretrained( hp.lm ) 
    tokens = ['[PERSON]', '[/PERSON]', '[ID]', '[/ID]']
    # model.set_finetune_layers()  #sn
    # init_tokens( tokens, tokenizer, model )
    return model, optimizer, scheduler, epoch

def init_tokens( tokens, tokenizer, model ):      # initialize custom vocab at centroid
    mean     = tokenizer.get_input_embeddings().weight.mean( dim=0 ) # average all embeddings
    tokenIds = tokenizer.convert_tokens_to_ids( tokens )
    for j in tokenIds:  self.bert.embeddings.word_embeddings.state_dict()['weight'][j] = mean
#        nn.init.normal_(self.fc1.weight, std=0.02);   nn.init.normal_(self.fc1.bias, 0)  #sn #d copying masayakondo
#        nn.init.normal_( self.fc.weight, std=0.02);   nn.init.normal_( self.fc.bias, 0)  #sn #d copying masayakondo

def rankprob( model, train_iter, threshold, outpath ):
    all_probs, all_y, yhat, yora = forwardSN( model, train_iter, threshold, label=True )
    data = []
    for i in range(len(all_probs)):
        line = i + 1; prob_0 = round(1 - all_probs[i], 3); prob_1 = round(all_probs[i], 3)
        label = all_y[i]; guess = yhat[i]
        data.append({"line":line,"prob-0":prob_0,"prob-1":prob_1,"label":label,"guess":guess})
    df = pandas.DataFrame(data); df.sort_values(by="prob-0", inplace=True)
    df.to_csv(outpath, index=False)


""" -------------- backup code ----------------
    # ------------ Fully Connected top, not transformer -----------
    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:  self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:            self.bert = AutoModel.from_pretrained(lm)
        self.device = device
        self.alpha_aug = alpha_aug
        hidden_size = self.bert.config.hidden_size  # 768
        mbedL = hidden_size
        self.dropout = nn.Dropout(0.2)
        self.fc1    = torch.nn.Linear( hidden_size, 20 )
        self.fc     = torch.nn.Linear( 20, 2 )
#        nn.init.normal_(self.fc1.weight, std=0.02);   nn.init.normal_(self.fc1.bias, 0)  #sn #d copying masayakondo
#        nn.init.normal_( self.fc.weight, std=0.02);   nn.init.normal_( self.fc.bias, 0)  #sn #d copying masayakondo

    def forward( self, x1, x2=None ):
        x1 = x1.to(self.device)     # (batch_size, seq_len)
        if x2 is not None:          # MixDA (Data Augmentation)
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)
            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :] #sn classification uses only the first vector, hence [:, 0, :] 
        xx  = torch.nn.functional.relu( self.dropout( self.fc1(enc) ) )
        out  = self.fc( xx )
        return out       #sn was:    return self.fc(enc) # .squeeze() # .sigmoid()
"""
"""    failed transformmer
# https://buomsoo-kim.github.io/attention/2020/04/22/Attention-mechanism-20.md/
    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:  self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:            self.bert = AutoModel.from_pretrained(lm)
        self.device = device
        self.alpha_aug = alpha_aug
        hidden_size  = self.bert.config.hidden_size  # 768
        mbedL        = hidden_size
        self.dropout = nn.Dropout(0.2)
        self.pe      = PositionalEncoding(embedding_dim, max_len = max_len)
        enc_layer    = nn.TransformerEncoderLayer( d_model=mbedL, , nhead=1, dim_feedforward=100, dropout=.2 )
        self.encoder = nn.TransformerEncoder( self.xf, num_layers=1 )  #       self.xf       = nn.TransformerEncoder( d_model = hidden_size, nhead=4, num_encoder_layers=1 )
        self.fc      = torch.nn.Linear( mbedL * 512, 2 )

    def forward(self, x):
      x = self.embedding(x).permute(1, 0, 2)
      x = self.pe(x)
      x = self.encoder(x)
      x = x.reshape(x.shape[1], -1)
      x = self.dense(x)
      return x

    def forward( self, x1, x2=None ):
        x1 = x1.to(self.device)     # (batch_size, seq_len)
        if x2 is not None:          # MixDA (Data Augmentation)
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)
            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, :, :] #sn classification uses only the first vector, hence [:, 0, :] 
        xx  = torch.nn.functional.relu( self.dropout( self.xf(enc) ) )
        out = self.fc( xx )
        pdb.set_trace()
        return out

#---------------------------------------
#         Position Encoding
# https://buomsoo-kim.github.io/attention/2020/04/22/Attention-mechanism-20.md
#d https://www.kaggle.com/code/masayakondo/tutorial-bert-classifier-using-pytorch#Declare-Classifier-Model
#---------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe           = torch.zeros(max_len, d_model)
        position     = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term     = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2]  = torch.sin(position * div_term)
        pe[:, 1::2]  = torch.cos(position * div_term)
        pe           = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
"""
