import numpy as np 
from src.modelling.model_evaluation import PenumbraEvaluation
import src.lib.deepmatcher as dm 
from src.lib.deepmatcher.optim import SoftNLLLoss, Optimizer
from src.lib.deepmatcher.runner import Runner, Statistics
from src.lib.deepmatcher.data.iterator import MatchingIterator
import src.utils.auto_config as config 
import torch 
from tqdm import tqdm

class ModelTrainer:
    def train_model(self, features, labels):
        pass


class PenumbraModelTrainer(ModelTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.data_columns = [
            'ltable_Email Address', 'ltable_First Name',
            'ltable_Quickbase Record ID#', 'ltable_Specialty', 'ltable_Last Name',
            'ltable_Status', 'ltable_SAP Number', 'ltable_Payments Made To:',
            'ltable_Payment Currency', 'ltable_HCP Category',
            'ltable_State/Region/Province', 'ltable_Full Name',
            'ltable_Contact Type', 'ltable_National Physician ID', 'ltable_Country',
            'ltable_SAP Entity Name', 'rtable_Email Address', 'rtable_First Name',
            'rtable_Quickbase Record ID#', 'rtable_Specialty', 'rtable_Last Name',
            'rtable_Status', 'rtable_SAP Number', 'rtable_Payments Made To:',
            'rtable_Payment Currency', 'rtable_HCP Category',
            'rtable_State/Region/Province', 'rtable_Full Name',
            'rtable_Contact Type', 'rtable_National Physician ID', 'rtable_Country',
            'rtable_SAP Entity Name', 'label'
       ]
        self.model_evaluator = PenumbraEvaluation()
        
    def train_model(self, df):
        pos_neg_ratio = np.sum(df['label'] == 1)/ np.sum(df['label'] == 0)
        dm.data.split(df, config.DATA_DIR, 'train.csv', 'valid.csv', 'test.csv',[3, 1, 1])

        train, validation, test = dm.data.process(
            path=config.DATA_DIR,
            cache='train_cache0.pth',
            train='train.csv',
            validation='valid.csv',
            test='test.csv',
            use_magellan_convention=True
        )

        model = dm.MatchingModel(attr_summarizer='hybrid')
        model.run_train(train, validation, epochs=3, batch_size=16, best_save_path= config.MODEL_FOLDER +"model.pth", pos_neg_ratio=pos_neg_ratio)
        self.model_evaluator.evaluate(model, test)
        return model
    
    def online_training(self, model, file_path = 'online_train.csv'):        
        online_train = dm.data.process(path=".",cache='online_train1.pth', train=file_path,use_magellan_convention=True)
        pos_neg_ratio = 1
        pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)
        neg_weight = 2 - pos_weight
        criterion = SoftNLLLoss(0.05, torch.Tensor([neg_weight, pos_weight]))

        print(f"neg_weight:{neg_weight}, pos_weight: {pos_weight}")

        label_attr = 'label'
        optimizer = Optimizer()
        stats = Statistics()

        losses = []

        run_iter = MatchingIterator(
                    online_train,
                    model.meta,
                    train=True,
                    batch_size=1,
                    device='cpu',
                    sort_in_buckets=True)

        for epoch in tqdm(range(10)):
            for batch_idx, batch in enumerate(run_iter):
                output = model(batch)
                loss = criterion(output, getattr(batch, label_attr))
                
                if hasattr(batch, label_attr):
                    scores = Runner._compute_scores(output, getattr(batch, label_attr))
                else:
                    scores = [0] * 4

                stats.update(float(loss), *scores)
                model.zero_grad()
                loss.backward()

                if not optimizer.params:
                    optimizer.set_parameters(model.named_parameters())
                optimizer.step()
            losses.append(loss)
            Runner._print_final_stats(epoch + 1, 0, 0, stats)

        save_to = config.MODEL_FOLDER +"retrained_model.pth"
        model.save_state(save_to)
        print(f"Saved model to: {save_to}")
        return model
        
            