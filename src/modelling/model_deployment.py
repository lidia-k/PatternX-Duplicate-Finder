import numpy as np 
import src.lib.deepmatcher as dm 
import os 

class ModelDeployer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelDeployer, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance

    @staticmethod
    def deploy_model(model):
        ModelDeployer._instance.model = model
        return ModelDeployer._instance

    @staticmethod
    def predict(data):
        return ModelDeployer._instance.model.predict(data)

class PenumbraModelDeployer(ModelDeployer):
    model = None

    def deploy_model(self, model_path):
        self.model = dm.MatchingModel(attr_summarizer='hybrid')
        self.model.load_state(model_path)

    def predict(self, new_data_file_path):
        candidate = dm.data.process_unlabeled(
                                    path=os.path.join(new_data_file_path),
                                    trained_model=self.model,
                                    ignore_columns=('ltable_id', 'rtable_id', 'label'))
        print(candidate.get_raw_table().columns)                                
        predictions = self.model.run_prediction(candidate, output_attributes=list(candidate.get_raw_table().columns))
        predictions = predictions.rename(columns={"match_score":"confident"})
        threshold = 0.8
        predictions['is_matched'] = np.where(predictions['confident'] > threshold, 1, 0)

        return predictions