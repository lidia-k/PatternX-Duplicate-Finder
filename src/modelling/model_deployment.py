import numpy as np 

class ModelDeployer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelDeployer, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance

    def deploy_model(self, model):
        self.model = model

    def predict(self, data):
        return self.model.predict(data)

class PenumbraModelDeployer(ModelDeployer):
    def predict(self, candidate):
        predictions = self.model.run_prediction(candidate, output_attributes=list(candidate.get_raw_table().columns))
        predictions = predictions.rename(columns={"match_score":"confident"})
        threshold = 0.8
        predictions['is_matched'] = np.where(predictions['confident'] > threshold, 1, 0)

        return predictions