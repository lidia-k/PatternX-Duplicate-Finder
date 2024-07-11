import src.lib.deepmatcher as dm
import src.utils.auto_config as config 


class PenumbraEvaluation:
    def evaluate(self, model, data):
        return model.run_eval(data)
    
    def load_model(self, model_path):
        model = dm.MatchingModel(attr_summarizer='hybrid')
        model.load_state(model_path)
        return model
    
    def compare_models(self, model1, model2, data):
        eval1 = self.evaluate(model1, data)
        eval2 = self.evaluate(model2, data)
        
        save_to = config.MODEL_FOLDER +"best_model.pth"
        
        print(f"Prediction score model1: {eval1}, model2: {eval2}")
        
        if eval1 > eval2:
            print("Model1 is better than model2")
            print("Save model1 to the best_model.pth")
            model1.save_state(save_to)
        else:
            print("Model2 is better than model1")
            print("Save model2 to the best_model.pth")
            model1.save_state(save_to)
        
    
        