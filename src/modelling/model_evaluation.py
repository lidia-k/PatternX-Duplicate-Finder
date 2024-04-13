class EvaluationStrategy:
    def evaluate(self, model, data):
        pass


class PenumbraEvaluation(EvaluationStrategy):
    def evaluate(self, model, data):
        return model.run_eval(data)