class EvaluationStrategy:
    def evaluate(self, model, data):
        pass


class PeNumBraEvaluation(EvaluationStrategy):
    def evaluate(self, model, data):
        return model.run_eval(data)