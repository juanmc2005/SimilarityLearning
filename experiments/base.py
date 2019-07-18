

class ModelEvaluationExperiment:

    def evaluate_on_dev(self, plot: bool) -> float:
        raise NotImplementedError

    def evaluate_on_test(self) -> float:
        raise NotImplementedError
