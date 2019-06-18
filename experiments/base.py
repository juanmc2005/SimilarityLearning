

class ModelEvaluationExperiment:

    def evaluate_on_dev(self) -> float:
        raise NotImplementedError

    def evaluate_on_test(self) -> float:
        raise NotImplementedError
