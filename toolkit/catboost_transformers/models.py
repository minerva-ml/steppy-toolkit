from catboost import CatBoostClassifier

from ..sklearn_transformers.models import MultilabelEstimators


class CatboostClassifierMultilabel(MultilabelEstimators):
    @property
    def estimator(self):
        return CatBoostClassifier
