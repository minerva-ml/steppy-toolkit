from xgboost import XGBClassifier

from ..sklearn_transformers.models import MultilabelEstimators


class XGBoostClassifierMultilabel(MultilabelEstimators):
    @property
    def estimator(self):
        return XGBClassifier