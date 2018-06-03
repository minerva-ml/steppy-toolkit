import lightgbm as lgb
from attrdict import AttrDict
from catboost import CatBoostClassifier
from sklearn.externals import joblib
from steppy.base import BaseTransformer
from steppy.utils import get_logger
from xgboost import XGBClassifier

from .sklearn_recipes.models import MultilabelEstimators

logger = get_logger()


class CatboostClassifierMultilabel(MultilabelEstimators):
    @property
    def estimator(self):
        return CatBoostClassifier


class XGBoostClassifierMultilabel(MultilabelEstimators):
    @property
    def estimator(self):
        return XGBClassifier


class LightGBM(BaseTransformer):
    def __init__(self, **params):
        self.params = params
        self.training_params = ['number_boosting_rounds', 'early_stopping_rounds']
        self.evaluation_function = None

    @property
    def model_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param not in self.training_params})

    @property
    def training_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param in self.training_params})

    def fit(self,
            X, y,
            X_valid, y_valid,
            feature_names=None,
            categorical_features=None,
            **kwargs):
        train = lgb.Dataset(X,
                            label=y,
                            feature_name=feature_names,
                            categorical_feature=categorical_features)
        valid = lgb.Dataset(X_valid,
                            label=y_valid,
                            feature_name=feature_names,
                            categorical_feature=categorical_features)

        evaluation_results = {}
        self.estimator = lgb.train(self.model_config,
                                   train,
                                   valid_sets=[train, valid],
                                   valid_names=['train', 'valid'],
                                   evals_result=evaluation_results,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function)
        return self

    def transform(self, X, y=None, **kwargs):
        prediction = self.estimator.predict(X)
        return {'prediction': prediction}

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.estimator, filepath)
