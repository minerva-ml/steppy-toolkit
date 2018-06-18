import lightgbm as lgb
import numpy as np
import pandas as pd
from attrdict import AttrDict
from catboost import CatBoostClassifier
from sklearn.externals import joblib
from steppy.base import BaseTransformer
from steppy.utils import get_logger
from xgboost import XGBClassifier

from .sklearn_transformers.models import MultilabelEstimators

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
        super().__init__()
        logger.info('initializing LightGBM...')
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

    def _check_target_shape_and_type(self, target, name):
        if isinstance(target, list):
            return np.array(target)
        try:
            assert len(target.shape) == 1, '"{}" must be 1-D. It is {}-D instead.'.format(name,
                                                                                          len(target.shape))
        except AttributeError:
            print('Cannot determine shape of the {}. '
                  'Type must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead'.format(name,
                                                                                                     type(target)))

        if isinstance(target, pd.Series):
            return target.values
        if not isinstance(target, np.ndarray):
            TypeError('"{}" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(name,
                                                                                                          type(target)))

    def fit(self,
            X, y,
            X_valid, y_valid,
            feature_names='auto',
            categorical_features='auto',
            **kwargs):

        y = self._check_target_shape_and_type(y, 'y')
        y_valid = self._check_target_shape_and_type(y_valid, 'y_valid')

        data_train = lgb.Dataset(data=X,
                                 label=y,
                                 feature_name=feature_names,
                                 categorical_feature=categorical_features,
                                 **kwargs)
        data_valid = lgb.Dataset(X_valid,
                                 label=y_valid,
                                 feature_name=feature_names,
                                 categorical_feature=categorical_features,
                                 **kwargs)

        logger.info('LightGBM, training data has {!s} examples.'.format(data_train.num_data()))
        logger.info('LightGBM, training data has {!s} features.'.format(data_train.num_feature()))

        logger.info('LightGBM, validation data has {!s} examples.'.format(data_valid.num_data()))
        logger.info('LightGBM, validation data has {!s} features.'.format(data_train.num_feature()))

        evaluation_results = {}
        self.estimator = lgb.train(self.model_config,
                                   data_train,
                                   feature_name=feature_names,
                                   categorical_feature=categorical_features,
                                   valid_sets=[data_train, data_valid],
                                   valid_names=['data_train', 'data_valid'],
                                   evals_result=evaluation_results,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function,
                                   **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        prediction = self.estimator.predict(X)
        return {'prediction': prediction}

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.estimator, filepath)
