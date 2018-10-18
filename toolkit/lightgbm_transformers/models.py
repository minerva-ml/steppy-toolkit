from toolkit.toolkit_base import SteppyToolkitError

try:
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from sklearn.externals import joblib
    from steppy.base import BaseTransformer
    from steppy.utils import get_logger
except ImportError as e:
    msg = 'SteppyToolkitError: you have missing modules. Install requirements specific to lightgbm_transformers.' \
          'Use this file: toolkit/lightgbm_transformers/requirements.txt'
    raise SteppyToolkitError(msg) from e

logger = get_logger()


class LightGBM(BaseTransformer):
    """
    Accepts three dictionaries that reflects LightGBM API:
        - booster_parameters  -> parameters of the Booster
          See: https://lightgbm.readthedocs.io/en/latest/Parameters.html
        - dataset_parameters  -> parameters of the lightgbm.Dataset class
          See: https://lightgbm.readthedocs.io/en/latest/Python-API.html#data-structure-api
        - training_parameters -> parameters of the lightgbm.train function
          See: https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api
    """
    def __init__(self,
                 booster_parameters=None,
                 dataset_parameters=None,
                 training_parameters=None):
        super().__init__()
        logger.info('initializing LightGBM transformer')
        if booster_parameters is not None:
            isinstance(booster_parameters, dict), 'LightGBM transformer: booster_parameters must be dict, ' \
                                                  'got {} instead'.format(type(booster_parameters))
        if dataset_parameters is not None:
            isinstance(dataset_parameters, dict), 'LightGBM transformer: dataset_parameters must be dict, ' \
                                                  'got {} instead'.format(type(dataset_parameters))
        if training_parameters is not None:
            isinstance(training_parameters, dict), 'LightGBM transformer: training_parameters must be dict, ' \
                                                   'got {} instead'.format(type(training_parameters))

        self.booster_parameters = booster_parameters or {}
        self.dataset_parameters = dataset_parameters or {}
        self.training_parameters = training_parameters or {}

    def fit(self, X, y, X_valid, y_valid):
        self._check_target_shape_and_type(y, 'y')
        self._check_target_shape_and_type(y_valid, 'y_valid')
        y = self._format_target(y)
        y_valid = self._format_target(y_valid)

        logger.info('LightGBM transformer, train data shape        {}'.format(X.shape))
        logger.info('LightGBM transformer, validation data shape   {}'.format(X_valid.shape))
        logger.info('LightGBM transformer, train labels shape      {}'.format(y.shape))
        logger.info('LightGBM transformer, validation labels shape {}'.format(y_valid.shape))

        data_train = lgb.Dataset(data=X,
                                 label=y,
                                 **self.dataset_parameters)
        data_valid = lgb.Dataset(data=X_valid,
                                 label=y_valid,
                                 **self.dataset_parameters)
        self.estimator = lgb.train(params=self.booster_parameters,
                                   train_set=data_train,
                                   valid_sets=[data_train, data_valid],
                                   valid_names=['data_train', 'data_valid'],
                                   **self.training_parameters)
        return self

    def transform(self, X, **kwargs):
        prediction = self.estimator.predict(X)
        return {'prediction': prediction}

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.estimator, filepath)

    def _check_target_shape_and_type(self, target, name):
        if not any([isinstance(target, obj_type) for obj_type in [pd.Series, np.ndarray, list]]):
            msg = '"target" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(type(target))
            raise SteppyToolkitError(msg)
        if not isinstance(target, list):
            assert len(target.shape) == 1, '"{}" must be 1-D. It is {}-D instead.'.format(name, len(target.shape))

    def _format_target(self, target):
        if isinstance(target, pd.Series):
            return target.values
        elif isinstance(target, np.ndarray):
            return target
        elif isinstance(target, list):
            return np.array(target)
        else:
            raise TypeError(
                '"target" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(
                    type(target)))
