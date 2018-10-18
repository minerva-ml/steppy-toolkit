from toolkit.toolkit_base import SteppyToolkitError

try:
    import xgboost as xgb
    from attrdict import AttrDict
    from steppy.base import BaseTransformer
    from steppy.utils import get_logger
    from xgboost import XGBClassifier

    from toolkit.sklearn_transformers.models import MultilabelEstimators
except ImportError as e:
    msg = 'SteppyToolkitError: you have missing modules. Install requirements specific to xgboost_transformers.' \
          'Use this file: toolkit/xgboost_transformers/requirements.txt'
    raise SteppyToolkitError(msg) from e

logger = get_logger()


class XGBoostClassifierMultilabel(MultilabelEstimators):
    @property
    def estimator(self):
        return XGBClassifier


class XGBoost(BaseTransformer):
    """
    Accepts three dictionaries that reflects XGBoost API:
        - dmatrix_parameters  -> parameters of the xgboost.DMatrix class.
          See: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.DMatrix
        - training_parameters -> parameters of the xgboost.train function.
          See: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train
        - predict_parameters -> parameters of the xgboost.Booster.predict function.
          See: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.predict
        - booster_parameters  -> parameters of the Booster.
          See: https://xgboost.readthedocs.io/en/latest/parameter.html
    """
    def __init__(self,
                 dmatrix_parameters=None,
                 training_parameters=None,
                 predict_parameters=None,
                 booster_parameters=None):
        super().__init__()
        logger.info('initializing XGBoost transformer')
        if dmatrix_parameters is not None:
            isinstance(dmatrix_parameters, dict), 'XGBoost transformer: dmatrix_parameters must be dict, ' \
                                                  'got {} instead'.format(type(dmatrix_parameters))
        if training_parameters is not None:
            isinstance(training_parameters, dict), 'XGBoost transformer: training_parameters must be dict, ' \
                                                   'got {} instead'.format(type(training_parameters))
        if predict_parameters is not None:
            isinstance(predict_parameters, dict), 'XGBoost transformer: predict_parameters must be dict, ' \
                                                  'got {} instead'.format(type(predict_parameters))
        if booster_parameters is not None:
            isinstance(booster_parameters, dict), 'XGBoost transformer: booster_parameters must be dict, ' \
                                                  'got {} instead'.format(type(booster_parameters))

        self.dmatrix_parameters = dmatrix_parameters or {}
        self.training_parameters = training_parameters or {}
        self.predict_parameters = predict_parameters or {}
        self.booster_parameters = booster_parameters or {}

    def fit(self, X, y, X_valid, y_valid):
        logger.info('XGBoost, train data shape        {}'.format(X.shape))
        logger.info('XGBoost, validation data shape   {}'.format(X_valid.shape))
        logger.info('XGBoost, train labels shape      {}'.format(y.shape))
        logger.info('XGBoost, validation labels shape {}'.format(y_valid.shape))

        train = xgb.DMatrix(data=X,
                            label=y,
                            **self.dmatrix_parameters)
        valid = xgb.DMatrix(data=X_valid,
                            label=y_valid,
                            **self.dmatrix_parameters)
        self.estimator = xgb.train(params=self.booster_parameters,
                                   dtrain=train,
                                   evals=[(train, 'train'), (valid, 'valid')],
                                   **self.training_parameters)
        return self

    def transform(self, X, y=None, **kwargs):
        X_DMatrix = xgb.DMatrix(X, label=y, **self.dmatrix_parameters)
        prediction = self.estimator.predict(X_DMatrix, **self.predict_parameters)
        return {'prediction': prediction}

    def load(self, filepath):
        self.estimator = xgb.Booster(params=self.booster_parameters)
        self.estimator.load_model(filepath)
        return self

    def persist(self, filepath):
        self.estimator.save_model(filepath)
