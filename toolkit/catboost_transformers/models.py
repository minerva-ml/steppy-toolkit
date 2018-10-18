from toolkit.toolkit_base import SteppyToolkitError

try:
    import catboost as ctb
    from catboost import CatBoostClassifier
    from steppy.base import BaseTransformer
    from steppy.utils import get_logger

    from toolkit.sklearn_transformers.models import MultilabelEstimators
except ImportError as e:
    msg = 'SteppyToolkitError: you have missing modules. Install requirements specific to catboost_transformers.' \
          'Use this file: toolkit/catboost_transformers/requirements.txt'
    raise SteppyToolkitError(msg) from e

logger = get_logger()


class CatboostClassifierMultilabel(MultilabelEstimators):
    @property
    def estimator(self):
        return CatBoostClassifier


class CatBoost(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.estimator = ctb.CatBoostClassifier(**kwargs)

    def fit(self,
            X, y,
            X_valid, y_valid,
            feature_names=None,
            categorical_features=None,
            **kwargs):

        logger.info('Catboost, train data shape        {}'.format(X.shape))
        logger.info('Catboost, validation data shape   {}'.format(X_valid.shape))
        logger.info('Catboost, train labels shape      {}'.format(y.shape))
        logger.info('Catboost, validation labels shape {}'.format(y_valid.shape))

        categorical_indeces = self._get_categorical_indices(feature_names, categorical_features)
        self.estimator.fit(X, y,
                           eval_set=(X_valid, y_valid),
                           cat_features=categorical_indeces)
        return self

    def transform(self, X, **kwargs):
        prediction = self.estimator.predict_proba(X)[:, 1]
        return {'prediction': prediction}

    def load(self, filepath):
        self.estimator.load_model(filepath)
        return self

    def persist(self, filepath):
        self.estimator.save_model(filepath)

    def _get_categorical_indices(self, feature_names, categorical_features):
        if categorical_features:
            return [feature_names.index(feature) for feature in categorical_features]
        else:
            return None
