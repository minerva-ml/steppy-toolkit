import numpy as np

import sklearn.linear_model as lr
from sklearn import ensemble
from sklearn import svm
from sklearn.externals import joblib

from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()


class SklearnBaseTransformer(BaseTransformer):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y)
        return self

    def save(self, filepath):
        joblib.dump(self.estimator, filepath)

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self


class SklearnClassifier(SklearnBaseTransformer):
    RESULT_KEY = 'predicted'

    def transform(self, X, y=None, **kwargs):
        prediction = self.estimator.predict_proba(X)
        return {SklearnClassifier.RESULT_KEY: prediction}


class SklearnRegressor(SklearnBaseTransformer):
    RESULT_KEY = 'predicted'

    def transform(self, X, y=None, **kwargs):
        prediction = self.estimator.predict(X)
        return {SklearnRegressor.RESULT_KEY: prediction}


class SklearnTransformer(SklearnBaseTransformer):
    RESULT_KEY = 'transformed'

    def transform(self, X, y=None, **kwargs):
        transformed = self.estimator.transform(X)
        return {SklearnTransformer.RESULT_KEY: transformed}


class SklearnPipeline(SklearnBaseTransformer):
    RESULT_KEY = 'transformed'

    def transform(self, X, y=None, **kwargs):
        transformed = self.estimator.transform(X)
        return {SklearnPipeline.RESULT_KEY: transformed}


class MultilabelEstimators(BaseTransformer):
    def __init__(self, label_nr, **kwargs):
        self.label_nr = label_nr
        self.estimators = self._get_estimators(**kwargs)

    @property
    def estimator(self):
        return NotImplementedError

    def _get_estimators(self, **kwargs):
        estimators = []
        for i in range(self.label_nr):
            estimators.append((i, self.estimator(**kwargs)))
        return estimators

    def fit(self, X, y, **kwargs):
        for i, estimator in self.estimators:
            logger.info('fitting estimator {}'.format(i))
            estimator.fit(X, y[:, i])
        return self

    def transform(self, X, y=None, **kwargs):
        predictions = []
        for i, estimator in self.estimators:
            prediction = estimator.predict_proba(X)
            predictions.append(prediction)
        predictions = np.stack(predictions, axis=0)
        predictions = predictions[:, :, 1].transpose()
        return {'predicted_probability': predictions}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.label_nr = params['label_nr']
        self.estimators = params['estimators']
        return self

    def save(self, filepath):
        params = {'label_nr': self.label_nr,
                  'estimators': self.estimators}
        joblib.dump(params, filepath)


class LogisticRegressionMultilabel(MultilabelEstimators):
    @property
    def estimator(self):
        return lr.LogisticRegression


class SVCMultilabel(MultilabelEstimators):
    @property
    def estimator(self):
        return svm.SVC


class LinearSVCMultilabel(MultilabelEstimators):
    @property
    def estimator(self):
        return LinearSVC_proba


class RandomForestMultilabel(MultilabelEstimators):
    @property
    def estimator(self):
        return ensemble.RandomForestClassifier


class LinearSVC_proba(svm.LinearSVC):
    def _platt_func(self, x):
        return 1.0 / (1 + np.exp(-x))

    def predict_proba(self, X):
        f = np.vectorize(self._platt_func)
        raw_predictions = self.decision_function(X)
        platt_predictions = f(raw_predictions).reshape(-1, 1)
        prob_positive = platt_predictions / platt_predictions.sum(axis=1)[:, None]
        prob_negative = 1.0 - prob_positive
        probabilities = np.hstack([prob_negative, prob_positive])
        return probabilities


def make_transformer(estimator, mode='classifier'):
    if mode == 'classifier':
        transformer = SklearnClassifier(estimator)
    elif mode == 'regressor':
        transformer = SklearnRegressor(estimator)
    elif mode == 'transformer':
        transformer = SklearnTransformer(estimator)
    elif mode == 'pipeline':
        transformer = SklearnPipeline(estimator)
    else:
        raise NotImplementedError("""Only classifier, regressor and transformer modes are available""")

    return transformer
