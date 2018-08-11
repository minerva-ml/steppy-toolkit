import numpy as np
import pandas as pd
from sklearn.externals import joblib

from steppy.base import BaseTransformer


class ClassPredictor(BaseTransformer):
    def transform(self, prediction_proba):
        predictions_class = np.argmax(prediction_proba, axis=1)
        return {'y_pred': predictions_class}

    def load(self, filepath):
        return ClassPredictor()

    def persist(self, filepath):
        joblib.dump({}, filepath)


class PredictionAverage(BaseTransformer):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def transform(self, prediction_proba_list):
        if self.weights is not None:
            reshaped_weights = self._reshape_weights(prediction_proba_list.shape)
            prediction_proba_list *= reshaped_weights
            avg_pred = np.sum(prediction_proba_list, axis=0)
        else:
            avg_pred = np.mean(prediction_proba_list, axis=0)
        return {'prediction_probability': avg_pred}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.weights = params['weights']
        return self

    def persist(self, filepath):
        joblib.dump({'weights': self.weights}, filepath)

    def _reshape_weights(self, prediction_shape):
        dim = len(prediction_shape)
        reshape_dim = (-1,) + tuple([1] * (dim - 1))
        reshaped_weights = np.array(self.weights).reshape(reshape_dim)
        return reshaped_weights


class PredictionAverageUnstack(BaseTransformer):
    def transform(self, prediction_probability, id_list):
        df = pd.DataFrame(prediction_probability)
        df['id'] = id_list
        avg_pred = df.groupby('id').mean().reset_index().drop(['id'], axis=1).values
        return {'prediction_probability': avg_pred}

    def load(self, filepath):
        return self

    def persist(self, filepath):
        joblib.dump({}, filepath)


class ProbabilityCalibration(BaseTransformer):
    def __init__(self, power):
        super().__init__()
        self.power = power

    def transform(self, predicted_probability):
        predicted_probability = np.array(predicted_probability) ** self.power
        return {'predicted_probability': predicted_probability}


class BlendingOptimizer(BaseTransformer):
    """Class for optimizing the weights in blending of different model predictions.

    Args:
        metric (Callable): Callable metric function to optimize.
        maximize (bool): Boolean indicating whether the `metric` needs to be maximized or minimized.
        power (float): Power to apply on each models' predictions before blending.
    Example:
        >>> from sklearn.metrics import mean_absolute_error
        >>> y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        >>> p_model_1 = [0.11, 0.19, 0.25, 0.37, 0.55, 0.62, 0.78, 0.81, 0.94]
        >>> p_model_2 = [0.07, 0.21, 0.29, 0.33, 0.53, 0.54, 0.74, 0.74, 0.91]
        >>> p = [p_model_1, p_model_2]
        >>> opt = BlendingOptimizer(metric=mean_absolute_error, maximize=False)
        >>> opt.fit(y=y, p=p)
        >>> y_pred = opt.transform(p=p)['y_pred']
        >>> print('MAE 1: {:0.3f}'.format(mean_absolute_error(y, p_model_1)))
        >>> print('MAE 2: {:0.3f}'.format(mean_absolute_error(y, p_model_2)))
        >>> print('MAE blended: {:0.3f}'.format(mean_absolute_error(y, y_pred)))
    """

    def __init__(self, metric, maximize=True, power=1):
        self.metric = metric
        self.maximize = maximize
        self._power = power
        self._score = None
        self._weights = None

    def fit(self, y, p, step=0.1, init_weights=None, warm_start: bool = False):
        """Fit the model on the given predictions.

        Args:
            y (array-like): Labels.
            p (array-like): Predictions of different models for the labels.
            step (float): Step size for optimizing the weights. Smaller step sizes most likely improve resulting score
                          but increase training time.
            init_weights (array-like): Initial weights for training.
            warm_start (bool): Continues training. Will only work when `fit` has been called with this object earlier.
                               Ignores `init_weights``
        Returns: Optimized weights.
        """
        if (init_weights is not None) and warm_start:
            print('Warning: When warm_start is used init_weights are ignored.')

        assert np.shape(p)[1] == len(y), (
            'Length of predictions and labels does not match: {} != {}'.format(np.shape(p)[1], len(y)))

        def __is_better_score(score_to_test, score):
            return score_to_test > score if self.maximize else not score_to_test > score

        if warm_start:
            assert self._weights is not None, 'Optimizer has to be fitted before `warm_start` can be used.'
            weights = self._weights
        elif init_weights is None:
            weights = np.array([1.0] * len(p))
        else:
            assert (len(init_weights) == np.shape(p)[0]), (
                'Length of predictions and weights does not match: {} != {}'.format(np.shape(p)[0], len(init_weights)))
            weights = init_weights

        score = 0
        best_score = self.maximize - 0.5

        while __is_better_score(best_score, score):
            best_score = self.metric(y, np.average(np.power(p, self._power), weights=weights, axis=0) ** (
                    1.0 / self._power))
            score = best_score
            best_index, best_step = -1, 0.0
            for j in range(len(p)):
                delta = np.array([(0 if k != j else step) for k in range(len(p))])
                s = self.metric(y, np.average(np.power(p, self._power), weights=weights + delta, axis=0) ** (
                        1.0 / self._power))
                if __is_better_score(s, best_score):
                    best_index, best_score, best_step = j, s, step
                    continue
                if weights[j] - step >= 0:
                    s = self.metric(y, np.average(np.power(p, self._power), weights=weights - delta, axis=0) ** (
                            1.0 / self._power))
                    if s > best_score:
                        best_index, best_score, best_step = j, s, -step
            if __is_better_score(best_score, score):
                weights[best_index] += best_step

        self._weights = weights
        self._score = best_score

        return self

    def transform(self, p):
        """Transform blended predictions using the trained weights.

        Args:
            p (array-like): Predictions of different models.
        Returns: Blended predictions.
        """
        print(self._weights)
        assert np.shape(p)[0] == len(self._weights), (
            'Length of predictions and weights does not match: {} != {}'.format(np.shape(p)[0], len(self._weights)))
        blended_predictions = np.average(np.power(p, self._power), weights=self._weights, axis=0) ** (1.0 / self._power)

        return {'y_pred': blended_predictions}

    def fit_transform(self, y, p, step=0.1, init_weights=None, warm_start=False):
        """Fit and transform the optimizer. See `fit` and `transform` for further explanation."""
        self.fit(y=y, p=p, step=step, init_weights=init_weights, warm_start=warm_start)

        return self.transform(p=p)

    def load(self, filepath):
        params = joblib.load(filepath)
        self.metric = params['metric']
        self.maximize = params['maximize']
        self._power = params['power']
        self._score = params['score']
        self._weights = params['weights']
        return self

    def persist(self, filepath):
        joblib.dump({'metric': self.metric, 'maximize': self.maximize, 'power': self._power, 'score': self._score,
                     'weights': self._weights}, filepath)
