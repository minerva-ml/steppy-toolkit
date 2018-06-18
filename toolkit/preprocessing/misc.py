import sklearn.decomposition as decomposition
from sklearn.externals import joblib
from sklearn.feature_extraction import text
from sklearn.preprocessing import Normalizer, MinMaxScaler

from steppy.base import BaseTransformer


class XYSplit(BaseTransformer):
    def __init__(self, x_columns, y_columns):
        super().__init__()
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.columns_to_get = None
        self.target_columns = None

    def transform(self, meta, train_mode):
        X = meta[self.x_columns].values
        if train_mode:
            y = meta[self.y_columns].values
        else:
            y = None

        return {'X': X,
                'y': y}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.columns_to_get = params['x_columns']
        self.target_columns = params['y_columns']
        return self

    def persist(self, filepath):
        params = {'x_columns': self.x_columns,
                  'y_columns': self.y_columns
                  }
        joblib.dump(params, filepath)


class TfIdfVectorizer(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.vectorizer = text.TfidfVectorizer(**kwargs)

    def fit(self, text):
        self.vectorizer.fit(text)
        return self

    def transform(self, text):
        return {'features': self.vectorizer.transform(text)}

    def load(self, filepath):
        self.vectorizer = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.vectorizer, filepath)


class TruncatedSVD(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.truncated_svd = decomposition.TruncatedSVD(**kwargs)

    def fit(self, features):
        self.truncated_svd.fit(features)
        return self

    def transform(self, features):
        return {'features': self.truncated_svd.transform(features)}

    def load(self, filepath):
        self.truncated_svd = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.truncated_svd, filepath)


class Steppy_Normalizer(BaseTransformer):
    def __init__(self):
        super().__init__()
        self.normalizer = Normalizer()

    def fit(self, X):
        self.normalizer.fit(X)
        return self

    def transform(self, X):
        X = self.normalizer.transform(X)
        return {'X': X}

    def load(self, filepath):
        self.normalizer = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.normalizer, filepath)


class Steppy_MinMaxScaler(BaseTransformer):
    def __init__(self):
        super().__init__()
        self.minmax_scaler = MinMaxScaler()

    def fit(self, X):
        self.minmax_scaler.fit(X)
        return self

    def transform(self, X):
        X = self.minmax_scaler.transform(X)
        return {'X': X}

    def load(self, filepath):
        self.minmax_scaler = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.minmax_scaler, filepath)


class MinMaxScalerMultilabel(BaseTransformer):
    def __init__(self):
        super().__init__()
        self.minmax_scalers = []

    def fit(self, X):
        for i in range(X.shape[1]):
            minmax_scaler = Steppy_MinMaxScaler()
            minmax_scaler.fit(X[:, i, :])
            self.minmax_scalers.append(minmax_scaler)
        return self

    def transform(self, X):
        for i, minmax_scaler in enumerate(self.minmax_scalers):
            X[:, i, :] = minmax_scaler.transform(X[:, i, :])
        return {'X': X}

    def load(self, filepath):
        self.minmax_scalers = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.minmax_scalers, filepath)
