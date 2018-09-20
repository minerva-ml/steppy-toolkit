import category_encoders as ce
import numpy as np
import pandas as pd
import sklearn.decomposition as decomposition
from fancyimpute import SimpleFill
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


class FillNan(BaseTransformer):
    def __init__(self, fill_method='zero', fill_missing=True, **kwargs):
        """Imputs NaN's using various filling methods like mean, zero, median, min, random


        Args:
            fill_method: How NaN's will be exchanged. Possible values: 'mean', 'zero', 'median', 'min', 'random'
            fill_missing: If True, transformer will fill NaN values by filling method
        """
        super().__init__()
        self.fill_missing = fill_missing
        self.filler = SimpleFill(fill_method)

    def transform(self, X):
        """
        Args:
            X: DataFrame with NaN's
        Returns:
            Dictionary with one key - 'X' corresponding to given DataFrame but without nan's

        """
        if self.fill_missing:
            X = self.filler.complete(X)
        return {'X': X}

    def load(self, filepath):
        self.filler = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.filler, filepath)


class CategoricalEncoder(BaseTransformer):
    def __init__(self):
        """Encode features to categorical type"""
        super().__init__()
        self.encoder_class = ce.OrdinalEncoder
        self.categorical_encoder = None

    def fit(self, X):
        """
        Args:
            X: DataFrame of categorical features to encode
        """
        self.categorical_encoder = self.encoder_class(cols=list(X))
        self.categorical_encoder.fit(X)
        return self

    def transform(self, X):
        """
        Args:
            X: DataFrame of categorical features to encode
        Returns:
            Dictionary with one key - 'categorical_features' corresponding to encoded features form X
        """
        X = self.categorical_encoder.transform(X)
        return {'categorical_features': X}

    def load(self, filepath):
        self.categorical_encoder = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.categorical_encoder, filepath)


class GroupbyAggregate(BaseTransformer):
    def __init__(self, id_column, groupby_aggregations):
        """Group and aggregate features by specified configuration


        Args:
            id_column: Column with id's which will be preprocessed
            groupby_aggregations: list of tuples

        Example
            groupby_aggregations = [(['f0'], [('f2', 'min'),
                                              ('f2', 'median')]),
                                    (['f0', 'f1'], [('f2', 'mean'),
                                                    ('f2', 'max'),
                                                    ('f2', 'kurt')])]
            X = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0]])
            X = pd.DataFrame(X)
            X.columns = ['f0', 'f1', 'f2']

            tr = GroupbyAggregate(list(range(3)), groupby_aggregations)
            aggregations = tr.fit_transform(X)
        """
        super().__init__()
        self.id_column = id_column
        self.groupby_aggregations = groupby_aggregations

    def fit(self, X):
        features = pd.DataFrame({self.id_column: X[self.id_column].unique()})
        for groupby_cols, specs in self.groupby_aggregations:
            group_object = X.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                features = features.merge(group_object[select]
                                          .agg(agg)
                                          .reset_index()
                                          .rename(index=str,
                                                  columns={select: groupby_aggregate_name})
                                          [groupby_cols + [groupby_aggregate_name]],
                                          on=groupby_cols,
                                          how='left')
        self.features = features
        return self

    def transform(self, X):
        return {'numerical_features': self.features}

    def load(self, filepath):
        self.features = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.features, filepath)

    def _create_colname_from_specs(self, groupby_cols, select, agg):
        return '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)


class FeatureJoiner(BaseTransformer):
    """Concatenate all features to one DataFrame of given id_column

    Args:
        id_column: Column with id's which will be preprocessed
    """

    def __init__(self, id_column):
        super().__init__()
        self.id_column = id_column

    def transform(self, numerical_feature_list, categorical_feature_list):
        """
        Args:
            numerical_feature_list: list of numerical features
            categorical_feature_list: list of categorical features

        Returns:
            Dictionary with following keys:
                features: DataFrame with concatenated features
                feature_names: list of features names
                categorical_features: list of categorical feature names
        """
        features = numerical_feature_list + categorical_feature_list
        for feature in features:
            feature = self._format_target(feature)
            feature.set_index(self.id_column, drop=True, inplace=True)
        features = pd.concat(features, axis=1).astype(np.float32).reset_index()

        outputs = dict()
        outputs['features'] = features
        outputs['feature_names'] = list(features.columns)
        outputs['categorical_features'] = self._get_feature_names(categorical_feature_list)
        return outputs

    def _get_feature_names(self, dataframes):
        feature_names = []
        for dataframe in dataframes:
            try:
                feature_names.extend(list(dataframe.columns))
            except Exception as e:
                print(e)
                feature_names.append(dataframe.name)

        return feature_names

    def _format_target(self, target):
        if isinstance(target, pd.Series):
            return pd.DataFrame(target)
        return target