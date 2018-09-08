__author__ = 'Bruce_H_Cottman'
__license__ = 'MIT License'
# License: BSD 3 
import pandas as pd
import copy, random
import numpy as np
# steppy immports
from steppy.base import Step, BaseTransformer, make_transformer
from steppy.adapter import Adapter, E
from steppy.utils import get_logger
import joblib

########
logger = get_logger()

class S_toDataFrame(BaseTransformer):
    """
    A Step to transform a list, list of lists, numpy 1-D or 2-D array 
    into a pandas Series or DataFrame. 
    Usually used to transform a Steppy numpy array into a Dataframe.
    
    Args: None
    
    Example:
    
        >>> toDataFrame_step = Step(name='toDataFrame',
        >>>    transformer=S_toDataFrame(),
        >>>    input_data=['input'],
        >>>     adapter=Adapter({
        >>>       'X': E('input','features_o'),
        >>>           'labels': E('input','labels')
        >>>       }),
        >>>     experiment_directory=EXPERIMENT_DIR,
        >>>     is_trainable=False,
        >>>     force_fitting=False)
        >>> 
        >>> toDataFrame_step.transform(data_Housing_list)['X'].head(2)
        """
    
    def __init__(self):
        super().__init__()
        self.toDataFrame = self

    def fit(self, X):
        return self

    def transform(self, X, labels=[], inplace=False):
        """
        Transform a list, list of lists, numpy 1-D or 2-D array into a pandas Series or DataFrame.
        
        Args:
            X: (dataFrame,numpy array,list)
            labels: (single label str or list label str):  The column labels name to use for new DataFrame.
            inplace : (bool):, default False. If True, do operation inplace on X and return None.
            
        Returns:
            (pandas Series or DataFrame)
            pandas Series or DataFrame as X, will pass back unchanged.

        Raises:
            Will cast almost anything into a dataframe.
            1. ValueError will result of not 1-D or 2-D numpy array or list.
            2. ValueError will result of labels is not str or list of strings. 
       """
        
        if type(X) ==  pd.core.frame.DataFrame:
            return {'X': X}  
        elif type(X)== np.ndarray:
            if inplace:
                X = pd.DataFrame(X)
                y = X
            else:
                if len(X.shape) == 1 or len(X.shape) == 2:
                    y = pd.DataFrame(X)
                else:
                    raise ValueError('S_toDataFrame:transform:X:\
                        is of wrong dimenion: {} '.format(str(X.shape)))
            # case labels = scalar string
            cname = []
            if str(type(labels)) == '<class \'str\'>':
                cname.append(labels)
                labels =[labels]
            #case labels = list of strs
            elif type(labels) == list:
                #case len labels < len n-col, make column_n names
                n_col_names = len(labels)
                if n_col_names > y.shape[1]: n_col_names = y.shape[1]
                for name in labels[0:n_col_names]:
                    cname.append(name)
            # case labels = otherwise error
            else:
                raise ValueError('S_toDataFrame:transform:labels: {}\
                is of wrong type: {} '.format(labels,str(type(labels))))

            n_col_names = len(labels)
            if n_col_names < y.shape[1]: #len labels lt n columns
                for nth in range(n_col_names,y.shape[1]):
                    cname.append('Column_'+str(nth))            
            y.columns = cname
            return {'X': y}
        else:
            raise ValueError('S_toDataFrame:transform:X:\
            is of wrong type: {} '.format(str(type(X))))

    def load(self, filepath):
        self.toDataFrame = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.toDataFrame, filepath)