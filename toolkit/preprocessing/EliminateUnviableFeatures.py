import pandas as pd
import copy, random
import numpy as np
# steppy imports
from steppy.base import Step, BaseTransformer, make_transformer
from steppy.adapter import Adapter, E
from steppy.utils import get_logger
__author__ = 'Bruce_H_Cottman'
__license__ = 'MIT License'
#

LOGGER = get_logger()

class EliminateUnviableFeatures(BaseTransformer):
    """
    - check if any feature has NA values
    - - if there are any NA, then raise error.
    - - since values for NA are so domain dependent, it is required that all be removed before calling this Steppy transformer.   
    - Elimnate Features not found in train and test
    - Eliminate duplicate features ; check all values
    - Eliminate single unique value features
    - Eliminate low variance features:  > std/(max/min)
    - All deleted features are logged


    """

    def __init__(self):
        """
            Parameters:
                VARIANCE_LIMIT = 0.4999 (default)
        
        """
        
        super().__init__()
        try: self.SD_LIMIT = SD_LIMIT
        except: self.SD_LIMIT = 0.317
        self.X = None
        self.X_test = None

#0    
    def _Check_No_NA_Values(self,df):    
        return(not df.isna().any().any())
#1        
    def _Eliminate_Features_not_found_in_train_and_test(self,train,test):
        rem = set(train.columns).difference(set(test.columns))
        efs = []; n = 0
        if len(rem) >= 1:
            for f in rem:
                train.drop(f,inplace=True,axis=1)
                efs.append(f)
                n += 1
        if n > 0:  LOGGER.info('_Eliminate_Features_not_found_in_TEST: {}'.format(str(efs)))
            
        rem = set(test.columns).difference(set(train.columns))
        efs = []; n = 0
        if len(rem) >= 1:
            for f in rem:
                test.drop(f,inplace=True,axis=1)
                efs.append(f)
                n += 1
        if n > 0:  LOGGER.info('_Eliminate_Features_not_found_in_TRAIN {}'.format(str(efs)))

        return(train,test)
#2    
    def _Eliminate_Duplicate_Features(self,df):
        Xp = df.T.drop_duplicates()
        dfp = Xp.T
        def diff(first, second):
            second = set(second)
            return [item for item in first if item not in second]
        LOGGER.info('_Eliminate_Duplicate_Features: {}'.format(str(diff(df.columns,dfp.columns))))
        return(df)
#3         
    def _Eliminate_Single_Unique_Value_Features(self,df):
        efs = []; n = 0
        for f in df.columns:
            #wierdness where df has 2 or more features with sam name
            if(len(df[f].squeeze().shape) == 1):  
                if (len( df[f].squeeze().unique()) == 1):
                    df.drop(f,inplace=True,axis=1)
                    efs.append(f)
                    n += 1
        if n > 0:  LOGGER.info('Eliminated Eliminate_Single_Unique_Value_Features {}'.format(str(efs)))
        return(df)
#4
    def _Eliminate_Low_Variance_Features(self,df):
        """
        At least 10 or more values in a feature
        
        """
        
        efs = []; n = 0
        for f in df.columns:
            #wierdness where df has 2 or more features with sam name
            if(len(df[f].squeeze().shape) == 1): 
                if df[f].dtype == np.number:
                    if (df[f].std()/(df[f].max()-df[f].min()) < self.SD_LIMIT):
                        if df[f].shape[0] <= 10:  #must have more than 10 values in feature
                            LOGGER.info('_Eliminate_Low_Variance_Features Before len lt 10 {}'.format(f))
                            pass
                        else:
                            df.drop(f,inplace=True,axis=1)
                            efs.append(f)
                            n += 1              
        if n > 0: LOGGER.info('Eliminated _Eliminate_Low_Variance_Features {}'.format(str(efs)))

        return(df)


    
    def transform(self, X, X_test):
        """
        Args:
            X: (DataFrame)
            X_test: (DataFrame)
        
        Returns:
            X: X: (DataFrame)
                EliminateUnviableFeatures transformed train DataFrame
            X_test: (DataFrame
                EliminateUnviableFeatures transformed test DataFrame
        
        Raises:
            ValueError: if any feature has NA values.
            ValueError: if any passed X or X_test arguments are other than datatype.
            ValueError: if X and X_test arguments have orthogonal features.
            ValueError: if X and X_test arguments after _Eliminate_Single_Unique_Value_Features has no features.  
            ValueError: if X and X_test arguments after _Eliminate_Low_Variance_Features has no features. 
        """
        
        if type(X) ==  pd.core.frame.DataFrame:
            pass 
        else:
            raise TypeError('EliminateUnviableFeatures:transform:X: \
                    must be type DataFrame, was type: {} '.format(str(type(X))))
        if type(X_test) ==  pd.core.frame.DataFrame:
            pass 
        else:
            raise ValueError('EliminateUnviableFeatures:transform:X_test: \
                    must be type DataFrame, was type: {} '.format(str(type(X_test))))
        if self._Check_No_NA_Values(X):
            pass
        else:
             raise ValueError('EliminateUnviableFeatures:transform:X_test: \
                    X DataFrame has NaN values: ')     
        if self._Check_No_NA_Values(X_test):
            pass
        else:
             raise ValueError('EliminateUnviableFeatures:transform:X_test: \
                    X_test DataFrame has NaN values: ') 
            
        X,X_test = self._Eliminate_Features_not_found_in_train_and_test(X,X_test)
        LOGGER.debug(' ******* {} {} {}'.format(X,'\n',X.shape))
        if (X.shape[1] == 0) or (X_test.shape[1] == 0):
            raise TypeError('_Elimnate_Features_not_found_in_train_and_test:transform:X and X_test orthogonal.')
        print(type(X))
#        X = self._Eliminate_Duplicate_Features(X)
        if (X.shape[1] == 0) :
            raise TypeError('_Eliminate_Duplicate_Features:transform:X: has no features.')            
            
        X = self._Eliminate_Single_Unique_Value_Features(X)
        if (X.shape[1] == 0) :
            raise TypeError('_Eliminate_Single_Unique_Value_Features:transform:X: has no features.')
        #   _Eliminate_Single_Unique_Value_Features should eliminate
        X = self._Eliminate_Low_Variance_Features(X)                
        if (X.shape[1] == 0) :
            raise TypeError('_Eliminate_Low_Variance_Features:transform:X: has no features.')     
            
        self.X = X
        self.X_test = X_test
        return {'X': X,
               'X_test': X_test }  
                
    def fit(self):
        """
        Not Used
        """
        
        return self
    
    def load(self, filepath):
        self.X,self.X_test = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump([self.X,self.X_test], filepath)