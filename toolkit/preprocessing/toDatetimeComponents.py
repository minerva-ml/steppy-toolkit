import pandas as pd
import copy, random
import numpy as np
from tqdm import tqdm
# steppy imports
import joblib
from steppy.base import Step, BaseTransformer, make_transformer
from steppy.adapter import Adapter, E
from steppy.utils import get_logger
__author__ = 'Bruce_H_Cottman'
__license__ = 'MIT License'
#

LOGGER = get_logger()


class toDatetimeComponents(BaseTransformer):
    """
    
    Args:

        dt_features: (list): (default) None

            None ==[Year', 'Month', 'Week', 'Day','Dayofweek'
                      , 'Dayofyear','Elapsed','Is_month_end'
                      , 'Is_month_start', 'Is_quarter_end'
                      , 'Is_quarter_start', 'Is_year_end', 'Is_year_start']dt_features = ['Year', 'Month', 'Week', 'Day','Dayofweek'
                      , 'Dayofyear','Elapsed','Is_month_end'
                      , 'Is_month_start', 'Is_quarter_end'
                      , 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
                      
            or set ``dt_features`` to one or compnents names in a list. Must be compnents names from defauly list.                        
    """

    def __init__(self,dt_features = None):

        super().__init__()
        self.X = None
        self.PREFIX_LENGTH = 3
        if dt_features == None:
            self.dt_features = ['Year', 'Month', 'Week', 'Day','Dayofweek'
                                  , 'Dayofyear','Elapsed','Is_month_end'
                                  , 'Is_month_start', 'Is_quarter_end'
                                  , 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        else:
            self.dt_features = dt_features
            
    def _Check_No_NA_Values(self,df,featurename):    
        return(not df[featurename].isna().any())
    
    def transform(self, X, featurename, prefix=True, drop=True):
        """
        Note:
            if X[featurename] not of datetime type  (such as ``object`` type)
            then an attempt to coerce X[featurename] to ``datetime`` type is made.
            Successful coercion to ``datetime`` costs aproximately 100x more than if 
            X[featurename] was already of type datetime. It is best if raw data field
            is read/input in as ``datetime`` rather than ``object``.            

        Args:
            X: (DataFrame)

            fldname: (string)
            
                A string that is the name of the feature column of type datetime. 
                If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
                
            prefix: (default) True
            
                If true then the featurename will be the prefix of the created  datetime component fetures.
                the posfix will be _<component> to create the new feature column <featurename>_<component>.
                if False only first 3 characters of featurename eill be used to create the new feature 
                column <featurename[0:2]>_<component>.
                
            drop: (default) True
            
                If true then the original date column will be removed.
                

        Returns:
            X: X: (DataFrame)
            
                toDatetimeComponents transformed into datetime feature components
        
        Raises:
            ValueError: if any feature has NA values.
 
        """
        if not (featurename in X.columns):
             raise ValueError('toDatetimeComponents:transform:X: unknown feature: {}'\
                              .format(str(featurename)))            
        if np.issubdtype( X[featurename].dtype, np.datetime64):
            pass
        else:
            X[featurename] = pd.to_datetime(X[featurename], infer_datetime_format=True)
            
        if self._Check_No_NA_Values(X,featurename):
            pass
        else:
             raise ValueError('toDatetimeComponents:transform:X: DataFrame {} has NaN values: '\
                              .format(str(featurename)))
                
        LOGGER.info('datetime feature components added: {}'.format(self.dt_features))
        for dt_feature in tqdm(self.dt_features):
            if (not dt_feature.lower() == 'Elapsed'.lower()):
                if(prefix): fn = featurename+'_'+dt_feature
                else: fn = featurename[0:self.PREFIX_LENGTH]+'_'+dt_feature        
                X[fn] = getattr(X[featurename].dt,dt_feature.lower()).astype(np.int32)

        if ('Elapsed' in self.dt_features): 
            if(prefix): fn = featurename+'_'+'Elapsed'
            else: fn = featurename[0: self.PREFIX_LENGTH]+'_'+'Elapsed'
            X[fn] = X[featurename].astype(np.int64) // 10**9  #ns to seconds
            
        if drop: X.drop(featurename, axis=1, inplace=True)

        self.X = X

        return {'X': X}  
                
    def fit(self):
        """
        Not Used
        
        """
        
        return self
    
    def load(self, filepath):
        """
        Args:
           filepath: stores X transform into datetime components at ``filepath``.
           
        """

        self.X = joblib.load(filepath)
        return self

    def persist(self, filepath):
        """
        Args:
           filepath: loads X transform into datetime components at ``filepath``.
           
        """
        
        joblib.dump(self.X, filepath)