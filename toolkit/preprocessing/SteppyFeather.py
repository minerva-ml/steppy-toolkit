'''
References:

[1] 

'''
__author__ = 'Bruce_H_Cottman'
__license__ = 'MIT License'
# License: BSD 3 
# move to sklearn_transformers??
import pandas as pd
import copy, random
import numpy as np
from numba import jit
# steppy immports
from steppy.base import Step, BaseTransformer, make_transformer
from steppy.adapter import Adapter, E
from steppy.utils import get_logger
import joblib
########
logger = get_logger()
# Coefficent Functions
class LambertScaler(BaseEstimator, TransformerMixin):

    def __init__(self, tol=1.22e-2, max_iter=1000):
        self.coefs_ = []  # Store tau for each transformed variable
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        logger.debug("******* FIT LambertScaler")  
        return self
    
    def transform(self, X):
        logger.debug("******* TRANSFORM LambertScaler")
        return ()
    

    def load(self, filepath):
        self = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self, filepath)