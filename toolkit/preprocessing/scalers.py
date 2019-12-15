import pandas as pd
import copy, random
import numpy as np
from numba import jit
# sklearn imports
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.utils import get_obj_cols, convert_input
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import QuantileTransformer
import sklearn.preprocessing.data as skpr
# scipy imports
from scipy.special import lambertw
from scipy.stats import kurtosis, norm, rankdata, boxcox
from scipy.optimize import fmin  # TODO: Explore efficacy of other opt. methods
# steppy immports
from steppy.base import Step, BaseTransformer, make_transformer
from steppy.adapter import Adapter, E
from steppy.utils import get_logger
import joblib
# inconsistent naming is historical.Scalar steppy transforner supported
# introduce __ScalerDict__ for Steppy universe
__author__ = 'Bruce_H_Cottman'
__license__ = 'MIT License'
__ScalerDict__ = {}
__ScalerDict__['StandardScaler'] = StandardScaler
__ScalerDict__['MinMaxScaler'] = MinMaxScaler
__ScalerDict__['Normalizer'] = Normalizer
__ScalerDict__['MaxAbsScaler'] = MaxAbsScaler
__ScalerDict__['RobustScaler'] = RobustScaler
__ScalerDict__['QuantileTransformer'] = QuantileTransformer
#__ScalerDict__['self'] = skpr.__all__,
#######
logger = get_logger()
# Coefficent Functions
@jit
def w_d(z, delta):
    # Eq. 9
    if delta < 1e-6: return z
    return(np.sign(z) * np.sqrt(np.real(lambertw(delta * z ** 2)) / delta))
@jit
def w_t(y, tau):
    # Eq. 8
    return(tau[0] + tau[1] * w_d((y - tau[0]) / tau[1], tau[2]))
@jit
def inverse(x, tau):
    # Eq. 6
    u = (x - tau[0]) / tau[1]
    return(tau[0] + tau[1] * (u * np.exp(u * u * (tau[2] * 0.5))))

def igmm(y, tol=1.22e-4, max_iter=100):
    # Infer mu, sigma, delta using IGMM in Alg.2, Appendix C
    if np.std(y) < 1e-4:
        return np.mean(y), np.std(y).clip(1e-4), 0
    delta0 = delta_init(y)
    tau1 = (np.median(y), np.std(y) * (1. - 2. * delta0) ** 0.75, delta0)
    for k in range(max_iter):
        tau0 = tau1
        z = (y - tau1[0]) / tau1[1]
        delta1 = delta_gmm(z)
        x = tau0[0] + tau1[1] * w_d(z, delta1)
        mu1, sigma1 = np.mean(x), np.std(x)
        tau1 = (mu1, sigma1, delta1)

        if np.linalg.norm(np.array(tau1) - np.array(tau0)) < tol:
            break
        else:
            if k == max_iter - 1:
                raise ValueError("Warning: No convergence after %d iterations. Increase max_iter." % max_iter)
    return tau1

def delta_gmm(z):
    # Alg. 1, Appendix C
    delta0 = delta_init(z)

    def func(q):
        u = w_d(z, np.exp(q))
        if not np.all(np.isfinite(u)):
            return 0.
        else:
            k = kurtosis(u, fisher=True, bias=False)**2
            if not np.isfinite(k) or k > 1e10:
                return 1e10
            else:
                return k

    res = fmin(func, np.log(delta0), disp=0)
    return np.around(np.exp(res[-1]), 6)

def delta_init(z):
    gamma = kurtosis(z, fisher=False, bias=False)
    with np.errstate(all='ignore'):
        delta0 = np.clip(1. / 66 * (np.sqrt(66 * gamma - 162.) - 6.), 0.01, 0.48)
    if not np.isfinite(delta0):
        delta0 = 0.01
    return delta0
########
# BoxCoxScaler

class BoxCoxScaler(BaseEstimator, TransformerMixin):
    """
    BoxCoxScaler method to Gaussianize heavy-tailed data. The Box-Cox transformation is to be used to:
        - stabilize variance
        - remove right tail skewness
    
    There are two major limitations of this approach:
        - only applies to positive data 
        - transforms into normal gaussian form only data with a Gausssian heavy right-hand tail.

    Attributes:
        None
    """

    def __init__(self):
        self.coefs_ = []  # Store tau for each transformed variable
#        self.verbose = verbose

    def _reset(self):
        self.coefs_ = []  # Store tau for each transformed variable

    def fit(self, X):
        """
        Args:
            X (array-like): Caclulates BoxCox coefficents for each variable/column in X.
            
        Returns: self
        
        Raises:
            ValueError will result of not 1-D or 2-D numpy array, list or Pandas Dataframe.
            
            ValueError will result if has any negative value elements.
            
            TypeError will result if not float or integer type.
        """
        
        logger.debug("******* FIT BoxCoxScaler")
        self._reset()
        X = np.asarray(X)
        if not np.asarray(X).dtype in [np.float,np.int]:
            raise TypeError("Passed X array elements must be integer or float.")
        elif X.min() < 0:
            raise ValueError("BoxCox fit does not work on any X array elements that are a negative value.")


        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        elif len(X.shape) != 2:
            raise ValueError("Passed X array elements must be  1-d or 2-d list or numpt array or pandas DataFrame.")

        for x_i in X.T:
            self.coefs_.append(boxcox(x_i)[1])
        return self

    def transform(self, X):
        """
        Transform data using a previous ``.fit`` that calulates BoxCox coefficents for data.
                
        Args:
            X (array-like): Transform  numpy 1-D or 2-D array distribution by BoxCoxScaler.

        Returns:
            X: tranformed by BoxCox and returns 2-d numpy array.numpy 1-D or 2-D arr
        
        """
        
        logger.debug("TRANSFORM BoxCoxScaler")
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        elif len(X.shape) != 2:
            raise ValueError("Data should be a 1-d list of samples to transform or a 2d array with samples as rows.")
        if X.shape[1] != len(self.coefs_):
            raise ValueError("%d variables in test data, but %d variables were in training data." % (X.shape[1], len(self.coefs_)))
        return(np.array([boxcox(x_i, lmbda=lmbda_i) for x_i, lmbda_i in zip(X.T, self.coefs_)]).T)

         
# is there a way to specify with step or does does step need enhancing?
    def inverse_transform(self, y):
        """
        Recover original data from BoxCox transformation.
        """
        
        return (np.array([(1. + lmbda_i * y_i) ** (1./lmbda_i) for y_i, lmbda_i in zip(y.T, self.coefs_)]).T)


__ScalerDict__['BoxCoxScaler'] = BoxCoxScaler
########
# LambertScaler

class LambertScaler(BaseEstimator, TransformerMixin):
    """
    LambertScaler method to Gaussianize heavy-tailed data.
        - a one-parameter (``w_t``) family based on Lambert's W function that removes heavy tails from data. 
        - the ``LambertScaler`` has no difficulties with negative values.
         
         Args:
            tol(optional): (default) 0.01, ``w_t`` is calulated internally to ``tol`` erance.
            
            max_iter(optional): (default) 1000. ``max_iter`` is maximum number of  iterations that will be used to determine ``w_t`` to ``tol``.
            
        Returns: none
        
    """

    def __init__(self, tol=0.001, max_iter=1000):
        self.coefs_ = []  # Store tau for each transformed variable
        self.tol = tol
        self.max_iter = max_iter
 
    def _reset(self):
        self.coefs_ = []  # Store tau for each transformed variable
        
    def fit(self, X):
        """
        Fit a one-parameter (``w_t``) family based on Lambert's W function that removes heavy tails from data. ``w_t`` will be determined iteratively (``max_iter``) to tolerance (``tol``) for each feature(column). Because of the iterative calulation of ``w_t`` ``LambertScaler`` can be slower by a factor of 10 than other scalers. 
                        
        Args:
            X (array-like):   Determine ``w_t`` of X, a 1-d or 2d numpy array. Shape assumed for array is (n-row) or (n-row,n-column).
        
        Raises:
            ValueError will result of not 1-D or 2-D numpy array, list or Pandas Dataframe.
            
            TypeError will result if not float or integer type.
            
        Returns: self
        """
        
        logger.debug("FIT LambertScaler")
        # Reset internal state before fitting
        self._reset()
        X = np.asarray(X)
        if not np.asarray(X).dtype in [np.float,np.int]:
            raise TypeError("passed X array elements must be integer or float.")
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        elif len(X.shape) != 2:
            raise ValueError("Passed X array elements must be  1-d or 2-d list or numpt array or pandas DataFrame.")
        for x_i in X.T:
            self.coefs_.append(igmm(x_i, tol=self.tol, max_iter=self.max_iter))
            
        return self
    
    def transform(self, X):
        """
        Transform data using a previously learned Lambert coefficents for data.
        
        Args:
            X (array-like):  1-d or 2d numpy array. 

            
        Returns: 
            X: tranformed by  Lambert's W function and returns 2-d numpy array.

        """
        
        logger.debug("TRANSFORM LambertScaler")
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        elif len(X.shape) != 2:
            raise ValueError ("Data should be a 1-d list of samples to transform or a 2d array with samples as rows.")
        if X.shape[1] != len(self.coefs_):
            raise ValueError("%d variables in test data, but %d variables were in training data." % (X.shape[1], len(self.coefs_)))
        return (np.array([w_t(x_i, tau_i) for x_i, tau_i in zip(X.T, self.coefs_)]).T)

    def inverse_transform(self, y):
        """
        Reverse transformation of data from Lambert transformation.
                
        Args:
            y (array-like):  1-d or 2d numpy array. 

            
        Returns: y, transformed back from Lambert's W function.
        
        """
        
        return (np.array([inverse(y_i, tau_i) for y_i, tau_i in zip(y.T, self.coefs_)]).T)

           
__ScalerDict__['LambertScaler'] = LambertScaler


########
# Steppy_Scaler

class S_Scaler(BaseTransformer):
    """
    **Scaling**

    Scaling means converting floating-point feature values from their natural range (for example, 100 to 900) into a standard range (for example, 0 to 1 or -1 to +1). If a feature set consists of only a single feature, then scaling provides little to no practical benefit. If, however, a feature set consists of multiple features, then feature scaling provides the following benefits:
        - Helps gradient descent converge more quickly. (i.e. deep learning sic.)
        - Helps avoid the "NaN trap," in which one number in the model becomes a NaN (e.g., when a value exceeds the floating-point precision limit during training), and—due to math operations—every other number in the model also eventually becomes a NaN.
        - Helps the model learn appropriate weights for each feature. Without feature scaling, the model will pay too much attention to the features having a wider range.
        - You don't have to give every floating-point feature exactly the same scale. Nothing terrible will happen if Feature A is scaled from -1 to +1 (Min/Max) while Feature B is scaled from -3 to +3 (normalizion, or Z-scaling). However, your model will react poorly if Feature B is scaled from 5000 to 100000.

    **Gaussian Scaling**

    This family of methods applys smooth, invertible transformations to some univariate data so that the distribution of the transformed data is as Gaussian as possible. This would/could be a pre-processing step for feature(s) upstream of futher data mangaling.  
        - A standard pre-processing step is to "whiten" data by subtracting the mean and scaling it to have standard deviation 1. Gaussianized data has these properties and more.
        - Robust statistics / reduce effect of outliers. Lots of real world data exhibits long tails. For machine learning, the small number of examples in the tails of the distribution can have a large effect on results. Gaussianized data will "squeeze" the tails in towards the center.
        - Gaussian distributions are very well studied with many unique properties (because it is a well behaved function...( that dates back to Gauss :)
        - Many statistical models assume Gaussianity (trees being the exception)
        """
 
    def __init__(self,encoderKey,*args,**kwargs):
        super().__init__()
        if encoderKey in __ScalerDict__:  
            Encoder = __ScalerDict__[encoderKey](*args)
        else: raise ValueError('No Scaler or Encoder named:',encoderKey)
        self.encoder = Encoder
        
    def fit(self, X):
        """
        Args:
            X (array-like): Predictions of different models for the labels.
            
        Returns: self
        """
        
        logger.debug("FIT Steppy_Scaler")
        self.encoder.fit(X)
        return self
    
    def transform(self, X):
        """
        Args:
            X (array-like): Predictions of different models for the labels.
            
        Returns: self
        """

        logger.debug("TRANSFORM Steppy_Scaler")
        X = self.encoder.transform(X)     
        return {'X': X}

    def load(self, filepath):
        self.encoder = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.encoder, filepath)
########
"""
    **References**

    :Title: **The Lambert Way to Gaussianize heavy tailed data with the inverse of Tukey's h as a special case**
    :Author: Georg M. Goerg
    :URL: https://arxiv.org/pdf/1010.2265.pdf
"""
