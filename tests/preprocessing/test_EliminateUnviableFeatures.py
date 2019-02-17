from pathlib import Path
import pandas as pd
import copy, random
import numpy as np
import pytest
# steppy imports
#
from steppy.base import Step, IdentityOperation, StepsError, make_transformer
from steppy.adapter import Adapter, E

#
import os,sys
from toolkit.preprocessing.EliminateUnviableFeatures import EliminateUnviableFeatures

@pytest.fixture()
def aX():
    return np.array([
            [4, 3.7, 8.9],
            [3, 3.2, 0.5],
            [1, 0.9, 8.9],
            [5, 5.0, 2.4],
            [5, 7.8, 2.4],
            [0, 7.0, 0.2],
            [8, 1.9, 7.8],
            [3, 9.2, 2.8],
            [5, 5.7, 4.5],
            [6, 5.3, 3.2]
        ])

@pytest.fixture()
def sX():
    return aX().shape

@pytest.fixture()
def cn():
    return ['integer_0','float_1','float_2']
@pytest.fixture()
def df_type():
    return pd.DataFrame(aX(),columns=[cn()])
@pytest.fixture()
def cno():
    return ['integer_10','float_11','float_12']
@pytest.fixture()
def df_typeo():
    return pd.DataFrame(aX(),columns=[cno()])
@pytest.fixture()
def df_typeNA():
    return ((pd.DataFrame(aX(),columns=[cn()])).replace(to_replace=5, value=np.nan))
@pytest.fixture()
def df_type_low_V():
    return ((pd.DataFrame(aX(),columns=[cn()]))\
           .replace(to_replace=0, value=0)\
           .replace(to_replace=4, value=1)
           .replace(to_replace=3, value=1)\
           .replace(to_replace=5, value=1)\
           .replace(to_replace=8, value=1)\
           .replace(to_replace=6, value=1))
@pytest.fixture()
def df_type_low_V11():
    df = df_type_low_V().copy()
    df.loc[10,:] = [1, 7.0, 0.2]
    return (df)
@pytest.fixture()
def df_type_SV():
    return (df_type_low_V11().replace(to_replace=0, value=1))

from sklearn.datasets import load_boston
boston = load_boston()

def df_City():
    City = pd.DataFrame(boston.data, columns = boston.feature_names )
#City = City[['CRIM', 'INDUS','NOX','TAX','B']]
    City['MEDV'] = boston.target
    return(City)

def test_df_EliminateUnviableFeatures_Class_init_WrongArg():
    with pytest.raises(TypeError):
        g = EliminateUnviableFeatures('bad')
        
g = EliminateUnviableFeatures()
def test_df_EliminateUnviableFeatures_passed_arg_type_error():
    with pytest.raises(TypeError):
        g.fit(aX)
        
def test_df_EliminateUnviableFeatures_passed_arg_type_correct():
    assert (g.transform(df_type(),df_type())['X'].iloc[:,0] == df_type().iloc[:,0]).all()
    
def test_df_EliminateUnviableFeatures_Orthognal():
    with pytest.raises(TypeError):
        g.transform(df_type(),df_typeo())
        
def test_df_EliminateUnviableFeatures_Orthognal_City():
    with pytest.raises(TypeError):
        g.transform(df_type(),df_City())
    
def test_df_EliminateUnviableFeatures_FitToSelf():
    assert g.fit() == g    
    
def test_df_EliminateUnviableFeatures_with_NA_Values_1stArg():
    with pytest.raises(ValueError):
        g.transform(df_typeNA(),df_typeo())
        
def test_df_EliminateUnviableFeatures_with_NA_Values_2nd_Arg():
    with pytest.raises(ValueError):
        g.transform(df_type(),df_typeNA())
    
#def test_df_Eliminate_Duplicate_Features():
#    assert (g.transform(pd.concat([df_City(),df_City()],axis=1),df_City())['X'] == df_City()).all().all()
    
def test_Eliminate_Single_Unique_Value_Features():
    assert (g.transform(df_type_SV(),df_type_low_V())['X'].iloc[:,0] == df_type_SV().iloc[:,1]).all().all()  
    
def test_Eliminate_Low_Variance_Features_LE_10_values():
    assert (g.transform(df_type_low_V(),df_type_low_V())['X'].iloc[:,0] == df_type_low_V().iloc[:,0]).all().all()   
    
def test_Eliminate_Low_Variance_Features_GT_10_values():
    assert (g.transform(df_type_low_V11(),df_type_low_V())['X'].iloc[:,0] == df_type_low_V11().iloc[:,1]).all().all()  