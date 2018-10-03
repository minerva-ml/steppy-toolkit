from pathlib import Path
import pandas as pd
import copy, random
import numpy as np
import pytest
# steppy imports
from steppy.base import Step, IdentityOperation, StepsError, make_transformer
from steppy.adapter import Adapter, E

#
import os,sys
#sys.path.append(os.path.abspath('../..'))
from toolkit.preprocessing.scalers import BoxCoxScaler,LambertScaler,S_Scaler

@pytest.fixture()
def Xneg():
    return np.array([
            [-4.2, -3.7, -8.9],
            [-3.1, -3.2, -0.5],
            [1.4, 0.9, 8.9],
            [5.8, 5.0, 2.4],
            [5.6, 7.8, 2.4],
            [0.1, 7.0, 0.2],
            [8.3, 1.9, 7.8],
            [3.8, 9.2, 2.8],
            [5.3, 5.7, 4.5],
            [6.8, 5.3, 3.2]
        ])

@pytest.fixture()
def sXn():
    return Xneg().shape

@pytest.fixture()
def Xzero():
    return np.array([
            [0.0, 3.7, 8.9],
            [3.1, 3.2, 0.5],
            [1.4, 0.9, 8.9],
            [5.8, 5.0, 2.4],
            [5.6, 7.8, 2.4],
            [0.1, 7.0, 0.2],
            [8.3, 1.9, 7.8],
            [3.8, 9.2, 2.8],
            [5.3, 5.7, 4.5],
            [6.8, 5.3, 3.2]
        ])

@pytest.fixture()
def X():
    return np.array([
            [4.2, 3.7, 8.9],
            [3.1, 3.2, 0.5],
            [1.4, 0.9, 8.9],
            [5.8, 5.0, 2.4],
            [5.6, 7.8, 2.4],
            [0.1, 7.0, 0.2],
            [8.3, 1.9, 7.8],
            [3.8, 9.2, 2.8],
            [5.3, 5.7, 4.5],
            [6.8, 5.3, 3.2]
        ])

@pytest.fixture()
def sX():
    return X().shape

@pytest.fixture()
def y():
    return np.array([4, 1, 1, 1, 1, 4, 2, 3, 2, 1])
@pytest.fixture()
def ystr():
    return np.array( ['x','y','z'])

@pytest.fixture()
def sy():
    return y().shape
@pytest.fixture()
def yz():
    return np.array([4, 0, 1, 1, 1, 4, 2, 3, 2, 1])
@pytest.fixture()
def syz():
    return yz().shape
@pytest.fixture()
def yn():
    return np.array([4, -1, 1, 1, 1, 4, 2, 3, 2, 1])
@pytest.fixture()
def syn():
    return yn().shape
@pytest.fixture()
def z():
    return pd.DataFrame(X(),columns=[cn()])
@pytest.fixture()
def cn():
    return ['Column_0','Column_1','Column_2']
@pytest.fixture()
def cnv():
    return ['x','y','z']

def test_df_Class_init_NoArg():
    with pytest.raises(TypeError):
        g = S_Scaler()
        
def test_df_Class_init_WrongScaler():
    with pytest.raises(ValueError):
        g = S_Scaler('MimMaxScaler')
        
# BoxCoxScaler unit tests
g = S_Scaler('BoxCoxScaler')
def test_df_BoxCox_instance_2d():
    assert g.fit(X()) == g
def test_df_BoxCox_2d_numpy():
    assert g.transform(X())['X'].shape == sX()
    
g = S_Scaler('BoxCoxScaler')
def test_df_BoxCox_instance_1d():
    assert g.fit(y()) == g
def test_df_BoxCox_1d_numpy():
    assert g.transform(y())['X'].shape[0] == sy()[0]

g = S_Scaler('BoxCoxScaler')    
#def test_df_BoxCox_zeroval_error():
#    with pytest.raises(ValueError):
#        g.fit(Xzero()) 
#def test_df_BoxCox_negval_error():
#    with pytest.raises(ValueError):
#        g.fit(Xneg())
def test_df_BoxCox_type_error():
    with pytest.raises(TypeError):
        g.fit(ystr())
        
# LambertScaler unit tests
g = S_Scaler('LambertScaler')
def test_df_Lambert_instance_2d():
    assert g.fit(X()) == g
def test_df_Lambert_2d_numpy():
    assert g.transform(X())['X'].shape == sX()
    
g = S_Scaler('LambertScaler')
def test_df_Lambert_instance_1d():
    assert g.fit(y()) == g
def test_df_Lambert_1d_numpy():
    assert g.transform(y())['X'].shape[0] == sy()[0]
    
g = S_Scaler('LambertScaler')   
def test_df_Lambert_type_error():
    with pytest.raises(TypeError):
        g.fit(ystr())

g = S_Scaler('LambertScaler')    
def test_df_Lambert_zeroval_1d():
    g.fit(yz())
    assert g.transform(yz())['X'].shape[0] == syz()[0]
def test_df_Lambert_negval_1d():
    g.fit(yn())
    assert g.transform(yn())['X'].shape[0] == syn()[0]
