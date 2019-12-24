from pathlib import Path
import pandas as pd
import copy, random
import numpy as np
import pytest
# steppy imports
from steppy.base import Step, IdentityOperation, StepsError, make_transformer
#ort Step, IdentityOperation, StepsError, make_transformer
from steppy.adapter import Adapter, E

#
import os,sys
from toolkit.preprocessing.toDataFrame import toDataFrame

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
def y():
    return np.array([4, 0, 0, 1, 0, 4, 2, 3, 2, 1])
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
        g = toDataFrame(1)

g = toDataFrame()
def test_df_2d_numpy():
    assert g.transform(X())['X'].columns[1] == cn()[1]
def test_df_1d_numpy():
    assert g.transform(y())['X'].columns[0] == cn()[0]   
def test_df_2d_named():
    assert g.transform(X(),cnv())['X'].columns[2] == cnv()[2]
# edge tests
def test_df_BlankList():
    with pytest.raises(ValueError):
        g.transform([])
def test_df_3dArray_numpy():
    with pytest.raises(ValueError):
        g.transform(np.reshape(X(),(-1,-1,1)))
#
def test_df_df():
    assert g.transform(pd.DataFrame(X(),columns=cnv()))['X'].columns[1] == cnv()[1]
