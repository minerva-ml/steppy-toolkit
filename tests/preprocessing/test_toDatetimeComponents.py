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
from toolkit.preprocessing.toDatetimeComponents import toDatetimeComponents



@pytest.fixture()
def cn():
    return ['integer_0','float_1','float_2']
@pytest.fixture()
def df_small():
    df = pd.DataFrame({
    'datetime_S_column': ['11/11/1906','11/11/1906','11/11/1906 12:13:14','11/11/1906','11/11/1906'], 
    'datetime_NY_column': ['11/11/1906','11/11/1907','11/11/1908','11/11/1909','11/11/1910'],
    'datetime_ND_column': ['11/11/1906','11/12/1907','11/13/1908','11/14/1909','11/15/1910'],
    'datetime_NA_column': ['11/11/1906',np.nan,'11/11/1908','11/11/1909','11/11/1910'],
    'datetime_EU_column': ['21.01.1906','21.11.1907','14.11.1908','13.11.1909','11.10.1910'],
    'obj_column': ['red','blue','green','pink',np.nan],
    'boolean': [True,False,True,False,True], 
    'integer': [1,2,33,44,34],
    'float': [1.,2.,35.,46,.37]
    
})
    return(df)

@pytest.fixture()
def df_small_NFeatures():
    return(df_small().shape[1])

@pytest.fixture()
def NComponentFeatures():
    dt_features = ['Year', 'Month', 'Week', 'Day','Dayofweek'
                      , 'Dayofyear','Elapsed','Is_month_end'
                      , 'Is_month_start', 'Is_quarter_end'
                      , 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    return(len(dt_features))

@pytest.fixture()
def df_big():
    ld = ['11/11/1906','11/11/1906','11/11/1906 12:13:14','11/11/1906','11/11/1906']*200000
    dfb = pd.DataFrame({'datetime_S_column':ld})
    return(dfb)
@pytest.fixture()
def df_big_dt():
    return( pd.DataFrame(pd.to_datetime(dfb['datetime_S_column'], infer_datetime_format=True)))

@pytest.fixture()
def df_small_NA():
    return (df_small().replace(to_replace='11/11/1906', value=np.nan))

def test_df_toDatetimeComponents_Class_init_WrongArg():
    with pytest.raises(TypeError):
        g = toDatetimeComponents('1','2')
        
g = toDatetimeComponents( dt_features = ['Year', 'Month', 'Week', 'Day','Dayofweek'
                      , 'Dayofyear','Elapsed','Is_month_end'
                      , 'Is_month_start', 'Is_quarter_end'
                      , 'Is_quarter_start', 'Is_year_end', 'Is_year_start'])

def test_toDatetimeComponents_passed_arg_type_error():
    with pytest.raises(TypeError):
        g.fit([1,2,3])
        
def test_toDatetimeComponents_passed_N_DT_Components():
    assert (g.transform(df_small(),'datetime_S_column')['X'].shape[1] == df_small_NFeatures()+NComponentFeatures()-1 )
    
def test_toDatetimeComponents_passed_N_DT_Components_drop_False():
    assert (g.transform(df_small(),'datetime_S_column',drop=False)['X'].shape[1] == df_small_NFeatures()+NComponentFeatures())

def test_toDatetimeComponents_with_NA_Values_2nd_Arg():
    with pytest.raises(ValueError):
        g.transform(df_small_NA(),'datetime_S_column')
        
h = toDatetimeComponents( dt_features = ['Year'])

def test_toDatetimeComponents_passed_N_DT_Components_1():
    assert (h.transform(df_small(),'datetime_S_column')['X'].shape[1] == df_small_NFeatures()+1-1 )
    
def test_toDatetimeComponents_passed_N_DT_Components_1_drop_False():
    assert (h.transform(df_small(),'datetime_S_column',drop=False)['X'].shape[1] == df_small_NFeatures()+1)
    
j = toDatetimeComponents( dt_features = ['Dayofyear','Elapsed','Is_month_end'])

def test_toDatetimeComponents_passed_N_DT_Components_3():
    assert (j.transform(df_small(),'datetime_S_column')['X'].shape[1] == df_small_NFeatures()+3-1 )
    
def test_toDatetimeComponents_passed_N_DT_Components_3_drop_False():
    assert (j.transform(df_small(),'datetime_S_column',drop=False)['X'].shape[1] == df_small_NFeatures()+3)