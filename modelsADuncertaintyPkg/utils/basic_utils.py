#Syngenta Open Source release: This file is part of code developed in the context of a Syngenta funded collaboration with the University of Sheffield: "Improved Estimation of Prediction Uncertainty Leading to Better Decisions in Crop Protection Research". In some cases, this code is a derivative work of other Open Source code. Please see under "If this code was derived from Open Source code, the provenance, copyright and license statements will be reported below" for further details.
#Copyright (c) 2021-2025  Syngenta
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#Contact: richard.marchese_robinson [at] syngenta.com
#==========================================================
#If this code was derived from Open Source code, the provenance, copyright and license statements will be reported below
#==========================================================
#######################
#Copyright (c)  2020-2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#######################
from collections import defaultdict
import pandas as pd
from pandas.testing import assert_frame_equal
import itertools
import scipy as sp
import numpy as np
import os,shutil
import pickle
from inspect import getframeinfo,currentframe
import time
from ..utils.time_utils import basic_time_task


def geometricMean(list_of_floats):
    #-----------------------------------
    assert type([])==type(list_of_floats),type(list_of_floats)
    assert 0 == len([v for v in list_of_floats if not type(0.5) == type(v)]),[type(v) for v in list_of_floats]
    #-----------------------------------
    return sp.stats.mstats.gmean(np.array(list_of_floats))

def check_geometricMean():
    examples_dict = defaultdict(dict)
    ############################
    
    examples_dict[1]['input'] = [2.0,4.0]
    
    examples_dict[1]['expected'] = 2.83
    
    ###########################
    
    for eg in examples_dict.keys():
        print('check_geometricMean: checking example %d' % eg)
        
        res = geometricMean(examples_dict[eg]['input'])
        
        assert round(res,2) == round(examples_dict[eg]['expected'],2),"example %d: res=%f,expected=%f" % (eg,res,examples_dict[eg]['expected'])
        
        print('check_geometricMean: CHECKED example %d' % eg)

#check_geometricMean()

def isInteger(n):
    
    if (n//1) == n:
        return True
    else:
        return False

def check_isInteger():
    assert isInteger(1.0)
    
    assert not isInteger(1.32)

#check_isInteger()

def getKeyFromValue(my_dict,my_value):
    
    values = [my_dict[k] for k in my_dict.keys()]
    
    assert len(values) == len(set(values)),"Can only get key from value if these are unique. Values contain duplicates: {}".format(findDups(values))
    
    keys = [k for k in my_dict.keys()]
    
    my_index = values.index(my_value)
    
    return keys[my_index]

def check_getKeyFromValue():
    
    assert 'a' == getKeyFromValue({'b':58,'a':1.5},1.5)
    
    next_failed = False
    
    try:
        assert 'a' == getKeyFromValue({'b':1.5,'a':1.5},1.5)
    except Exception:
        next_failed = True
    
    assert next_failed
    
#check_getKeyFromValue()

def findDups(a_list):
    e2c = defaultdict(int)
    
    for e in a_list:
        e2c[e] += 1
    
    return [e for e in e2c.keys() if e2c[e] > 1]

def check_findDups():
    assert [] == findDups(['a','b'])
    assert ['b'] == findDups(['b','a','b'])

#check_findDups()

def convertDefaultDictDictIntoDataFrame(dd,col_name_for_first_key='ID'):
    return pd.DataFrame(dd).transpose().reset_index().rename({'index':col_name_for_first_key},inplace=False,axis=1)

def check_convertDefaultDictDictIntoDataFrame():
    
    dd=defaultdict(dict)
    dd[1]['In']=True
    dd[1]['p']=0.05
    dd[2]['In']=False
    dd[2]['p']=0.01
    
    expected_df = pd.DataFrame({'ID':[1,2],'In':[True,False],'p':[0.05,0.01]})
    
    df = convertDefaultDictDictIntoDataFrame(dd)
    
    assert_frame_equal(df,expected_df,check_dtype=False) #This originally failed due to attribute "dtype" being different: https://www.skytowner.com/explore/pandas_dataframe_transpose_method
    
#check_convertDefaultDictDictIntoDataFrame()

def flatten(list_of_lists):
    #===============================
    assert type([]) == type(list_of_lists),type(list_of_lists)
    not_lists = [l for l in list_of_lists if not type([])==type(l)]
    assert 0 == len(not_lists),"{}".format(not_lists)
    #==================================
    
    return list(itertools.chain(*list_of_lists))

def check_flatten():
    assert [1,3,5,4]==flatten([[1,3],[5,4]])

#check_flatten()


def neverEndingDefaultDict():
    #Inspired by the recursion logic here, but uses an explict recursive function rather than lambda: https://twitter.com/raymondh/status/343823801278140417
    return defaultdict(neverEndingDefaultDict)

def returnDefDictOfLists():
    return defaultdict(list)

def doubleDefaultDictOfLists(returnDefDictOfLists=returnDefDictOfLists):
    return defaultdict(returnDefDictOfLists)

def createOrReplaceDir(dir_):
    try:
        os.mkdir(dir_)
    
    except Exception as err:
        print('{} : {} already exists! Hence, remove before creating an empty version ...'.format(err,dir_))
        
        shutil.rmtree(dir_, ignore_errors=True)
        
        os.mkdir(dir_)

def reportSubDirs(dir_,absolute_names=True):
    #https://docs.python.org/3/library/os.html#os.scandir
    
    subdirs = []
    
    with os.scandir(dir_) as ScanDirObj:
        for dir_or_file_obj in ScanDirObj:
            if dir_or_file_obj.is_dir():
                if absolute_names:
                    dir_or_file_name = os.path.sep.join([dir_,dir_or_file_obj.name])
                else:
                    dir_or_file_name = dir_or_file_obj.name
                subdirs.append(dir_or_file_name)
    
    return subdirs

def report_subdirs_recursively(dir_,absolute_names=True):

    all_subdirs = []

    for subdir in reportSubDirs(dir_=dir_,absolute_names=absolute_names):

        all_subdirs += [subdir]

        all_subdirs += report_subdirs_recursively(dir_=subdir,absolute_names=absolute_names)


    return all_subdirs

def this_data_frame_contains_missing_values(df):
    assert isinstance(df,pd.DataFrame),type(df)

    intermediate_res_1 = df.isnull()
    has_some_missing_values = intermediate_res_1.values.any()

    return has_some_missing_values


def check_no_missing_values(df):
    
    has_some_missing_values = this_data_frame_contains_missing_values(df)
    
    
    if has_some_missing_values:
        raise Exception(f'columns with missing values={df.isnull().any()}')

def insert_list_values_into_list_of_lists(lol,my_list,pos=0):
    assert len(my_list)==len(lol)
    
    [lol[i].insert(pos,my_list[i]) for i in range(len(lol))] #THis will update each sub-list in place!
    
    assert len(my_list)==len(lol)
    
    return lol

def add_elements_to_start_of_list_of_lists(lol,my_list): #This appears to be needed because I observed insert_list_values_into_list_of_lists(...) appears to behave inconsistently at times with a large list?!
    assert len(my_list)==len(lol)
    
    res = [[my_list[i]]+lol[i] for i in range(0,len(my_list))]
    
    return res

def load_from_pkl_file(pkl_file):
    f_in = open(pkl_file,'rb')
    try:
        obj = pickle.load(f_in)
    finally:
        f_in.close()
    
    return obj

def create_pkl_file(pkl_file,obj):
    f_o = open(pkl_file,'wb')
    try:
        pickle.dump(obj,f_o)
    finally:
        f_o.close()

def get_pandas_df_row_as_df(pandas_df,row_index,monitor_time=False):
    if monitor_time:
        start = time.time()

    row_df = pandas_df.iloc[[row_index]].reset_index(drop=True,inplace=False)

    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start,units='seconds')
        del task,end,start

    return row_df

def finite_value_could_not_be_computed(v):
    #np.isinf(np.inf) = True AND np.isinf(-np.inf) = True AND (-np.inf < 0) = True [numpy version 1.21.5]; pd.isna(np.nan) = True [numpy version 1.21.5, pandas version 1.1.5]
    return (np.isinf(v) or pd.isna(v))

def report_name_of_function_where_this_is_called():
    #https://docs.python.org/3/library/inspect.html
    frame = currentframe()
    
    outer_frame = frame.f_back
    
    function_being_executed_in_outer_frame = outer_frame.f_code
    
    function_name = f'{function_being_executed_in_outer_frame.co_name}'
    
    return function_name

def numpy_array_contains_zeros(np_array):
    assert isinstance(np_array,np.ndarray),type(np_array)

    if np.all(np_array):
        return False
    else:
        return True

def sort_numpy_array_in_descending_order(np_array):
    #=======================
    #based upon https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding
    #array[i:j:k] means select the elements from indices i to j with step size k
    #:: means select all indices. k = -1 means return the elements in reverse order. :: means select all indices.
    #=======================
    assert isinstance(np_array,np.ndarray),type(np_array)
    k=-1
    np_array = np.sort(np_array)[::k]

    return np_array

    





