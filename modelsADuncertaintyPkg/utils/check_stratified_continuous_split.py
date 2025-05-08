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
############################
#Copyright (c) 2023 Syngenta
#contact richard.marchese_robinson [at] syngenta.com
#This file was adapted from the following on 02/11/23: https://github.com/DanilZherebtsov/verstack/blob/master/verstack/unittest/test_stratified_continuous_split.py
#############################
############################
# MIT License
#
# Copyright (c) 2020 DanilZherebtsov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
############################
import pytest
#import sys #RLMR: Not needed after moved outside of verstack package (due to tricky dependency conflicts on my machine)
#sys.path.append('../../')
import pandas as pd
from .stratified_continuous_split import scsplit
##############
#RLMR:
from .load_example_datasets_for_code_checks import generateExampleDatasetForChecking_Housing as generateExampleRegressionDataset
from .load_example_datasets_for_code_checks import Housing_y_name as y_col_regression_example
from .load_example_datasets_for_code_checks import generateExampleDatasetForChecking_BCWD as generateExampleClassificationDataset
from .load_example_datasets_for_code_checks import BCWD_y_name as y_col_class_example
from .ML_utils import checkTrainTestHaveUniqueDistinctIDs
###############
all_rnd_seeds_to_consider = [42,100,763]
acceptable_deviation = 0.2 #=5 c.f. original code, after convert 0.05 to true percentage difference

#============================
#RLMR: helper functions for main test_ (renamed check_) functions below

def check_y_labels_for_binary_classification(y_labels_list):
    unique_class_labels_list = list(set(y_labels_list))
    unique_class_labels_list.sort()
    assert [0,1]==unique_class_labels_list,f'unique_class_labels_list={unique_class_labels_list}'

def check_train_test_y_distributions_are_consistent(train_y,test_y,all_y,regression):
    assert isinstance(train_y,pd.Series)
    assert isinstance(test_y,pd.Series)
    
    for reference_y in [test_y,all_y]:

        percent_diff_in_mean_of_column_used_for_stratification = 100*(train_y.mean() - reference_y.mean()) / reference_y.mean()
    
        assert percent_diff_in_mean_of_column_used_for_stratification <= acceptable_deviation,f'percent_diff_in_mean_of_column_used_for_stratification={percent_diff_in_mean_of_column_used_for_stratification}' #RLMR: more informative AssertionError

        percent_diff_in_stdev_of_column_used_for_stratification = 100*(train_y.std() - reference_y.std()) / reference_y.std()

        assert percent_diff_in_stdev_of_column_used_for_stratification <= acceptable_deviation,f'percent_diff_in_stdev_of_column_used_for_stratification = {percent_diff_in_stdev_of_column_used_for_stratification}'

    if not regression:
        check_y_labels_for_binary_classification(train_y.tolist())
        check_y_labels_for_binary_classification(test_y.tolist())

def check_no_mols_are_lost_via_scplit_applied_to_df(train,test,no_orig_mols):
    print('Checking no molecules are dropped (df input)!')
    assert isinstance(train,pd.DataFrame)
    assert isinstance(test,pd.DataFrame)

    assert (train.shape[0]+test.shape[0]) == no_orig_mols,f"no_orig_mols={no_orig_mols},train.shape[0]={train.shape[0]},test.shape[0]={test.shape[0]}"


def check_no_mols_are_lost_via_scplit_applied_to_x_and_y(train_x, test_x, train_y, test_y,no_orig_mols):
    print('Checking no molecules are dropped (x,y input)!')
    assert isinstance(train_x,pd.DataFrame)
    assert isinstance(test_x,pd.DataFrame)
    assert isinstance(train_y,pd.Series)
    assert isinstance(test_y,pd.Series)

    assert train_x.shape[0] == train_y.shape[0]
    assert test_x.shape[0] == test_y.shape[0]

    assert (train_x.shape[0]+test_x.shape[0]) == no_orig_mols,f"no_orig_mols={no_orig_mols},train_x.shape[0]={train_x.shape[0]},test_x.shape[0]={test_x.shape[0]}"

def check_scplit_with_a_given_df(df,y_col,regression,id_col='ID',):
    all_y = df[y_col]

    no_orig_mols = df.shape[0]

    for seed in all_rnd_seeds_to_consider: #RLMR: check robustness
        train, test = scsplit(df, stratify=df[y_col], test_size = 0.5,random_state=seed,continuous=regression)
        
        assert isinstance(train,pd.DataFrame)
        assert isinstance(test,pd.DataFrame)
        
        train_y = train[y_col]
        test_y=test[y_col]
        
        checkTrainTestHaveUniqueDistinctIDs(train,test,id_col)

        check_train_test_y_distributions_are_consistent(train_y,test_y,all_y,regression)

        check_no_mols_are_lost_via_scplit_applied_to_df(train,test,no_orig_mols)

def check_scplit_with_a_given_df_when_supply_x_and_y(df,y_col,regression):
    X = df.drop(y_col,axis=1,inplace=False)
    y = df[y_col]
    all_y = y
    
    assert isinstance(X,pd.DataFrame)
    assert isinstance(y,pd.Series)

    no_orig_mols = X.shape[0]
    
    for seed in all_rnd_seeds_to_consider:
        #https://verstack.readthedocs.io/en/latest/#stratified-continuous-split
        train_x, test_x, train_y, test_y = scsplit(X, y, stratify = y,test_size = 0.5, random_state = seed, continuous=regression)
        
        assert isinstance(train_x,pd.DataFrame)
        assert isinstance(test_x,pd.DataFrame)
        
        
        check_train_test_y_distributions_are_consistent(train_y,test_y,all_y,regression)

        check_no_mols_are_lost_via_scplit_applied_to_x_and_y(train_x, test_x, train_y, test_y,no_orig_mols)
#==================================

#RLMR: main test_ (renamed check_) functions

def check_scsplit():
    df = generateExampleRegressionDataset() #RLMR: #pd.read_parquet('boston_train.parquet') #Do not want to rely upon more complext dependencies: "ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'."
    
    check_scplit_with_a_given_df(df=df,y_col=y_col_regression_example,regression=True)

def check_scsplit_with_separate_x_and_y():
    df = generateExampleRegressionDataset()
    
    check_scplit_with_a_given_df_when_supply_x_and_y(df=df,y_col=y_col_regression_example,regression=True)
    

def check_scsplit_with_categorical_dataset():
    df = generateExampleClassificationDataset()

    check_scplit_with_a_given_df(df,y_col=y_col_class_example,regression=False)

def check_scsplit_with_separate_x_and_categorical_y():
    df = generateExampleClassificationDataset()

    check_scplit_with_a_given_df_when_supply_x_and_y(df,y_col=y_col_class_example,regression=False)
