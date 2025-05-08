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
############################################################
#Copyright (c) 2022-2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#The function pred_ints(...) was taken and adapted by Zied Hosni as part of a Syngenta fundeded collaboration with the University of Sheffield
#Further adaptations were made by richard.marchese_robinson [at] syngenta.com:
#1. Make the prediction intervals symmetric (in some original trial runs, some forest predictions were identified to lie outside the confidence interval of tree predictions)
#2. v1 - > v2: Try to speed up this calculation
#This function was originally provided by Ando Saabas here: https://blog.datadive.net/prediction-intervals-for-random-forests/
#Provenance, copyright and license information for the original lines of code is provided below. Correspondence with Ando Saabas confirmed the copyright terms and that the code could be considered available under the MIT license.
#############################################################
#MIT License
#
#Copyright (c) 2015 Ando Saabas
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################
import numpy as np
import pandas as pd
import os,sys
from pytest import approx
from ..utils.ML_utils import get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance
from ..utils.basic_utils import get_pandas_df_row_as_df
from ..utils.efficient_forest_distribution_statistics import _return_random_forest_tree_predictions
from ..utils.ML_utils import compute_pred_intervals


def pred_ints_v1(model,X,y_pred,ci_percentage,ignore_user_warning):
    #'percentile' below is actually the confidence interval percentage, for the distribution of tree predictions
    percentile = ci_percentage

    list_of_all_trees = [tree for tree  in model.estimators_]
    
    err_down = []
    err_up = []
    
    for row_index in X.index.tolist():
        
        test_X_instance = get_pandas_df_row_as_df(pandas_df=X,row_index=row_index)
        
        preds = [get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance(tree=list_of_all_trees[tree_index],ensemble=model,tree_index=tree_index,test_X=test_X_instance,expected_type_of_test_X=pd.DataFrame,want_class_probs=False,ignore_user_warning=ignore_user_warning) for tree_index in range(0,len(list_of_all_trees))]
        
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    
    nrTestCases = len(y_pred)
    intervals = np.zeros((nrTestCases, 2))
    # Compute the intervals for each test case
    for k in range(0, nrTestCases):
        
        #========================================
        #Ensure symmetric prediction intervals centred on the prediction:
        width_of_PI = abs(err_up[k] - err_down[k])
        
        intervals[k, 0] = y_pred[k] - (width_of_PI/2)
        intervals[k, 1] = y_pred[k] + (width_of_PI/2)
        #==============================
        #RLMR: checks:
        info_string_in_case_checks_fail = f'y_pred[k]={y_pred[k]},intervals[k, 0]={intervals[k, 0]},intervals[k, 1]={intervals[k, 1]},err_down[k]={err_down[k]},err_up[k]={err_up[k]}'
        #Should always be true:
        assert err_down[k] <= err_up[k],info_string_in_case_checks_fail
        assert intervals[k, 0] <= intervals[k, 1],info_string_in_case_checks_fail
        #=============================
        #pytest approx(...) needed here after following error due to small differences being lost: 
        #assert intervals[k, 0] < intervals[k, 1],f'intervals[k, 0]={intervals[k, 0]},intervals[k, 1]={intervals[k, 1]},err_down[k]={err_down[k]},err_up[k]={err_up[k]}' AssertionError: intervals[k, 0]=5.579050000000004,intervals[k, 1]=5.579050000000004,err_down[k]=5.6,err_up[k]=5.6000000000000005
        
        if not err_down[k] == approx(err_up[k]): 
            assert intervals[k, 0] < intervals[k, 1],info_string_in_case_checks_fail
        else:
            assert intervals[k, 0] == approx(intervals[k, 1]),info_string_in_case_checks_fail
            print('SURPRISING: zero-sized prediction intervals with nRF!')
        #==============================
    
    return intervals

def get_confidence_interval_bounds_for_one_instance(tree_preds_for_instance,ci_percentage):
    lower = np.percentile(tree_preds_for_instance, (100 - ci_percentage) / 2. )
    upper = np.percentile(tree_preds_for_instance, 100 - (100 - ci_percentage) / 2.)
    #---------------------
    assert lower <= upper
    #---------------------
    return (lower,upper)

def get_confidence_interval_bounds(all_tree_preds_for_all_rows_in_X,ci_percentage):
    
    ci_bounds = np.apply_along_axis(get_confidence_interval_bounds_for_one_instance,0,all_tree_preds_for_all_rows_in_X,ci_percentage)
    #---------------------
    assert ci_bounds.shape[0] == 2
    assert ci_bounds.shape[1] == all_tree_preds_for_all_rows_in_X.shape[1]
    #---------------------
    err_down = ci_bounds[0].tolist()
    err_up = ci_bounds[1].tolist()

    return err_down,err_up

def get_half_size_of_prediction_intervals(err_down,err_up):
    return [abs(err_up[k] - err_down[k])/2 for k in range(len(err_down))]

def pred_ints_v2(model,X,y_pred,ci_percentage,ignore_user_warning):
    
    all_tree_preds_for_all_rows_in_X = _return_random_forest_tree_predictions(X,model.estimators_)
    
    err_down,err_up = get_confidence_interval_bounds(all_tree_preds_for_all_rows_in_X,ci_percentage)

    half_pred_interval_sizes = get_half_size_of_prediction_intervals(err_down,err_up)

    intervals = compute_pred_intervals(y_pred,half_pred_interval_sizes)
    
    return intervals

def pred_ints(model, X, y_pred, ci_percentage,ignore_user_warning=False,use_v1=False):
    #===================
    assert isinstance(X,pd.DataFrame),type(X)
    assert isinstance(y_pred,np.ndarray),type(y_pred)
    assert X.index.tolist()==list(range(X.shape[0])),X.index.tolist()
    #====================
    
    if use_v1:
        intervals = pred_ints_v1(model,X,y_pred,ci_percentage,ignore_user_warning)
    else:
        intervals = pred_ints_v2(model,X,y_pred,ci_percentage,ignore_user_warning)
        
    return intervals


