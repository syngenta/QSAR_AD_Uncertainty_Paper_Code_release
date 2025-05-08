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
########################################################
#Copyright (c) 2022-2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#The original version of this code file (RegressionICP.py) was developed at the Uppsala University and downloaded from https://github.com/pharmbio/SCPRegression [See below for the original copyright and license information.]
#Edits made, where necessary, by Zied Hosni (z.hosni [at] sheffield.ac.uk), whilst working on a Syngenta funded project
#Further edits made to support reuse of code within a Python package by Richard Marchese Robinson (richard.marchese_robinson [at] syngenta.com) #RLMR
####################################################
#Copyright (c) 2019-2022 Uppsala University
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#Kindly cite our paper:
#Gauraha, N. and Spjuth, O.
#Synergy Conformal Prediction for Regression
#Proceedings of the 10th International Conference on Pattern Recognition Applications and Methods. vol 1: ICPRAM, 212-221. (2021). DOI: #10.5220/0010229402120221
#http://dx.doi.org/10.5220/0010229402120221
#############################################
#Original code
# Author: Niharika Gauraha
# ICP: Inductive Conformal Prediction
#        for Regression using un-normalized
#        conformity measures
#############################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import sys
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression
from math import exp, ceil, floor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
import sklearn.base
import pandas as pd
import time
#======================
from ..utils.ML_utils import get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance
from ..utils.basic_utils import get_pandas_df_row_as_df
from ..utils.time_utils import basic_time_task
from ..utils.basic_utils import report_name_of_function_where_this_is_called
from ..utils.efficient_forest_distribution_statistics import _return_std
from ..utils.ML_utils import compute_pred_intervals
from ..utils.basic_utils import sort_numpy_array_in_descending_order
#======================

# Simple Conformal regression #RLMR: assuming RandomForestRegressor

def compute_tree_predictions_stdev_for_one_row_of_X(X,row_index,list_of_all_trees,RF_reg_model,ignore_user_warning,monitor_time=False):
    if monitor_time:
        start = time.time()

    test_X_instance = get_pandas_df_row_as_df(pandas_df=X,row_index=row_index,monitor_time=monitor_time)
        
    tree_preds = [get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance(tree=list_of_all_trees[tree_index],ensemble=RF_reg_model,tree_index=tree_index,test_X=test_X_instance,expected_type_of_test_X=pd.DataFrame,want_class_probs=False,ignore_user_warning=ignore_user_warning) for tree_index in range(0,len(list_of_all_trees))]
        
    y_std_ = np.std(tree_preds)

    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start,units='seconds')
        del task,end,start

    return y_std_

def compute_y_std_v1(X, numberTrees,RF_reg_model,ignore_user_warning=True,monitor_time=False):

    list_of_all_trees = [tree for tree  in RF_reg_model.estimators_]
    
    y_std = [compute_tree_predictions_stdev_for_one_row_of_X(X,row_index,list_of_all_trees,RF_reg_model,ignore_user_warning,monitor_time) for row_index in X.index.tolist()]

    return y_std

def compute_y_std_v2(X,RF_reg_model,ignore_user_warning):

    return _return_std(X,RF_reg_model.estimators_).tolist()


def compute_y_std(X, numberTrees,RF_reg_model,ignore_user_warning=True,monitor_time=False,use_v1=False): 
    ##############################################
    #Purpose of this function: Return mean and standard deviation of predictions across tress in Random Forest model - for all instances (compounds) corresponding to X
    #We need the standard deviation to provide the scaling factor for the nonconformity measures as per equations (1) and (2) of Svensson et al. (https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00054)
    ##############################################
    if monitor_time:
        start = time.time()
    #---------------------------------
    assert isinstance(X,pd.DataFrame),f"X must be a pandas Dataframe, not {type(X)}!"
    assert isinstance(RF_reg_model,RandomForestRegressor),f"type(RF_reg_model)={type(RF_reg_model)}!"
    assert numberTrees == len(RF_reg_model.estimators_),"numberTrees = {} len(RF_reg_model.estimators_) {}".format(numberTrees,len(RF_reg_model.estimators_))
    #---------------------------------

    if use_v1:
        y_std = compute_y_std_v1(X, numberTrees,RF_reg_model,ignore_user_warning)
    else:
        y_std = compute_y_std_v2(X,RF_reg_model,ignore_user_warning)
    
    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start,units='seconds')
        del task,end,start

    return y_std


def lamda_exp_std(y_std, w=1,monitor_time=False):
    #scaling factor for the non-conformity measures as per equation (2) of Svensson et al. (https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00054)
    
    if monitor_time:
        start = time.time()

    lamda = np.exp(w * np.array(y_std))

    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start)
        del task,end,start

    return lamda

def computeNonConformityScores(pred, y, lamda_calib,monitor_time=False):
    #######################################
    # Compute normalized non-conformity scores
    #scaling factor for the non-conformity measures as per equation (1) of Svensson et al. (https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00054)
    #######################################
    if monitor_time:
        start = time.time()
    
    res = np.abs(y - pred)

    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start)
        del task,end,start

    return res/lamda_calib

def find_nonconformity_score_for_test_cmpds(calibration_set_nonconformity_scores,sigma,reading='revisited'):
    #################################
    #RLMR: The selection of the non-conformity score for a test compound was adjusted to match our reading of the literature (Papdopoulos et al. (2011) [https://doi.org/10.1613/jair.3198])
    # - see section 4 pages 822 - 824
    ################################
    #===============================
    assert isinstance(calibration_set_nonconformity_scores,np.ndarray),f'type(confScore)={type(calibration_set_nonconformity_scores)}'
    assert isinstance(sigma,float),f'type(sigma)={type(sigma)}'
    #================================

    ##############################################
    #RLMR: changes from original code:
    #confScore = calibration_set_nonconformity_scores
    #epsilon = sigma
    #
    #Sort the confidence scores
    #confScore = np.sort(confScore)
    #################
    #RLMR: Moved outside loop over test instances inside computeInterval(...), to speed up calculations!
    #################
    # Compute threshold for split conformal, at level alpha.
    #n = len(confScore)
    ##########################
    #if (ceil((n) * epsilon) <= 1): 
    #    q = np.inf
    #else:
    #    q= (confScore[ceil((n) * (1 - epsilon))])
    #
    #non_conformity_score_to_return = q
    #########################
    ##############################################
    
    if 'revisited' == reading:
        #################################################
        #Following notation of paper as much as possible
        #(Papdopoulos et al. (2011) [https://doi.org/10.1613/jair.3198])
        # - see section 4 pages 822 - 824, especially Algorithm 2
        ##################################################

        descending_calibration_set_nonconformity_scores = sort_numpy_array_in_descending_order(calibration_set_nonconformity_scores)

        q_meaning_calibration_set_size_here = len(descending_calibration_set_nonconformity_scores)

        #########################
        #I think this is the correct interpretation of equation (18)
        #Here, "s" is the index from 1 ... q_meaning_calibration_set_size of the calibration set non-conformity score to return, after they are sorted in descending order
        s= floor(sigma*(q_meaning_calibration_set_size_here+1))
        #######################

        if s >=1 and s <= q_meaning_calibration_set_size_here:
            array_index_of_non_conformity_score_to_return = (s-1)

            non_conformity_score_to_return = descending_calibration_set_nonconformity_scores[array_index_of_non_conformity_score_to_return]
        elif s > q_meaning_calibration_set_size_here:
            print(f'WARNING: ICP: s > q_meaning_calibration_set_size_here is not explicitly covered by Papdopoulos et al. (2011)')
            #This is not explicitly covered by the paper, but this would imply that a non-conformity score less than the minimum  for the calibration set was selected .... but the paper indicates that only non-conformity scores from the calibration set should be selected!
            non_conformity_score_to_return = min(descending_calibration_set_nonconformity_scores)
        elif 0 == s:
            print(f'WARNING: ICP: s = 0 is not explicitly covered by Papdopoulos et al. (2011)')
            #This is not explicitly covered by the paper, but p.823 indicates that the largest alpha_i (non-conformity score) should be selected
            #If it is assumed that the true y for a new example could give rise to an infinitely large non-conformity score, I can see why setting this to infinity here was considered in the original code we inherited
            #However, that would surely violate the assumption that the new examples follow the same distribution as the true training and calibration set
            non_conformity_score_to_return = max(descending_calibration_set_nonconformity_scores)
        else:
            raise Exception(f's={s}?!')

        
    else:
        raise Exception(f'Unexpected reading = {reading}')

    return non_conformity_score_to_return


def estimate_residuals_for_test_cmpds(confScore,lamda_test,epsilon):
    #=======================
    assert isinstance(lamda_test,np.ndarray),f'type(lambda_test)={type(lamda_test)}'
    assert isinstance(confScore,np.ndarray),f'type(confScore)={type(confScore)}'
    #=======================
    
    q = find_nonconformity_score_for_test_cmpds(calibration_set_nonconformity_scores=confScore,sigma=epsilon)

    q*=lamda_test

    return q

def computeInterval(confScore, testPred, epsilon, lamda_test):
    if confScore is None:
        sys.exit("\n NULL model \n")
    
    #=================
    assert isinstance(confScore,(np.ndarray,pd.Series))
    #=================

    confScore = np.array(confScore)

    q = estimate_residuals_for_test_cmpds(confScore,lamda_test,epsilon)

    intervals = compute_pred_intervals(y_pred=testPred,half_pred_interval_sizes=q)

    return intervals

def fit_RF(X_train, y_train, nrTrees=100, global_random_seed=3): #RLMR: adapt this to only fit RF model!
    clf = RandomForestRegressor(n_estimators=nrTrees, random_state=global_random_seed,max_features=(1.0/3))
    clf.fit(X_train, y_train)
    return clf
