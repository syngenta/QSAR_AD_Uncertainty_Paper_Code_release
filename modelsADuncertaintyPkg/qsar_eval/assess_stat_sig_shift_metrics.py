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
#Copyright (c) 2023-2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#########################################################
#########################################################
#**********************************************************
#Purpose of the code in this file:
#Scenario: We have two sets of compounds (e.g. subsets of a test set lying inside or outside the applicability domain) for which we can compute some model performance metric and the difference in this metric between the two sets, i.e. the 'shift-metric' = 'shift-metric' (see below). [This is possible as each set matches a unique compound ID to an endpoint value (y) and a predicted endpoint value (y_pred), where y and y_pred could be continuous (regression) or categorical (classification) and the categorical class labels can be represented as 0,1, ... etc.]
#Objective: We wish to determine whether we can reject the null-hypothesis that the difference in the performance metric (the 'shift-metric') arose due to chance, i.e. would the probability of getting a metric at least this extreme be less than some significance-level (sig_level) if we had randomly partitioned the compounds into two sets with the same number and endpoint distribution as the original split.
#N.B. (1) A one-tail test would be appropriate if we assume the sets are inside and outside a split defined by some applicability domain method, for which we expect a priori that the performance should be better inside the domain and, hence, we can provide a shift-metric defined as metric[inside] - metric[outside]. HOWEVER, we could not use this to conclude that anomalous results (performance inside was worse than outside) were statistical flukes!
#N.B. (2) To ensure the same distribution for the random partitions, we need to sample the compounds from different categories independently. For the continuous endpoints, this means we need to discretize the endpoint values in the two subsets consistently first.
#**********************************************************
import re,os,sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import sklearn
from collections import defaultdict

from modelsADuncertaintyPkg.qsar_eval.reg_perf_pred_stats import rmse
from .aggregate_p_vals import aggregate_p_vals
from .enforce_minimum_no_instances import size_of_inputs_for_stats_is_big_enough
from .all_key_class_stats_and_plots import get_experi_class_1_probs
from .all_key_reg_stats_and_plots import getIntervalsWidthOfInterest
from . import all_key_class_stats_and_plots as ClassEval
from . import all_key_reg_stats_and_plots as RegEval
from ..utils.basic_utils import findDups,neverEndingDefaultDict,returnDefDictOfLists,doubleDefaultDictOfLists
import seaborn as sb
sb.set_style("ticks")
import matplotlib.pyplot as pp


def estimated_probabilities_are_too_close(class_1_probs,experi_vals, delta=0.05):
    delta_calib_plot_experi_probs = get_experi_class_1_probs(class_1_probs,experi_vals,delta_prob=delta)
    
    if 1 == len(set(delta_calib_plot_experi_probs)):
        return True
    else:
        return False

def consistentlyDiscretizeEndpointValues(y_set_one_list,y_set_two_list,n_bins=5,rand_seed=42):
    #-------------------------------
    assert isinstance(y_set_one_list,list),f"type(y_set_one_list)={type(y_set_one_list)}. value={y_set_one_list}"
    assert isinstance(y_set_two_list,list),f"type(y_set_two_list)={type(y_set_two_list)}. value={y_set_two_list}"
    #--------------------------------
    
    y_set_one_ready = np.array(y_set_one_list).reshape(-1,1)
    
    y_set_two_ready = np.array(y_set_two_list).reshape(-1,1)
    
    if float('.'.join(sklearn.__version__.split('.')[:2])) < 1.1:
        discretizer=KBinsDiscretizer(n_bins=n_bins,encode='ordinal',strategy='uniform')
    else:
        discretizer=KBinsDiscretizer(n_bins=n_bins,encode='ordinal',strategy='uniform',random_state=rand_seed) #Is this actually needed for reproducible results unless subsample=int and strategy='quantile'?
    
    discretizer.fit(y_set_one_ready)
    
    y_set_one_discretized = discretizer.transform(y_set_one_ready).flatten().tolist()
    
    y_set_two_discretized = discretizer.transform(y_set_two_ready).flatten().tolist()
    
    return y_set_one_discretized,y_set_two_discretized,discretizer

def checkUniqueNonOverlappingSubsetIds(dict_of_raw_subset_results):
    all_ids = []
    for subset in dict_of_raw_subset_results.keys():
        subset_test_ids = dict_of_raw_subset_results[subset]['subset_test_ids']
        
        #------------------------------------------------
        if not isinstance(subset_test_ids,list): raise Exception('subset_test_ids must be of type=list! Currently it is of type {}'.format(type(subset_test_ids)))
        #------------------------------------------------
        
        all_ids += subset_test_ids
    
    #=======================
    if not len(all_ids)==len(set(all_ids)): raise Exception('Duplicate IDs={}'.format(findDups(all_ids)))
    #=======================

def create_subset_sig_level_2_prediction_interval_val(dict_of_raw_subset_results,subset,index):
    
    all_sig_levels = list(dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'].keys())
    
    subset_sig_level_2_prediction_interval_val = dict(zip(all_sig_levels,[dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'][sig_level][index] for sig_level in all_sig_levels]))
    
    return subset_sig_level_2_prediction_interval_val

def getDictOfMergedIdsMatched2RawResults(dict_of_raw_subset_results,type_of_modelling='binary_class'):
    
    ################################
    ###Function input description###
    ################################
    #dict_of_raw_subset_results = dictionary matching subset name, for two subsets (e.g. 'Inside' vs. 'Outside' the applicability domain [AD]), to the following - where all iterables are in the same order as subset_test_ids:
    #subset_test_ids
    #subset_test_y (1s and 0s if type_of_modelling='binary_class', or continuous values if type_of_modelling='regression')
    ##if type_of_modelling='binary_class':
    ###subset_probs_for_class_1
    ###subset_predicted_y
    ##if type_of_modelling='regression':
    ###subset_test_predictions
    ###subset_sig_level_2_prediction_intervals
    #################################
    ###Function output description###
    #################################
    #dict_of_merged_ids_matched_to_raw_res = dictionary matching each of the IDs (merged across both subsets), to the following:
    #subset_test_y_val (1 or 0 if type_of_modelling='binary_class', or a continuous value if type_of_modelling='regression')
    ##if type_of_modelling='binary_class':
    ###subset_prob_for_class_1_val
    ###subset_predicted_y_val
    ##if type_of_modelling='regression':
    ###subset_test_prediction_val
    ###subset_sig_level_2_prediction_interval_val
    ##############################
    
    #------------------------------
    if not 2 == len(list(dict_of_raw_subset_results.keys())): raise Exception('There should only be two subset names, e.g. "inside AD" and "outside AD". But, the raw results dictionary you have provided contains names for these subsets={}'.format(list(dict_of_raw_subset_results.keys())))
    #-----------------------------
    
    dict_of_merged_ids_matched_to_raw_res = defaultdict(dict)
    
    checkUniqueNonOverlappingSubsetIds(dict_of_raw_subset_results)
    
    for subset in dict_of_raw_subset_results.keys():
        subset_test_ids = dict_of_raw_subset_results[subset]['subset_test_ids']
        
        subset_test_y = dict_of_raw_subset_results[subset]['subset_test_y']
        
        #------------------------------------------------
        if not isinstance(subset_test_y,list): raise Exception('subset_test_y must be of type=list! Currently it is of type {}'.format(type(subset_test_y)))
        #------------------------------------------------
        
        #----------------------------------------
        if 'binary_class' == type_of_modelling:
            type_of_modelling_specific_var_names = ['subset_probs_for_class_1','subset_predicted_y']
        elif 'regression' == type_of_modelling:
            type_of_modelling_specific_var_names = ['subset_test_predictions','subset_sig_level_2_prediction_intervals']
        else:
            raise Exception('Unsupported type_of_modelling = {}'.format(type_of_modelling))
        
        for var_name in type_of_modelling_specific_var_names:
            if not 'subset_sig_level_2_prediction_intervals' == var_name:
                if not isinstance(dict_of_raw_subset_results[subset][var_name],list): raise Exception('{} must be a list! Currently it is of type {}'.format(var_name,type(dict_of_raw_subset_results[subset][var_name])))
            else:
                if not isinstance(dict_of_raw_subset_results[subset][var_name],dict): raise Exception('{} must be a dict! Currently it is of type {}'.format(var_name,type(dict_of_raw_subset_results[subset][var_name])))
                
                subset_sig_level_2_prediction_intervals = dict_of_raw_subset_results[subset][var_name]
                
                for sig_level in subset_sig_level_2_prediction_intervals.keys():
                    if not isinstance(subset_sig_level_2_prediction_intervals[sig_level],np.ndarray): raise Exception('{} must be a numpy.ndarray! Currently it is of type {}'.format("subset_sig_level_2_prediction_intervals[sig_level]",type(subset_sig_level_2_prediction_intervals[sig_level])))
        #----------------------------------------
        
        for index in range(0,len(subset_test_ids)):
            dict_of_merged_ids_matched_to_raw_res[subset_test_ids[index]]['subset_test_y_val'] = subset_test_y[index]
            
            if 'binary_class' == type_of_modelling:
                dict_of_merged_ids_matched_to_raw_res[subset_test_ids[index]]['subset_prob_for_class_1_val'] = dict_of_raw_subset_results[subset]['subset_probs_for_class_1'][index]
                
                dict_of_merged_ids_matched_to_raw_res[subset_test_ids[index]]['subset_predicted_y_val'] = dict_of_raw_subset_results[subset]['subset_predicted_y'][index]

            elif 'regression' == type_of_modelling:
                dict_of_merged_ids_matched_to_raw_res[subset_test_ids[index]]['subset_test_prediction_val'] = dict_of_raw_subset_results[subset]['subset_test_predictions'][index]
                
                dict_of_merged_ids_matched_to_raw_res[subset_test_ids[index]]['subset_sig_level_2_prediction_interval_val'] = create_subset_sig_level_2_prediction_interval_val(dict_of_raw_subset_results,subset,index)
            else:
                raise Exception('Unsupported type_of_modelling = {}'.format(type_of_modelling))

    return dict_of_merged_ids_matched_to_raw_res


def setRandGeneratorSeed(rand_seed):
    #This random generator needs to be consistent with the random generator used by getSubsetIdsForCurrentRandSplit(...)
    np.random.seed(rand_seed)

def getClass2Ids(all_strat_y_vals,all_ids,type_of_modelling):
    
    #====================================
    if not isinstance(all_strat_y_vals,list): raise Exception('type(all_strat_y_vals)={}'.format(all_strat_y_vals))
    if not isinstance(all_ids,list): raise Exception('type(all_ids)={}'.format(all_ids))
    if not len(all_strat_y_vals)==len(all_ids): raise Exception('len(all_strat_y_vals)={},len(all_ids)={}'.format(len(all_strat_y_vals),len(all_ids)))
    #====================================
    
    class2Ids = defaultdict(list)
    
    for index in range(0,len(all_strat_y_vals)):
        class2Ids[all_strat_y_vals[index]].append(all_ids[index])
    
    #=============================
    if 'binary_class' == type_of_modelling:
        if not 2 == len(list(class2Ids.keys())): raise Exception('list(class2Ids.keys())={}'.format(list(class2Ids.keys())))
    #=============================
    
    return class2Ids

def getSubsetIdsForCurrentRandSplit(dict_of_subset_results_ready_for_strat_rand_splits,subset_1_name,subset_2_name,strat_rand_split_y_name,
                                    rand_split_no,type_of_modelling, max_while_loop=10,delta_calib_prob=0.05,metrics_of_interest=[]):
    #############

    if type_of_modelling =="regression":
        subset_1_predicted_vals_of_interest = dict_of_subset_results_ready_for_strat_rand_splits[subset_1_name]["subset_test_predictions"]
        subset_2_predicted_vals_of_interest = dict_of_subset_results_ready_for_strat_rand_splits[subset_2_name]["subset_test_predictions"]
    elif type_of_modelling =="binary_class":
        subset_1_predicted_vals_of_interest = dict_of_subset_results_ready_for_strat_rand_splits[subset_1_name]["subset_probs_for_class_1"]
        subset_2_predicted_vals_of_interest = dict_of_subset_results_ready_for_strat_rand_splits[subset_2_name]["subset_probs_for_class_1"]
    else:
        raise Exception(f"type of modelling={type_of_modelling}")
    ##################

    subset_1_strat_y_vals = dict_of_subset_results_ready_for_strat_rand_splits[subset_1_name][strat_rand_split_y_name]
    subset_2_strat_y_vals = dict_of_subset_results_ready_for_strat_rand_splits[subset_2_name][strat_rand_split_y_name]
    
    #------------------------
    if not isinstance(subset_1_strat_y_vals,list): raise Exception('type(subset_1_strat_y_vals)={},subset_1_strat_y_vals={}'.format(type(subset_1_strat_y_vals),subset_1_strat_y_vals))
    if not isinstance(subset_2_strat_y_vals,list): raise Exception('type(subset_2_strat_y_vals)={},subset_2_strat_y_vals={}'.format(type(subset_2_strat_y_vals),subset_2_strat_y_vals))
    #-----------------------
    
    subset_1_ids = dict_of_subset_results_ready_for_strat_rand_splits[subset_1_name]['subset_test_ids']
    subset_2_ids = dict_of_subset_results_ready_for_strat_rand_splits[subset_2_name]['subset_test_ids']
    
    #---------------------------
    if not isinstance(subset_1_ids,list): raise Exception(type(subset_1_ids))
    if not isinstance(subset_2_ids,list): raise Exception(type(subset_2_ids))
    #---------------------------
    
    all_ids = subset_1_ids+subset_2_ids
    ##########
    all_strat_y_vals = subset_1_strat_y_vals + subset_2_strat_y_vals
    id_to_y_val_map = {id_: y_val for id_, y_val in zip(all_ids, all_strat_y_vals)}
    ##########

    ##########
    all_pred_y_vals = subset_1_predicted_vals_of_interest+subset_2_predicted_vals_of_interest
    id_to_pred_y_val_map = {id_: y_val for id_, y_val in zip(all_ids, all_pred_y_vals)}
    ##########

    ##########
    #all_probs_vals = subset_1_predicted_vals_of_interest + subset_2_predicted_vals_of_interest
    #id_to_probs_val_map = {id_: y_val for id_, y_val in zip(all_ids, all_pred_y_vals)}
    ##########
    #---------------------------
    assert len(all_ids)==len(set(all_ids)),"Duplicates = {}".format(findDups(all_ids))
    #----------------------------
    
    class2Ids = getClass2Ids(all_strat_y_vals,all_ids,type_of_modelling)

    #########

    sufficient_diversity = False
    while_loop_counter = 0
    while not sufficient_diversity:
        while_loop_counter += 1
        if while_loop_counter > max_while_loop:
            break
        rand_subset_1_ids = []

        for class_label in class2Ids.keys():
            class_specific_ids = np.asarray(class2Ids[class_label])
            size_sub_1 = len([v for v in subset_1_strat_y_vals if v == class_label])

            class_specific_inside_ad_ids = np.random.choice(class_specific_ids, size=size_sub_1, replace=False).tolist()
            rand_subset_1_ids += class_specific_inside_ad_ids

        rand_subset_2_ids = [id_ for id_ in all_ids if id_ not in rand_subset_1_ids]

        # Check for diversity in generated ids
        rand_subset_1_ys = [id_to_y_val_map[id_] for id_ in rand_subset_1_ids]
        rand_subset_2_ys = [id_to_y_val_map[id_] for id_ in rand_subset_2_ids]

        rand_subset_1_pred_ys = [id_to_pred_y_val_map[id_] for id_ in rand_subset_1_ids]
        rand_subset_2_pred_ys = [id_to_pred_y_val_map[id_] for id_ in rand_subset_2_ids]

        if len(set(rand_subset_1_ys)) > 1 and len(set(rand_subset_2_ys)) > 1 and len(set(rand_subset_1_pred_ys)) > 1 and len(set(
                rand_subset_2_pred_ys)) > 1:
            #Technically, the previous statement might not be relevant if we were ONLY interested in RMSE or MAD, but this is very unlikely!
            if type_of_modelling == "regression":
                sufficient_diversity = True
            elif type_of_modelling =="binary_class":
                if 0 < len(set(metrics_of_interest).intersection(set(['R2 (cal)','Pearson coefficient (cal)','Spearman coefficient (cal)']))):
                    #remember pred_ys are actually class_1_probabilities
                    if not (estimated_probabilities_are_too_close(class_1_probs=rand_subset_1_pred_ys,experi_vals=rand_subset_1_ys,delta=delta_calib_prob) or estimated_probabilities_are_too_close(class_1_probs=rand_subset_2_pred_ys,experi_vals=rand_subset_2_ys,delta=delta_calib_prob)):
                        sufficient_diversity = True
                else:
                    sufficient_diversity = True
            else:
                raise Exception(f"Unexpected type of modelling={type_of_modelling}.")

        else:
            print("Random subset created without sufficient diversity.")
    
    #########
    #---------------------------
    new_all_ids = rand_subset_1_ids+rand_subset_2_ids
    
    all_ids.sort()
    new_all_ids.sort()
    
    assert all_ids==new_all_ids,"all_ids = {} vs. new_all_ids = {}".format(all_ids,new_all_ids)
    #----------------------------
    
    ##########
    # Extract y values for each subset
    y_values_subset_1 = [id_to_y_val_map[id_] for id_ in rand_subset_1_ids]
    y_values_subset_2 = [id_to_y_val_map[id_] for id_ in rand_subset_2_ids]

    # Extract pred y values for each subset
    pred_y_values_subset_1 = [id_to_pred_y_val_map[id_] for id_ in rand_subset_1_ids]
    pred_y_values_subset_2 = [id_to_pred_y_val_map[id_] for id_ in rand_subset_2_ids]

    if not while_loop_counter > max_while_loop:
        # Assertion to confirm diversity in y values
        if type_of_modelling =="regression":
            assert len(set(y_values_subset_1)) > 1 and len(set(y_values_subset_2)) > 1 and len(set(pred_y_values_subset_1)) > 1 and len(set(
            pred_y_values_subset_2)) > 1, "Lack of diversity in y values for generated subsets"
        elif type_of_modelling == "binary_class":
            #remember that predicted ys actually mean the probabilities for class 1 here.
            if 0 < len(set(metrics_of_interest).intersection(set(['R2 (cal)','Pearson (cal)','Spearman (cal)']))):
                assert not (estimated_probabilities_are_too_close(class_1_probs=pred_y_values_subset_1,experi_vals=y_values_subset_1,delta=delta_calib_prob) or estimated_probabilities_are_too_close(class_1_probs=pred_y_values_subset_2,experi_vals=y_values_subset_2,delta=delta_calib_prob)), "Class 1 probabilities are too close."
                
            assert len(set(y_values_subset_1)) > 1 and len(set(y_values_subset_2)) > 1 and len(set(pred_y_values_subset_1)) > 1 and len(set(pred_y_values_subset_2)) > 1, "Lack of diversity in y values for generated subsets"
            
            
        else:
            raise Exception(f"Unexpected type of modelling={type_of_modelling}.")
    else:
        print("WARNING: We could not find a random subset with sufficient diversity - some metrics will fail! However, this may make no difference if the shift-metric could not be computed for the corresponding metrics for the original split!")

    #print("y_values_subset_2", y_values_subset_2)
    ##########
    return rand_subset_1_ids,rand_subset_2_ids


def addSubsetIdsForCurrentRandSplit(dict_of_splits_matched_to_subset_ids,dict_of_subset_results_ready_for_strat_rand_splits,subset_1_name,subset_2_name,strat_rand_split_y_name,rand_split_no,type_of_modelling,delta_calib_prob=0.05,metrics_of_interest=[]):
    
    rand_subset_1_ids,rand_subset_2_ids = getSubsetIdsForCurrentRandSplit(dict_of_subset_results_ready_for_strat_rand_splits,subset_1_name,subset_2_name,strat_rand_split_y_name,rand_split_no,type_of_modelling,delta_calib_prob=delta_calib_prob,metrics_of_interest=metrics_of_interest)
    
    split_name = 'Random-{}'.format(rand_split_no)
    
    dict_of_splits_matched_to_subset_ids[split_name][subset_1_name] = rand_subset_1_ids
    
    dict_of_splits_matched_to_subset_ids[split_name][subset_2_name] = rand_subset_2_ids
    
    return dict_of_splits_matched_to_subset_ids


def addSubsetIdsForOriginalSplit(dict_of_splits_matched_to_subset_ids,dict_of_raw_subset_results,subset_1_name,subset_2_name):
    
    split_name = 'Original'
    
    for subset in [subset_1_name,subset_2_name]:
        dict_of_splits_matched_to_subset_ids[split_name][subset] = dict_of_raw_subset_results[subset]['subset_test_ids']
    
    return dict_of_splits_matched_to_subset_ids

def define_no_bins_for_regression_discretization(y_set_one_list,default=5):
    assert isinstance(y_set_one_list,list),type(y_set_one_list)

    if len(list(set(y_set_one_list))) < default:
        if len(list(set(y_set_one_list))) >= 2:
            return len(list(set(y_set_one_list)))
        else:
            return 2
    else:
        return default

def getReadyForStratRandSplit(dict_of_raw_subset_results,strat_rand_split_y_name,type_of_modelling,subset_1_name,subset_2_name,default_no_bins_for_reg_strat=5):
    
    dict_of_subset_results_ready_for_strat_rand_splits = defaultdict(dict)
    
    y_set_one_list = dict_of_raw_subset_results[subset_1_name]['subset_test_y']
    
    y_set_two_list = dict_of_raw_subset_results[subset_2_name]['subset_test_y']
    ############
    #Debug:
    #print(f'y_set_one_list={y_set_one_list}')
    #print(f'y_set_two_list={y_set_two_list}')
    #############

    #print("type_of_modelling=",type_of_modelling)
    if 'binary_class' == type_of_modelling:
        y_set_one_discretized = y_set_one_list
        y_set_two_discretized = y_set_two_list
    elif 'regression' == type_of_modelling:
        n_bins = define_no_bins_for_regression_discretization(y_set_one_list,default=default_no_bins_for_reg_strat)

        y_set_one_discretized,y_set_two_discretized,discretizer = consistentlyDiscretizeEndpointValues(y_set_one_list,y_set_two_list,n_bins)
    else:
        raise Exception(f'Unrecognised type_of_modelling={type_of_modelling}')
    
    ############
    #Debug:
    #print(f'y_set_one_discretized={y_set_one_discretized}')
    #print(f'y_set_two_discretized={y_set_two_discretized}')
    #############
        
    dict_of_subset_results_ready_for_strat_rand_splits[subset_1_name][strat_rand_split_y_name] = y_set_one_discretized
    
    dict_of_subset_results_ready_for_strat_rand_splits[subset_2_name][strat_rand_split_y_name] = y_set_two_discretized
    
    for subset in [subset_1_name,subset_2_name]:
        for object_name in dict_of_raw_subset_results[subset].keys():
            dict_of_subset_results_ready_for_strat_rand_splits[subset][object_name] = dict_of_raw_subset_results[subset][object_name]
    
    return dict_of_subset_results_ready_for_strat_rand_splits

def getDictOfSplitsMatchedToSubsetIDs(dict_of_raw_subset_results,metrics_of_interest,subset_1_name='Inside',subset_2_name='Outside',type_of_modelling='binary_class',no_rand_splits=100,strat_rand_split_y_name='strat_y',rand_seed=42,delta_calib_prob=0.05):
    #
    #See def getDictOfMergedIdsMatched2RawResults(...): for a description of dict_of_raw_subset_results
    
    dict_of_splits_matched_to_subset_ids = defaultdict(dict)
    
    dict_of_splits_matched_to_subset_ids = addSubsetIdsForOriginalSplit(dict_of_splits_matched_to_subset_ids,dict_of_raw_subset_results,subset_1_name,subset_2_name)
    
    dict_of_subset_results_ready_for_strat_rand_splits = getReadyForStratRandSplit(dict_of_raw_subset_results,strat_rand_split_y_name,type_of_modelling,subset_1_name,subset_2_name)
    
    setRandGeneratorSeed(rand_seed)
    
    for rand_split_no in range(0,no_rand_splits):
        dict_of_splits_matched_to_subset_ids = addSubsetIdsForCurrentRandSplit(dict_of_splits_matched_to_subset_ids,dict_of_subset_results_ready_for_strat_rand_splits,subset_1_name,subset_2_name,strat_rand_split_y_name,rand_split_no,type_of_modelling,delta_calib_prob=delta_calib_prob,metrics_of_interest=metrics_of_interest)
    
    
    return dict_of_splits_matched_to_subset_ids

def create_new_subset_sig_level_2_prediction_intervals(dict_of_merged_ids_matched_to_raw_res,subset_ids):
    all_sig_levels = list(dict_of_merged_ids_matched_to_raw_res[subset_ids[0]]['subset_sig_level_2_prediction_interval_val'].keys())
    
    new_subset_sig_level_2_prediction_intervals = {}
    
    for sig_level in all_sig_levels:
        new_subset_sig_level_2_prediction_intervals[sig_level] = np.array([dict_of_merged_ids_matched_to_raw_res[id_]['subset_sig_level_2_prediction_interval_val'][sig_level] for id_ in subset_ids])
    
    return new_subset_sig_level_2_prediction_intervals
    
def getThisSplitsRawRes(dict_of_merged_ids_matched_to_raw_res,subset_1_ids,subset_2_ids,subset_1_name='Inside',subset_2_name='Outside',type_of_modelling='binary_class'):
    subset_matched_to_raw_res = defaultdict(dict)
    
    
    for subset_name in [subset_1_name,subset_2_name]:
        if subset_name == subset_1_name:
            subset_ids = subset_1_ids
        elif subset_name == subset_2_name:
            subset_ids = subset_2_ids
        else:
            raise Exception('Unrecognised subset name = {}'.format(subset_name))
        
        subset_matched_to_raw_res[subset_name]['subset_test_y_vals'] = [dict_of_merged_ids_matched_to_raw_res[id_]['subset_test_y_val'] for id_ in subset_ids]
        
        if 'binary_class' == type_of_modelling:
            subset_matched_to_raw_res[subset_name]['subset_probs_for_class_1'] = [dict_of_merged_ids_matched_to_raw_res[id_]['subset_prob_for_class_1_val'] for id_ in subset_ids]
            
            subset_matched_to_raw_res[subset_name]['subset_predicted_y'] = [dict_of_merged_ids_matched_to_raw_res[id_]['subset_predicted_y_val'] for id_ in subset_ids]
            #print("subset_matched_to_raw_res[subset_name]['subset_predicted_y']",subset_matched_to_raw_res[subset_name]['subset_predicted_y'])
        elif 'regression' == type_of_modelling:
            subset_matched_to_raw_res[subset_name]['subset_test_predictions'] = [dict_of_merged_ids_matched_to_raw_res[id_]['subset_test_prediction_val'] for id_ in subset_ids]
            
            subset_matched_to_raw_res[subset_name]['subset_sig_level_2_prediction_intervals'] = create_new_subset_sig_level_2_prediction_intervals(dict_of_merged_ids_matched_to_raw_res,subset_ids)
        else:
            raise Exception('Unsupported type of modelling={}'.format(type_of_modelling))
    
    return subset_matched_to_raw_res

def computeMetricsOfInterestForCurrentSplit(dict_of_merged_ids_matched_to_raw_res,metrics_of_interest,subset_1_ids,subset_2_ids,subset_1_name='Inside',subset_2_name='Outside',type_of_modelling='binary_class',metrics_include_delta_calibration_plot_metrics=True,metrics_include_reg_uncertainty=True,delta_probs=[0.05],sig_level_of_interest=0.32,metrics_for_which_shift_metric_cannot_be_computed_for_orig_split=[],dummy_value_if_shift_metric_cannot_be_computed=None):
    
    subset_matched_to_raw_res = getThisSplitsRawRes(dict_of_merged_ids_matched_to_raw_res,subset_1_ids,subset_2_ids,subset_1_name,subset_2_name,type_of_modelling)
    #
    dict_of_metrics_matched_to_subset_vals = defaultdict(dict)

    #================================================
    if len(metrics_of_interest) == len(set(metrics_of_interest).intersection(set(metrics_for_which_shift_metric_cannot_be_computed_for_orig_split))):
        for subset_name in [subset_1_name,subset_2_name]:
            for metric in metrics_of_interest:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = dummy_value_if_shift_metric_cannot_be_computed
        return dict_of_metrics_matched_to_subset_vals
    #================================================
    
    for subset_name in [subset_1_name,subset_2_name]:

        if 'binary_class' == type_of_modelling:
            experi_class_labels=subset_matched_to_raw_res[subset_name]['subset_test_y_vals']
            
            class_1_probs_in_order=subset_matched_to_raw_res[subset_name]['subset_probs_for_class_1']
            
            y_pred = subset_matched_to_raw_res[subset_name]['subset_predicted_y']
            

            #====================================
            if metrics_include_delta_calibration_plot_metrics:
                CalibMetrics = ClassEval.computeDeltaCalibrationPlot(class_1_probs_in_order, delta_probs, experi_class_labels, path='',skip_delta_plot_image=True,include_Spearman_pvalue=True)

                rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal = CalibMetrics[0]
                
            #=====================================

        elif 'regression' == type_of_modelling:
            test_y = subset_matched_to_raw_res[subset_name]['subset_test_y_vals']
            
            e = test_y
            
            test_predictions = subset_matched_to_raw_res[subset_name]['subset_test_predictions']
            
            p = test_predictions
            
            sig_level_2_prediction_intervals = subset_matched_to_raw_res[subset_name]['subset_sig_level_2_prediction_intervals']
            
            #===============================
            if metrics_include_reg_uncertainty: #We should not need to compute these all if we are only interested in some metrics .... but this makes the code simpler to write for now!
                validity, efficiency,ECE_new,ENCE,errorRate_s,scc = RegEval.getRelevantRegUncertainyMetrics(y_test=test_y, testPred=test_predictions,sig_level_2_prediction_intervals=sig_level_2_prediction_intervals,sig_level_of_interest=sig_level_of_interest)
            
            #================================
        else:
            raise Exception('Unsupported type of modelling={}'.format(type_of_modelling))
        
        for metric in metrics_of_interest:
            #==========================
            if metric in metrics_for_which_shift_metric_cannot_be_computed_for_orig_split:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = dummy_value_if_shift_metric_cannot_be_computed
                continue
            #==========================
            
            if 'Balanced Accuracy' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = ClassEval.computeBA(experi_class_labels=experi_class_labels, y_pred=y_pred)
            
            elif 'MCC' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = ClassEval.computeMCC(experi_class_labels=experi_class_labels, y_pred=y_pred)
            
            elif 'AUC' == metric:
                
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = ClassEval.computeAUC_TwoCategories(experi_class_labels=experi_class_labels, class_1_probs_in_order=class_1_probs_in_order)
            
            elif 'Kappa' == metric:
                
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = ClassEval.computeKappa(experi_class_labels=experi_class_labels, y_pred=y_pred)
            
            elif 'Stratified Brier Score' == metric:
            
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] =  ClassEval.computeStratifiedBrier_TwoCategories(class_1_probs_in_order=class_1_probs_in_order,experi_class_labels=experi_class_labels)
            
            elif 'RMSE (cal)' == metric:
                
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = rmseCal
            
            elif 'R2 (cal)' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = coeffOfDeterminationCal
                
            elif 'Pearson coefficient (cal)' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = PearsonCoeffCal
            
            elif 'Spearman coefficient (cal)' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = SpearmanCoeffCal
            
            elif 'RMSE' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = RegEval.RegPred.rmse(e,p)
            
            elif 'R2' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = RegEval.RegPred.coeffOfDetermination(e,p)
            
            elif 'Pearson coefficient' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = RegEval.RegPred.PearsonCoeff(e,p)
            
            elif 'Spearman coefficient' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = RegEval.RegPred.SpearmanCoeff(e,p)
    
            elif 'Validity' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = validity
            
            elif 'Efficiency' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = efficiency
            
            elif 'ECE' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = ECE_new
            
            elif 'ENCE' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = ENCE
            
            elif 'Spearman coefficient (PIs vs. residuals)' == metric:
                dict_of_metrics_matched_to_subset_vals[metric][subset_name] = scc
            else:
                raise Exception('Unrecognised metric : {}'.format(metric))
        
        
    
    return dict_of_metrics_matched_to_subset_vals

def computeShiftMetricsOfInterestForCurrentSplit(dict_of_metrics_matched_to_subset_vals,subset_1_name='Inside',subset_2_name='Outside',metrics_for_which_shift_metric_cannot_be_computed_for_orig_split=[],dummy_value_if_shift_metric_cannot_be_computed=None):
    
    dict_of_metrics_matched_to_shift_metric_vals = {}
    
    for metric in dict_of_metrics_matched_to_subset_vals.keys():
        
        subset_1_val = dict_of_metrics_matched_to_subset_vals[metric][subset_1_name]
        subset_2_val = dict_of_metrics_matched_to_subset_vals[metric][subset_2_name]

        #==========================================
        if metric in metrics_for_which_shift_metric_cannot_be_computed_for_orig_split:
            dict_of_metrics_matched_to_shift_metric_vals[metric] = dummy_value_if_shift_metric_cannot_be_computed
            continue
        #=========================================

        if not (pd.isna(subset_1_val) or pd.isna(subset_2_val)):

            if not (np.isinf(subset_1_val) and np.isinf(subset_2_val)):
                dict_of_metrics_matched_to_shift_metric_vals[metric] = subset_1_val - subset_2_val
            else:
                #np.(inf-inf) = nan
                dict_of_metrics_matched_to_shift_metric_vals[metric] = np.nan
                print(f'metric={metric} is infinitely large inisde and outside the domain. This shift-metric is not a number and will be ignored!')
            #print(f"metric={metric}, dict_of_metrics_matched_to_subset_vals[metric][subset_1_name]",dict_of_metrics_matched_to_subset_vals[metric][
            #    subset_1_name])
            #print(f"metric={metric}, dict_of_metrics_matched_to_subset_vals[metric][subset_2_name]",dict_of_metrics_matched_to_subset_vals[metric][
            #    subset_2_name])
            #print(f"dict_of_metrics_matched_to_shift_metric_vals[metric]={dict_of_metrics_matched_to_shift_metric_vals[metric]}")
        else:
            dict_of_metrics_matched_to_shift_metric_vals[metric] = None
            print(f"Warning: shift metric could not be computed for metric name = {metric}. Subset_1_value = "
                  f"{dict_of_metrics_matched_to_subset_vals[metric][subset_1_name]}, Subset 2 value = {dict_of_metrics_matched_to_subset_vals[metric][subset_2_name]}")

    return dict_of_metrics_matched_to_shift_metric_vals

def getMetric2AllSplitsShiftMetricVals(dict_of_merged_ids_matched_to_raw_res,dict_of_splits_matched_to_subset_ids,subset_1_name='Inside',subset_2_name='Outside',type_of_modelling='binary_class',metrics_of_interest=None,metrics_for_which_shift_metric_cannot_be_computed_for_orig_split=[],conformal_sig_level=0.32,delta_calib_prob=0.05):
    
    #=================================================
    if metrics_of_interest is None: raise Exception('You must specify a list of the metrics of interest!')
    #=================================================
    #=================================================
    if not type_of_modelling in ['binary_class','regression']: raise Exception('Unsupported type_of_modelling={}'.format(type_of_modelling))
    #=================================================
    
    dict_of_metrics_matched_to_all_splits_shift_metric_vals = defaultdict(dict)#neverEndingDefaultDict()
    
    for split_name in dict_of_splits_matched_to_subset_ids.keys():
        subset_1_ids = dict_of_splits_matched_to_subset_ids[split_name][subset_1_name]
        subset_2_ids = dict_of_splits_matched_to_subset_ids[split_name][subset_2_name]
        
        dict_of_metrics_matched_to_subset_vals = computeMetricsOfInterestForCurrentSplit(dict_of_merged_ids_matched_to_raw_res,metrics_of_interest,subset_1_ids,subset_2_ids,subset_1_name,subset_2_name,type_of_modelling,metrics_for_which_shift_metric_cannot_be_computed_for_orig_split=metrics_for_which_shift_metric_cannot_be_computed_for_orig_split,sig_level_of_interest=conformal_sig_level,delta_probs=[delta_calib_prob])
        #print("split_name", split_name)
        #print("dict_of_metrics_matched_to_subset_vals", dict_of_metrics_matched_to_subset_vals)
        dict_of_metrics_matched_to_shift_metric_vals = computeShiftMetricsOfInterestForCurrentSplit(dict_of_metrics_matched_to_subset_vals,subset_1_name,subset_2_name,metrics_for_which_shift_metric_cannot_be_computed_for_orig_split)
        
        
        
        for metric in metrics_of_interest:
            dict_of_metrics_matched_to_all_splits_shift_metric_vals[metric][split_name] = dict_of_metrics_matched_to_shift_metric_vals[metric]
            ###############
            #Debug:
            if 'R2 (cal)' == metric:
                print(f'split_name={split_name} - R2(cal) shift-metric = {dict_of_metrics_matched_to_all_splits_shift_metric_vals[metric][split_name]}')
            ##############
    return dict_of_metrics_matched_to_all_splits_shift_metric_vals

def thisIsRandSplitName(split_name):
    if re.match('(Random-[0-9+])',split_name):
        return True
    else:
        return False

def getMetricSpecificOriginalVsRandShiftMetricVals(dict_of_splits_matched_to_shift_metric_vals):
    
    orig_splits = [split_name for split_name in dict_of_splits_matched_to_shift_metric_vals.keys() if not thisIsRandSplitName(split_name)]
    
    #-----------------------------
    assert 1 == len(orig_splits),"orig_splits={}".format(orig_splits)
    #-----------------------------
    
    the_orig_split = orig_splits[0]
    
    rand_splits = [split_name for split_name in dict_of_splits_matched_to_shift_metric_vals.keys() if thisIsRandSplitName(split_name)]
    
    #------------------------------------------
    assert len(rand_splits) > 1,"rand_splits={}".format(rand_splits)
    assert len(rand_splits)==(len([split_name for split_name in dict_of_splits_matched_to_shift_metric_vals.keys()])-1),"rand_splits={} vs. [split_name for split_name in dict_of_splits_matched_to_shift_metric_vals.keys()]={}".format(rand_splits,[split_name for split_name in dict_of_splits_matched_to_shift_metric_vals.keys()])
    #------------------------------------------
    
    orig_shift_metric_val = dict_of_splits_matched_to_shift_metric_vals[the_orig_split]
    
    rand_shift_metric_vals = [dict_of_splits_matched_to_shift_metric_vals[split] for split in rand_splits]
    
    return orig_shift_metric_val,rand_shift_metric_vals

def getForAllMetricsOriginalVsRandShiftMetricVals(dict_of_metrics_matched_to_all_splits_shift_metric_vals):
    
    dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals = defaultdict(dict)
    
    for metric in dict_of_metrics_matched_to_all_splits_shift_metric_vals.keys():
        dict_of_splits_matched_to_shift_metric_vals = dict_of_metrics_matched_to_all_splits_shift_metric_vals[metric]
        
        orig_shift_metric_val,rand_shift_metric_vals = getMetricSpecificOriginalVsRandShiftMetricVals(dict_of_splits_matched_to_shift_metric_vals)
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals[metric]['original'] = orig_shift_metric_val
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals[metric]['random'] = rand_shift_metric_vals
    
    return dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals


def compute_shift_metric_p_val(metric_dict_of_orig_and_rand_shift_metric_vals,one_sided_sig_test,expected_sub_1_minus_sub_2_sign,metric,sig_level_perc):
    
    #-----------------
    if pd.isna(metric_dict_of_orig_and_rand_shift_metric_vals['original']):
        print(f'WARNING: p-value could not be computed for metric = {metric} - setting to np.nan!')
        return np.nan
    #-----------------

    if not one_sided_sig_test:
        p_val_calc_ready_orig_shift_metric_val = abs(metric_dict_of_orig_and_rand_shift_metric_vals['original'])
        
        p_val_calc_ready_rand_shift_metric_vals = [abs(v) for v in metric_dict_of_orig_and_rand_shift_metric_vals['random'] if not pd.isna(v)]
    else:
        if not expected_sub_1_minus_sub_2_sign in [1,-1]: raise Exception('You need to specify the expected sign of metric[subset 1]-metric[subset 2] to perform a one-tail statistical significance test!')
        
        p_val_calc_ready_orig_shift_metric_val = metric_dict_of_orig_and_rand_shift_metric_vals['original']*expected_sub_1_minus_sub_2_sign
        
        p_val_calc_ready_rand_shift_metric_vals = [(v*expected_sub_1_minus_sub_2_sign) for v in metric_dict_of_orig_and_rand_shift_metric_vals['random'] if not pd.isna(v)]
    
    #----------------
    assert len(p_val_calc_ready_rand_shift_metric_vals) <= len(metric_dict_of_orig_and_rand_shift_metric_vals['random']),f"len(p_val_calc_ready_rand_shift_metric_vals) ={len(p_val_calc_ready_rand_shift_metric_vals)} vs. len(metric_dict_of_orig_and_rand_shift_metric_vals['random']) = {len(metric_dict_of_orig_and_rand_shift_metric_vals['random'])}"
    if len(p_val_calc_ready_rand_shift_metric_vals) < len(metric_dict_of_orig_and_rand_shift_metric_vals['random']):
        print(f'WARNING: for metric = {metric}, shift-metric values could only be computed for {len(p_val_calc_ready_rand_shift_metric_vals)} random splits!')
    if not len(p_val_calc_ready_rand_shift_metric_vals) > 0:
        print(f'WARNING: for metric = {metric}, shift-metric values could not be computed for ANY random split!')
        ################
        #The presence of only zero width PIs in a bin would make any ENCE value for either the in-domain or out-of-domain subset become infinite, making the shift-metric for any partition NaN (if both ENCE values were infinite) or of infinite magnitude
        #A positive ENCE shift-metric should always be statistically insignificant
        ###############
        if 'ENCE' == metric and  (pd.isna(p_val_calc_ready_orig_shift_metric_val)):
            p_val = np.nan
            return p_val
        elif 'ENCE' == metric and np.isposinf(p_val_calc_ready_orig_shift_metric_val):
            p_val = np.nan
            return p_val
        else:
            raise Exception(f'PROBLEM: for metric = {metric}, shift-metric values could not be computed for ANY random split!')
    
    assert not pd.isna(p_val_calc_ready_orig_shift_metric_val),f'Unexpected for metric = {metric}'
    #----------------

    p_val = len([v for v in p_val_calc_ready_rand_shift_metric_vals if v >= p_val_calc_ready_orig_shift_metric_val])/len(p_val_calc_ready_rand_shift_metric_vals)
    
    return p_val

def compute_all_raw_shift_metric_p_vals(dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals,one_sided_sig_test=True,metric_to_expected_sub_1_minus_sub_2_sign=None,metrics_of_interest=None,metrics_for_which_shift_metric_cannot_be_computed_for_orig_split=[],sig_level_perc=5):
    ########################################
    if metrics_of_interest is None: raise Exception('You must specify metrics of interest!')
    assert isinstance(metrics_of_interest,list),metrics_of_interest
    ######################################## 
    #########################################
    if one_sided_sig_test:
        if metric_to_expected_sub_1_minus_sub_2_sign is None: raise Exception('A metric_to_expected_sub_1_minus_sub_2_sign is needed if we specify one_sided_sig_test=True.')
    ##########################################
    #########################################
    if len(metrics_for_which_shift_metric_cannot_be_computed_for_orig_split) > 0:
        ###########################
        #Surprisingly, an RMSE p-value of 0.77 was observed when one_sided_sig_test=True. However, looking at original SYN boxplots, the one-tail and two-tail p-values were different and consistent with boxplots, including some example where one-tail p-value was > 0.5 and two-tail p-value = 1.0.
        dummy_p_val_if_metric_cannot_be_computed = np.nan
        ##############################
    ##########################################
    
    dict_of_metrics_matched_to_p_vals = {}
    
    for metric in metrics_of_interest:
        #========================
        if metric in metrics_for_which_shift_metric_cannot_be_computed_for_orig_split:
            dict_of_metrics_matched_to_p_vals[metric] = dummy_p_val_if_metric_cannot_be_computed
            continue
        #========================


        if metric in dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals.keys():
            
            metric_dict_of_orig_and_rand_shift_metric_vals = dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals[metric]
        
            if one_sided_sig_test:
                expected_sub_1_minus_sub_2_sign = metric_to_expected_sub_1_minus_sub_2_sign[metric]
            else:
                expected_sub_1_minus_sub_2_sign = None
        
            p_val = compute_shift_metric_p_val(metric_dict_of_orig_and_rand_shift_metric_vals,one_sided_sig_test,expected_sub_1_minus_sub_2_sign,metric,sig_level_perc)

            dict_of_metrics_matched_to_p_vals[metric] = p_val

            
        else:
            raise Exception(f'THIS SHOULD NOT HAPPEN: Due to failure to compute pre-requisites, the code is unable to compute current p-value for {metric}!')
    
    return dict_of_metrics_matched_to_p_vals

def getInputForGraphicalSummaries(input_for_graphical_summaries,metric,orig_shift_metric_val,rand_shift_metric_vals,x_lab='Groups',y_lab='Shift-Metric',legend_lab='Split Basis'):
    
    input_for_graphical_summaries[metric][y_lab].append(orig_shift_metric_val)
    
    input_for_graphical_summaries[metric][legend_lab].append('Original')
    
    input_for_graphical_summaries[metric][x_lab].append('Original')
    
    for diff in rand_shift_metric_vals:
        input_for_graphical_summaries[metric][y_lab].append(diff)
        
        input_for_graphical_summaries[metric][legend_lab].append('Random')
        
        input_for_graphical_summaries[metric][x_lab].append('Random')
    
    return input_for_graphical_summaries

def getAllInputsForGraphicalSummaries(dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals):
    
    input_for_graphical_summaries = doubleDefaultDictOfLists(returnDefDictOfLists)
    
    for metric in dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals.keys():
        
        orig_shift_metric_val = dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals[metric]['original']
        
        rand_shift_metric_vals = dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals[metric]['random']
    
        input_for_graphical_summaries = getInputForGraphicalSummaries(input_for_graphical_summaries,metric,orig_shift_metric_val,rand_shift_metric_vals)
    
    return input_for_graphical_summaries

def avoid_problematic_file_names_on_linux(plot_file):
    return plot_file.replace("N/A","N.A")

def produceGraphicalSummaries(input_for_graphical_summaries,plot_file_prefix,x_lab='Groups',y_lab='Shift-Metric',legend_lab='Split Basis',sig_level_perc=5,one_sided_sig_test=False,debug=False):
    for short_stat_name in input_for_graphical_summaries.keys():
        plot_input = pd.DataFrame(input_for_graphical_summaries[short_stat_name])
        
        plot_file = '{}_{}.tiff'.format(plot_file_prefix,short_stat_name)

        plot_file = avoid_problematic_file_names_on_linux(plot_file)

        if not one_sided_sig_test:
            old_name_for_plot_file = plot_file
            plot_file = re.sub('(\.tiff$)','_two_tail.tiff',old_name_for_plot_file)
            assert not plot_file == old_name_for_plot_file,old_name_for_plot_file
            whiskers_lower_limit = sig_level_perc/2
        else:
            whiskers_lower_limit = sig_level_perc
        
        whiskers_percentile_values = [whiskers_lower_limit,(100-whiskers_lower_limit)]

        
        axes_obj = sb.boxplot(x = x_lab, y = y_lab, data = plot_input, hue = legend_lab, palette="muted",whis=whiskers_percentile_values)
        
        pp.ylabel('{}: {}'.format(short_stat_name,y_lab), fontsize = 16)
        
        pp.xlabel(x_lab, fontsize = 16)
        
        pp.savefig(plot_file)
        pp.clf()

def get_metrics_for_which_shift_metric_cannot_be_computed_for_orig_split(dict_of_raw_subset_results,subset_1_name,subset_2_name,type_of_modelling,metrics_of_interest,sig_level_of_interest=0.32,delta_calib_prob=0.05,consistent_min_no_cmpds_for_stats = 2):
    #######################
    #default for consistent_min_no_cmpds_for_stats = 2 is consistent with calls to size_of_inputs_for_stats_is_big_enough(....) in all_key_reg_stats_and_plots.py and all_key_class_stats_and_plots.py
    #######################

    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split = []

    for subset in [subset_1_name,subset_2_name]:
        ##########################
        #The following was copied and adapted from check_missing_or_inf_metrics_are_for_the_expected_reasons.py:
        #DONE: Double check all of the below, including following checks on raw results, against that script!

        if 'regression' == type_of_modelling:
            predictions = dict_of_raw_subset_results[subset]['subset_test_predictions'] 

            experi_vals = dict_of_raw_subset_results[subset]['subset_test_y']

            class_1_probs = None

            try:
                reg_pred_intervals = dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'][sig_level_of_interest]
            except KeyError:
                assert 0 == len(predictions),f'predictions={predictions}'
                reg_pred_intervals = None

            delta_calib_plot_experi_probs = None
        elif 'binary_class' == type_of_modelling:
            predictions = dict_of_raw_subset_results[subset]['subset_predicted_y']

            experi_vals = dict_of_raw_subset_results[subset]['subset_test_y']

            try:
                class_1_probs = dict_of_raw_subset_results[subset]['subset_probs_for_class_1']
            except KeyError:
                assert 0 == len(predictions),f'predictions={predictions}'
                class_1_probs = None

            reg_pred_intervals = None

            ###################
            delta_calib_plot_experi_probs = get_experi_class_1_probs(class_1_probs,experi_vals,delta_prob=delta_calib_prob)
            ####################
        else:
            raise Exception(f'type_of_modelling={type_of_modelling}')
        ###########################

        assert len(experi_vals) == len(predictions),f'len(experi_vals)={len(experi_vals)} vs. len(predictions)={len(predictions)}'

        #######################
        #The following was copied and adapted from def computeMetricsOfInterestForCurrentSplit(...) for consistency in the names assigned to the metrics:

        for metric in metrics_of_interest:
            if not size_of_inputs_for_stats_is_big_enough(experi_vals,predictions,limit=consistent_min_no_cmpds_for_stats):
                metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                continue

            if 'Balanced Accuracy' == metric:
                if 1 == len(set(experi_vals)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue
            
            elif 'MCC' == metric:
                if 1 == len(set(experi_vals)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue
            
                
            elif 'AUC' == metric:
                if 1 == len(set(experi_vals)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue
                    
            
            elif 'Kappa' == metric:
                if 1 == len(set(experi_vals)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue
                    
                
            elif 'Stratified Brier Score' == metric:
                if 1 == len(set(experi_vals)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue
                
                    
            elif 'RMSE (cal)' == metric:
                pass
                    
            
            elif 'R2 (cal)' == metric:
                if 1 == len(set(experi_vals)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue

                if 1 == len(set(delta_calib_plot_experi_probs)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue
                        
            elif 'Pearson coefficient (cal)' == metric:
                if 1 == len(set(experi_vals)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue

                if 1 == len(set(delta_calib_plot_experi_probs)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue
            
                
            elif 'Spearman coefficient (cal)' == metric:
                if 1 == len(set(experi_vals)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue

                if 1 == len(set(delta_calib_plot_experi_probs)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue
            
                
            elif 'RMSE' == metric:
                pass
            
            
            elif 'R2' == metric:
                if 1 == len(set(experi_vals)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue
            
                
            elif 'Pearson coefficient' == metric:
                if 1 == len(set(experi_vals)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue

                if 1 == len(set(predictions)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue
            
                
            elif 'Spearman coefficient' == metric:
                if 1 == len(set(experi_vals)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue

                if 1 == len(set(predictions)):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue
            
        
            elif 'Validity' == metric:
                pass
            
                
            elif 'Efficiency' == metric:
                pass
            
                
            elif 'ECE' == metric:
                pass
            
                
            elif 'ENCE' == metric:
                pass 
            
            elif 'Spearman coefficient (PIs vs. residuals)' == metric:

                if all([0==v for v in getIntervalsWidthOfInterest(sig_level_2_prediction_intervals={sig_level_of_interest:reg_pred_intervals},sig_level_of_interest=sig_level_of_interest).tolist()]):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue

                if 1 == len(set(getIntervalsWidthOfInterest(sig_level_2_prediction_intervals={sig_level_of_interest:reg_pred_intervals},sig_level_of_interest=sig_level_of_interest).tolist())):
                    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split.append(metric)
                    continue
            
            else:
                raise Exception('Unrecognised metric : {}'.format(metric))
    ######################

    #It is possible that some metrics cannot be computed for both subsets of the original split (based on the AD method), but this is not expected:
    metrics_for_which_shift_metric_cannot_be_computed_for_orig_split = list(set(metrics_for_which_shift_metric_cannot_be_computed_for_orig_split))


    return metrics_for_which_shift_metric_cannot_be_computed_for_orig_split

def get_dict_of_metrics_matched_to_original_shift_values(dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals):
    if dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals is None:
        dict_of_metrics_matched_to_original_shift_values = None
    else:
        dict_of_metrics_matched_to_original_shift_values = {}
        for metric in dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals:
            dict_of_metrics_matched_to_original_shift_values[metric] = dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals[metric]['original']


    return dict_of_metrics_matched_to_original_shift_values

def workflowForOneSetOfRawResults(out_dir,test_set_plus_methods_name,dict_of_raw_subset_results,subset_1_name='Inside',subset_2_name='Outside',type_of_modelling='binary_class',no_rand_splits=100,strat_rand_split_y_name='strat_y',rand_seed=42,metrics_of_interest=None,x_lab='Groups',y_lab='Shift-Metric',legend_lab='Split Basis',sig_level_perc=5,one_sided_sig_test=True,metric_to_expected_sub_1_minus_sub_2_sign=None,create_plots=True,set_p_to_nan_if_metric_cannot_be_computed_for_orig_split=True,conformal_sig_level=0.32,delta_calib_prob=0.05,debug=False):
    
    #=======================
    if debug:
        print(f'dict_of_raw_subset_results={dict_of_raw_subset_results}')
    #=======================

    if set_p_to_nan_if_metric_cannot_be_computed_for_orig_split:
        metrics_for_which_shift_metric_cannot_be_computed_for_orig_split = get_metrics_for_which_shift_metric_cannot_be_computed_for_orig_split(dict_of_raw_subset_results,subset_1_name,subset_2_name,type_of_modelling,metrics_of_interest,sig_level_of_interest=conformal_sig_level)
    else:
        metrics_for_which_shift_metric_cannot_be_computed_for_orig_split = []
    #
    if not len(metrics_of_interest) == len(set(metrics_of_interest).intersection(metrics_for_which_shift_metric_cannot_be_computed_for_orig_split)):
    
        dict_of_splits_matched_to_subset_ids = getDictOfSplitsMatchedToSubsetIDs(dict_of_raw_subset_results=dict_of_raw_subset_results,subset_1_name=subset_1_name,subset_2_name=subset_2_name,type_of_modelling=type_of_modelling,no_rand_splits=no_rand_splits,strat_rand_split_y_name=strat_rand_split_y_name,rand_seed=rand_seed,delta_calib_prob=delta_calib_prob,metrics_of_interest=metrics_of_interest)
    
        dict_of_merged_ids_matched_to_raw_res = getDictOfMergedIdsMatched2RawResults(dict_of_raw_subset_results=dict_of_raw_subset_results,type_of_modelling=type_of_modelling)
    
        dict_of_metrics_matched_to_all_splits_shift_metric_vals = getMetric2AllSplitsShiftMetricVals(dict_of_merged_ids_matched_to_raw_res=dict_of_merged_ids_matched_to_raw_res,dict_of_splits_matched_to_subset_ids=dict_of_splits_matched_to_subset_ids,subset_1_name=subset_1_name,subset_2_name=subset_2_name,type_of_modelling=type_of_modelling,metrics_of_interest=metrics_of_interest,metrics_for_which_shift_metric_cannot_be_computed_for_orig_split=metrics_for_which_shift_metric_cannot_be_computed_for_orig_split,conformal_sig_level=conformal_sig_level,delta_calib_prob=delta_calib_prob)
    
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals = getForAllMetricsOriginalVsRandShiftMetricVals(dict_of_metrics_matched_to_all_splits_shift_metric_vals)
    
        if create_plots:
            input_for_graphical_summaries = getAllInputsForGraphicalSummaries(dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals)
        
        
            plot_file_prefix = os.path.sep.join([out_dir,'{}_ADSplitVsRandSplits'.format(test_set_plus_methods_name)])
        
            produceGraphicalSummaries(input_for_graphical_summaries,plot_file_prefix,x_lab=x_lab,y_lab=y_lab,legend_lab=legend_lab,sig_level_perc=sig_level_perc,one_sided_sig_test=one_sided_sig_test)
    else:
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals = None

    dict_of_metrics_matched_to_original_shift_values = get_dict_of_metrics_matched_to_original_shift_values(dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals)

    dict_of_metrics_matched_to_p_vals = compute_all_raw_shift_metric_p_vals(dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals=dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals,one_sided_sig_test=one_sided_sig_test,metric_to_expected_sub_1_minus_sub_2_sign=metric_to_expected_sub_1_minus_sub_2_sign,metrics_of_interest=metrics_of_interest,metrics_for_which_shift_metric_cannot_be_computed_for_orig_split=metrics_for_which_shift_metric_cannot_be_computed_for_orig_split,sig_level_perc=sig_level_perc)
    
    return dict_of_metrics_matched_to_p_vals,dict_of_metrics_matched_to_original_shift_values

def writeTableOfRawPVals(out_dir,p_vals_table,test_set_plus_methods_name_matched_to_dict_of_metrics_matched_to_p_vals,test_set_plus_methods_name_matched_to_overall_dict_of_metrics_matched_to_p_vals_metadata):
    ######################################
    #p_vals_table: the relative name, relative to out_dir folder, of a csv file populated with the following columns:
    #'Scenario','Metric','Shift-Metric P-value' + [list of p-value metadata]
    ######################################
    p_vals_dict_of_lists = defaultdict(list)
    
    for test_name_plus_methods_name in test_set_plus_methods_name_matched_to_dict_of_metrics_matched_to_p_vals.keys():
        for metric_name in test_set_plus_methods_name_matched_to_dict_of_metrics_matched_to_p_vals[test_name_plus_methods_name].keys():
            p_vals_dict_of_lists['Scenario'].append(test_name_plus_methods_name)
            p_vals_dict_of_lists['Metric'].append(metric_name)
            p_vals_dict_of_lists['Shift-Metric P-value'].append(test_set_plus_methods_name_matched_to_dict_of_metrics_matched_to_p_vals[test_name_plus_methods_name][metric_name])

            p_vals_metadata_dict = test_set_plus_methods_name_matched_to_overall_dict_of_metrics_matched_to_p_vals_metadata[test_name_plus_methods_name][metric_name]

            for metadatum_name in p_vals_metadata_dict.keys():
                p_vals_dict_of_lists[metadatum_name].append(test_set_plus_methods_name_matched_to_overall_dict_of_metrics_matched_to_p_vals_metadata[test_name_plus_methods_name][metric_name][metadatum_name])
            
    
    df = pd.DataFrame(p_vals_dict_of_lists)
    
    df.to_csv(os.path.sep.join([out_dir,p_vals_table]),index=False)

def workflowForAFamilyOfRawResults(out_dir,dict_for_all_test_sets_of_raw_subset_results,subset_1_name='Inside',subset_2_name='Outside',
                                   type_of_modelling='binary_class',no_rand_splits=None,strat_rand_split_y_name='strat_y',rand_seed=42,
                                   metrics_of_interest=None,x_lab='Groups',y_lab='Shift-Metric',legend_lab='Split Basis',sig_level_perc=5,
                                   one_sided_sig_test=True,metric_to_expected_sub_1_minus_sub_2_sign=None,p_vals_table='PVals.csv',
                                   adjusted_p_vals_table_name=None,results_are_obtained_over_multiple_folds_and_or_seeds=False,create_plots=True,
                                   debug=False, scenarios_with_errors=[],adjust_p_vals=False,set_p_to_nan_if_metric_cannot_be_computed_for_orig_split=True,conformal_sig_level=0.32,delta_calib_prob=0.05,p_val_aggregation_option="conservative_twice_average"):
    #######################################
    #if results_are_obtained_over_multiple_folds_and_or_seeds=False, dict_for_all_test_sets_of_raw_subset_results has the following structure:
    #dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name] = dict_of_raw_subset_results #N.B. The test_set_plus_methods_name could correspond to the results across different datasets for different endpoints if conclusions needed to be made for the AD method across those settings!
    #======================
    #See def getDictOfMergedIdsMatched2RawResults(...): for a description of dict_of_raw_subset_results
    #=====================
    #if results_are_obtained_over_multiple_folds_and_or_seeds=True, dict_for_all_test_sets_of_raw_subset_results has the following structure:
    #dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name] = dict_for_all_folds_and_or_model_seeds_of_raw_subset_results, where dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results
    #=====================
    #######################################

    ########################################
    if metrics_of_interest is None: raise Exception('You must specify metrics of interest!')
    assert isinstance(metrics_of_interest,list),metrics_of_interest

    if not isinstance(no_rand_splits,int): raise Exception('You must specify a number of random permutations for the AD p-values!')

    if not adjusted_p_vals_table_name is None or adjust_p_vals: raise Exception(f'Local adjustment of p-values is not needed and does not make sense!')
    
    if not set_p_to_nan_if_metric_cannot_be_computed_for_orig_split: raise Exception(f'This should note be changed : set_p_to_nan_if_metric_cannot_be_computed_for_orig_split = True')
    

    if one_sided_sig_test:
        if not 'one_tail' in os.path.basename(p_vals_table):
            print(f'WARNING: We should be computing one-tail AD p-values, but p_vals_table={p_vals_table}')
            p_vals_table = os.path.sep.join([os.path.dirname(p_vals_table),f'one_tail_{os.path.basename(p_vals_table)}'])
        
        if metric_to_expected_sub_1_minus_sub_2_sign is None: raise Exception(f'We need this for one=tail AD p-values: {metric_to_expected_sub_1_minus_sub_2_sign}')
        if not isinstance(metric_to_expected_sub_1_minus_sub_2_sign,dict): raise Exception(f'We need this dictionary for one=tail AD p-values: {metric_to_expected_sub_1_minus_sub_2_sign}')
    else:
        if 'one_tail' in os.path.basename(p_vals_table):
            raise Exception(f'We should be computing two-tail AD p-values, but p_vals_table={p_vals_table}')
    ######################################## 

    
    test_set_plus_methods_name_matched_to_dict_of_metrics_matched_to_p_vals = {}
    test_set_plus_methods_name_matched_to_overall_dict_of_metrics_matched_to_p_vals_metadata = {}
    
    for test_set_plus_methods_name in dict_for_all_test_sets_of_raw_subset_results.keys():
        ###############
        # Debugging:
        if debug:
            print(f"test_set_plus_methods_name={test_set_plus_methods_name}")
            if not 0 == len(scenarios_with_errors):
                if not test_set_plus_methods_name in scenarios_with_errors:
                    continue
                else:
                    print("Found problem scenario.")

        ###############
        if not results_are_obtained_over_multiple_folds_and_or_seeds:
            raise Exception('Since this is not relevant for the current paper, we have currently removed support for results which are not obtained over multiple folds and/or seeds! \n Restoring this would require support for overall_dict_of_metrics_matched_to_p_vals_metadata inside def workflowForOneSetOfRawResults(...)')
        else:
            dict_for_all_folds_and_or_model_seeds_of_raw_subset_results = dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name]
            
            dict_of_metrics_matched_to_p_vals,overall_dict_of_metrics_matched_to_p_vals_metadata = workflowForOneSetOfCrossValidationAndOrMultipleSeedsResults(dict_for_all_folds_and_or_model_seeds_of_raw_subset_results,out_dir,test_set_plus_methods_name,subset_1_name,subset_2_name,type_of_modelling,no_rand_splits,strat_rand_split_y_name,rand_seed,metrics_of_interest,x_lab,y_lab,legend_lab,sig_level_perc,one_sided_sig_test,metric_to_expected_sub_1_minus_sub_2_sign,create_plots,set_p_to_nan_if_metric_cannot_be_computed_for_orig_split,conformal_sig_level=conformal_sig_level,delta_calib_prob=delta_calib_prob,debug=debug,p_val_aggregation_option=p_val_aggregation_option)
        
        
        test_set_plus_methods_name_matched_to_dict_of_metrics_matched_to_p_vals[test_set_plus_methods_name] = dict_of_metrics_matched_to_p_vals

        test_set_plus_methods_name_matched_to_overall_dict_of_metrics_matched_to_p_vals_metadata[test_set_plus_methods_name] = overall_dict_of_metrics_matched_to_p_vals_metadata

    writeTableOfRawPVals(out_dir,p_vals_table,test_set_plus_methods_name_matched_to_dict_of_metrics_matched_to_p_vals,test_set_plus_methods_name_matched_to_overall_dict_of_metrics_matched_to_p_vals_metadata)

    

def replace_None_with_nan(v):
    if v is None:
        return np.nan
    else:
        return v

def get_average_shift_metric_val(dict_of_metrics_matched_to_original_shift_values_list,metric):
    all_values = dict_of_metrics_matched_to_original_shift_values_list[metric]
    #####################
    #The following calculations should be consistent with how we compute mean original shift-metric values in the AD_ranking.py script, where we use grouped.mean():
    #https://pandas.pydata.org/pandas-docs/version/0.25.0/reference/api/pandas.core.groupby.GroupBy.mean.html
    #https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html
    #######################
    #However, this calculation requires missing values to be np.nan, not None!
    all_values = [replace_None_with_nan(v) for v in all_values]
    return np.nanmean(all_values,axis=0)

def has_wrong_sign(average_shift_metric_val,metric,metric_to_expected_sub_1_minus_sub_2_sign):
    if pd.isna(average_shift_metric_val):
        return np.nan

    if (average_shift_metric_val*metric_to_expected_sub_1_minus_sub_2_sign[metric]) <= 0:
        return True
    else:
        return False

def workflowForOneSetOfCrossValidationAndOrMultipleSeedsResults(dict_for_all_folds_and_or_model_seeds_of_raw_subset_results,out_dir,test_set_plus_methods_name,subset_1_name,subset_2_name,type_of_modelling,no_rand_splits,strat_rand_split_y_name,rand_seed,metrics_of_interest,x_lab,y_lab,legend_lab,sig_level_perc,one_sided_sig_test,metric_to_expected_sub_1_minus_sub_2_sign,create_plots=True,set_p_to_nan_if_metric_cannot_be_computed_for_orig_split=True,conformal_sig_level=0.32,delta_calib_prob=0.05,debug=False,p_val_aggregation_option="conservative_twice_average"):
    ########################################################
    #Consider a scenario in which multiple, random train/test splits of the dataset are performed, e.g. as per cross-validation, and/or multiple random seeds are chosen to generate the modelling results. 
    #For each of those sub-scenarios, i.e. a given train/test split and/or random seed value, we can compute a p-value for each of the shift-metrics.
    #To assess whether the shift-metrics show statistically significant differences between inside and outside the domain across the different train/test splits and/or random seeds used for modelling, we need to aggregate the p-values.
    #########################################################
    
    overall_dict_of_metrics_matched_to_p_vals = {}

    overall_dict_of_metrics_matched_to_p_vals_metadata = defaultdict(dict)
    
    dict_of_metrics_matched_to_p_vals_list = defaultdict(list)

    dict_of_metrics_matched_to_original_shift_values_list = defaultdict(list)
    
    for fold_and_or_model_seed_combination in dict_for_all_folds_and_or_model_seeds_of_raw_subset_results.keys():
        #=======================
        if debug:
            print(f'fold_and_or_model_seed_combination={fold_and_or_model_seed_combination}')
        #======================

        dict_of_raw_subset_results = dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination]
        
        
        current_test_set_plus_methods_name = '{}_{}'.format(test_set_plus_methods_name,fold_and_or_model_seed_combination) #Original test_set_plus_methods_name denotes the dataset and a type of splitting (e.g. random or clustered)
        
        
        dict_of_metrics_matched_to_p_vals,dict_of_metrics_matched_to_original_shift_values = workflowForOneSetOfRawResults(out_dir,current_test_set_plus_methods_name,dict_of_raw_subset_results,subset_1_name,subset_2_name,type_of_modelling,no_rand_splits,strat_rand_split_y_name,rand_seed,metrics_of_interest,x_lab,y_lab,legend_lab,sig_level_perc,one_sided_sig_test,metric_to_expected_sub_1_minus_sub_2_sign,create_plots,set_p_to_nan_if_metric_cannot_be_computed_for_orig_split,conformal_sig_level=conformal_sig_level,delta_calib_prob=delta_calib_prob,debug=debug)
        
        for metric in dict_of_metrics_matched_to_p_vals.keys():
            dict_of_metrics_matched_to_p_vals_list[metric].append(dict_of_metrics_matched_to_p_vals[metric])
            if not dict_of_metrics_matched_to_original_shift_values is None:
                dict_of_metrics_matched_to_original_shift_values_list[metric].append(dict_of_metrics_matched_to_original_shift_values[metric])
            else:
                dict_of_metrics_matched_to_original_shift_values_list[metric].append(None)
    
    for metric in dict_of_metrics_matched_to_p_vals_list.keys():
        overall_dict_of_metrics_matched_to_p_vals[metric] = aggregate_p_vals(dict_of_metrics_matched_to_p_vals_list[metric],p_val_aggregation_option=p_val_aggregation_option)

        ##########################
        #Additional information relevant to analyzing these results:
        all_p_vals_str = ';'.join([str(p) for p in dict_of_metrics_matched_to_p_vals_list[metric]])
        all_shift_metrics_str = ';'.join([str(mv) for mv in dict_of_metrics_matched_to_original_shift_values_list[metric]])
        average_shift_metric_val = get_average_shift_metric_val(dict_of_metrics_matched_to_original_shift_values_list,metric)
        
        if one_sided_sig_test:
            average_shift_metric_val_has_wrong_sign = has_wrong_sign(average_shift_metric_val,metric,metric_to_expected_sub_1_minus_sub_2_sign)
        else:
            average_shift_metric_val_has_wrong_sign = None
        
        overall_dict_of_metrics_matched_to_p_vals_metadata[metric]['all_p_vals_str'] = all_p_vals_str
        overall_dict_of_metrics_matched_to_p_vals_metadata[metric]['all_shift_metrics_str'] = all_shift_metrics_str
        overall_dict_of_metrics_matched_to_p_vals_metadata[metric]['average_shift_metric_val'] = average_shift_metric_val
        overall_dict_of_metrics_matched_to_p_vals_metadata[metric]['average_shift_metric_val_has_wrong_sign'] = average_shift_metric_val_has_wrong_sign
        
        if debug:
            print(f'metric={metric}')
            print(f'Contents of overall_dict_of_metrics_matched_to_p_vals_metadata :')
            for extra_info in overall_dict_of_metrics_matched_to_p_vals_metadata[metric].keys():
                print(f'{extra_info} : value = {overall_dict_of_metrics_matched_to_p_vals_metadata[metric][extra_info]}')
                print(f'{extra_info} : type = {type(overall_dict_of_metrics_matched_to_p_vals_metadata[metric][extra_info])}')
        #########################
        
    return overall_dict_of_metrics_matched_to_p_vals,overall_dict_of_metrics_matched_to_p_vals_metadata
#
