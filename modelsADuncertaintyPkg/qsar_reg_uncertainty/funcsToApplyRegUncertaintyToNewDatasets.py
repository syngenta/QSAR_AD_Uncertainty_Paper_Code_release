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
##############################
#Copright (c) 2022-2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#This file was created to implement the Python package, by Richard Liam Marchese Robinson
#The ACP and SCP functions were adapted from  calibration_plots.py, plus multiPart.py where indicated, developed at Uppsala University and downloaded from https://github.com/pharmbio/SCPRegression [See below for the original copyright and license information.]
#Edits were made, prior to incoporation of relevant lines of code into the ACP and SCP functions, by Zied Hosni (z.hosni [at] sheffield.ac.uk), whilst working on a Syngenta funded project
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
######################################################
###############################
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import random
from pytest import approx
#------------------
from . import RegressionICP as icp #https://docs.python.org/3/reference/import.html
from .combine_pred_intervals import aggregateIntervals
from .RF_prediction_intervals import pred_ints as rfPredInts
from .RF_ENS_confidence_intervals import estimateConfidenceIntervalOfExperimentalValues as rfPredInts_ens
from ..utils.ML_utils import singleRandomSplit,get_multiple_random_splits
from ..utils.time_utils import basic_time_task
from ..utils.basic_utils import report_name_of_function_where_this_is_called

#RLMR: all occurences to clf below replaced with model
####################
#Key globals used by functions:
consistent_default_calib_fraction = 0.25
#The following global variable can be switched to True by the outer function (getTestPredictionsAndPredictionIntervalsForAllSignificanceLevels(...)) for use in scripts which seem to be taking a surprisingly long time to execute:
monitor_time=False
#####################

def ICPtrainComputeCalibSetConfScores(X_train,X_calib,y_train,y_calib,ml_alg,non_conformity_scaling,global_random_seed,nrTrees=100):
    if "RandomForestRegressor" == ml_alg and "exp(stdev of tree predictions)" == non_conformity_scaling:
        
        calib_pred,model = ICPtrainComputeCalibSetPredictions(X_train,X_calib,y_train,ml_alg,nrTrees,global_random_seed)
        
        if monitor_time:
            start = time.time()

        y_cal_std =  icp.compute_y_std(X=X_calib, numberTrees=nrTrees,RF_reg_model=model,monitor_time=monitor_time)
        
        lamda_cal = icp.lamda_exp_std(y_cal_std, w=1,monitor_time=monitor_time)
        
        confScores = icp.computeNonConformityScores(calib_pred, y_calib, lamda_cal,monitor_time=monitor_time)

        if monitor_time:
            end = time.time()

            task = 'compute calibration set nonconformity scores'

            basic_time_task(task,end,start)
            del task,end,start
    else:
        raise Exception("Conformal regression is only currently supported for Random Forest regression and non-coformity scaling using exp(stdev of tree predictions)")
    
    return confScores,model

def ICPtrainComputeCalibSetPredictions(X_train,X_calib,y_train,ml_alg,nrTrees,global_random_seed):
    if "RandomForestRegressor" == ml_alg:
        
        if monitor_time:
            build_start = time.time()
        
        model = icp.fit_RF(X_train, y_train, nrTrees=nrTrees, global_random_seed=global_random_seed)

        if monitor_time:
            build_end = time.time()

            build_task = 'model building'

            basic_time_task(build_task,build_end,build_start)
            del build_task,build_end,build_start
    else:
        raise Exception("Conformal regression is only currently supported for Random Forest regression")
    
    if monitor_time:
        pred_start = time.time()

    calib_pred = model.predict(X_calib)

    if monitor_time:
        pred_end = time.time()

        pred_task = 'make predictions for calibration set'

        basic_time_task(pred_task,pred_end,pred_start)
        del pred_task,pred_end,pred_start
    
    return calib_pred,model

def getConformalRegressionModelsPlusCalibDetails_ICP(train_inc_calib_x,train_inc_calib_y,global_random_seed,calib_fraction=consistent_default_calib_fraction,ml_alg="RandomForestRegressor",nrTrees=100,non_conformity_scaling="exp(stdev of tree predictions)",stratified=True):
    
    train_x,train_y,calib_x,calib_y = singleRandomSplit(data_x=train_inc_calib_x,data_y=train_inc_calib_y,test_fraction=calib_fraction,random_state=global_random_seed,stratified=stratified)
    
    #================================
    X_train = train_x
    y_train = train_y
    X_calib = calib_x
    y_calib = calib_y
    #=================================
    
    confScores,model = ICPtrainComputeCalibSetConfScores(X_train,X_calib,y_train,y_calib,ml_alg,non_conformity_scaling,global_random_seed,nrTrees)
    
    return confScores,model

def applyConformalRegressionToNewCmpds_ICP(confScores,model,X_test,sig_level=0.32,ml_alg="RandomForestRegressor",nrTrees=100,non_conformity_scaling="exp(stdev of tree predictions)"):
    if monitor_time:
        start = time.time()

    testPred = model.predict(X_test)
    
    if "RandomForestRegressor" == ml_alg and "exp(stdev of tree predictions)" == non_conformity_scaling:
        #####################
        assert len(model.estimators_) == nrTrees,"Inconsistent number of trees assumed during ICP training/calibration and predictions/prediction interval calculations! \n len(model.estimators_) = {} nrTrees = {}.".format(len(model.estimators_),nrTrees)
        ####################
        
        y_test_std = icp.compute_y_std(X=X_test, numberTrees=nrTrees,RF_reg_model=model)
    
        lamda_test = icp.lamda_exp_std(y_test_std, w=1)
    else:
        raise Exception("Conformal regression is only currently supported for Random Forest regression and non-coformity scaling using exp(stdev of tree predictions)")
    
    
    intervals = icp.computeInterval(confScores, testPred, sig_level, lamda_test)
    
    testPred = testPred.tolist()

    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start)
        del task,end,start

    return testPred,intervals

def getConformalRegressionModelsPlusCalibDetails_ACP(train_inc_calib_x,train_inc_calib_y,global_random_seed,number_of_calib_splits=100,ml_alg="RandomForestRegressor",nrTrees=100,non_conformity_scaling="exp(stdev of tree predictions)",stratified=True,calib_fraction=consistent_default_calib_fraction):
    
    start = time.time()

    #==============================
    XX = pd.DataFrame(train_inc_calib_x).reset_index(drop=True,inplace=False)
    yy = pd.Series(train_inc_calib_y).reset_index(drop=True,inplace=False)
    
    n_source = number_of_calib_splits
    #=============================
    
    ACP_dict_of_confScores = {}
    ACP_dict_of_model = {}
    
    sourceIndex = 0
    
    dict_of_splits = get_multiple_random_splits(no_splits=n_source,data_x=XX,data_y=yy,test_fraction=calib_fraction,random_state=global_random_seed,stratified=stratified,reset_indices=True)
    
    for split in dict_of_splits.keys():
        
        X_train = dict_of_splits[split]['train_x']
        y_train = dict_of_splits[split]['train_y']
        X_calib = dict_of_splits[split]['test_or_calib_x']
        y_calib = dict_of_splits[split]['test_or_calib_y']
        
        confScores,model = ICPtrainComputeCalibSetConfScores(X_train,X_calib,y_train,y_calib,ml_alg,non_conformity_scaling,global_random_seed,nrTrees)
        
        ACP_dict_of_confScores[sourceIndex] = confScores
        
        ACP_dict_of_model[sourceIndex] = model
        
        sourceIndex += 1
    
    #==============================
    assert n_source == sourceIndex,"n_source = {} vs. sourceIndex {}. But, I assume these are equal in applyConformalRegressionToNewCmpds_ACP(...)!".format(n_source,sourceIndex)
    #===============================
    
    end = time.time()

    task = report_name_of_function_where_this_is_called()

    basic_time_task(task,end,start)
    del task,end,start

    return ACP_dict_of_confScores,ACP_dict_of_model

def replace_nan_with_zero(v):
    if np.isnan(v):
        return 0.0
    else:
        return v

def check_consistency_of_nan_acp_predictions_and_expected_inf_intervals(test_predictions,intervals):
    #==============================================
    #In the initial version of the ICP code, the prediction intervals could be set explicitly to -np.inf,np.inf
    #However, this should no longer be relevant (14/09/24)!
    #==============================================
    if monitor_time:
        start = time.time()
    
    nan_predictions_indices = [i for i in range(len(test_predictions)) if np.isnan(test_predictions[i])]
    inf_intervals_indices = [i for i in range(len(test_predictions)) if np.array_equal(np.array([-np.inf,np.inf]),intervals[i])]
    assert nan_predictions_indices == inf_intervals_indices,f'nan_predictions_indices={nan_predictions_indices} vs. inf_intervals_indices={inf_intervals_indices}'

    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start)
        del task,end,start

def parse_acp_nan_predictions(test_predictions,intervals):
    if monitor_time:
        start = time.time()
    
    assert isinstance(test_predictions,np.ndarray),f'type(test_predictions)={type(test_predictions)}'
    assert isinstance(intervals,np.ndarray),f'type(intervals)={type(intervals)}'
    assert len(intervals)==len(test_predictions),f'len(intervals)={len(intervals)} vs. len(test_predictions)={len(test_predictions)}'

    if any(np.isnan(test_predictions)):
        check_consistency_of_nan_acp_predictions_and_expected_inf_intervals(test_predictions,intervals)

        test_predictions = np.array([replace_nan_with_zero(v) for v in test_predictions.tolist()])

    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start)
        del task,end,start

    return test_predictions

def compute_prediction_as_the_midpoint_of_prediction_intervals(prediction_intervals):
    ########################
    #RLMR:
    #This appears to be consistent with the previous literature on ACP for regression applied to cheminformatics problems:
    #https://pubs.acs.org/doi/full/10.1021/acs.molpharmaceut.7b00007
    #Corresponding code: https://github.com/MartinLindh/CPSP/blob/master/conformal_skin_prediction.py
    #Looking at def predict_from_smiles_conformal_median(...) in that code also suggests that this approach was used to compute ACP predictions from the prediction intervals, for which the lower and upper bounds are the medians of the lower and upper bounds from all of the different ICP models from multiple train/calibration splits. However, this code was not adapted for our work.
    #######################
    
    if monitor_time:
        start = time.time()

    predictions = np.median(prediction_intervals,axis=1)

    predictions = parse_acp_nan_predictions(test_predictions=predictions,intervals=prediction_intervals)

    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start)
        del task,end,start

    return predictions

def applyConformalRegressionToNewCmpds_ACP(ACP_dict_of_confScores,ACP_dict_of_model,X_test,sig_level=0.32,ml_alg="RandomForestRegressor",nrTrees=100,non_conformity_scaling="exp(stdev of tree predictions)"):
    
    if monitor_time:
        start = time.time()

    sigLevels = np.array([sig_level])#np.linspace(0.01, .99, 100)
    
    i = 0 
    
    n_source = max(list(ACP_dict_of_confScores.keys()))+1
    
    intervals = np.zeros((len(sigLevels), n_source, X_test.shape[0], 2))
    
    for sourceIndex in range(0,n_source):
        
        icp_testPred,icp_intervals = applyConformalRegressionToNewCmpds_ICP(confScores=ACP_dict_of_confScores[sourceIndex],model=ACP_dict_of_model[sourceIndex],X_test=X_test,sig_level=sig_level,ml_alg=ml_alg,nrTrees=nrTrees,non_conformity_scaling=non_conformity_scaling)
        
        intervals[i, sourceIndex, :, :] = icp_intervals
        
           
    combined_intervals = aggregateIntervals(intervals[i])
    
    testPred = compute_prediction_as_the_midpoint_of_prediction_intervals(prediction_intervals=combined_intervals).tolist()

    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start)
        del task,end,start
    
    return testPred,combined_intervals

def getConformalRegressionModelsPlusCalibDetails_SCP(train_inc_calib_x,train_inc_calib_y,global_random_seed,number_of_train_splits=5,calib_fraction=consistent_default_calib_fraction,ml_alg="RandomForestRegressor",nrTrees=100,non_conformity_scaling="exp(stdev of tree predictions)",stratified=True):
    #####################
    #RLMR:
    #Updates made based upon "Algorithm 1: Synergy Conformal Predictor for Regression with data partitioning." in http://dx.doi.org/10.5220/0010229402120221
    #Specifically, rather than average the calibration set predictions and then use these to compute the non-coformity scores, the non-coformity scores are computed for each SCP model and then averaged prior to computing the prediction intervals
    #####################
    
    
    train_x,train_y,calib_x,calib_y = singleRandomSplit(data_x=train_inc_calib_x,data_y=train_inc_calib_y,test_fraction=calib_fraction,random_state=global_random_seed,stratified=stratified)
    
    #=============================
    X_train = train_x
    y_train = train_y
    X_calib = calib_x
    y_calib = calib_y
    #===============================
    #This is needed:
    n_source = number_of_train_splits
    #=================================
    
    SCP_dict_of_model = {}
    
    # Populate the training set with the right indices
    nrTrainCases = len(y_train)
    
    #control the randomness of the partition splitting for the calibration and the training
    random.seed(global_random_seed)
    
    randIndex = random.sample(list(range(0, nrTrainCases)), nrTrainCases)
    #---------------------
    #Checking:
    assert isinstance(randIndex,list)
    assert len(randIndex) == nrTrainCases
    assert len(randIndex)==len(set(randIndex))
    #----------------------

    splitLen = int(nrTrainCases / n_source)

    #=========
    #RLMR: Some indices originally went missing if no_extra_indices>0!
    no_extra_indices = len(randIndex)-(splitLen*n_source)
    #==========

    # split training data into equal parts
    trainIndex = randIndex[0:splitLen]

    array_used_to_compute_meanCalibConfScores = np.zeros(len(y_calib)) #RLMR: adapted from multiPart.py of https://github.com/pharmbio/SCPRegression/tree/master
    
    #-------------------------------
    #For checking:
    old_trainIndex = None
    all_train_indices = []
    #--------------------------------

    for indexSrc in range(0, n_source):
        #----------------------
        #Checking:
        assert isinstance(trainIndex,list)
        assert (len(trainIndex) == splitLen or len(trainIndex)==(splitLen+no_extra_indices))
        assert len(trainIndex)==len(set(trainIndex))
        if not old_trainIndex is None:
            assert not old_trainIndex == trainIndex
        all_train_indices += trainIndex
        #----------------------

        # Select the samples to be picked for the training set
        sourceData = X_train.iloc[trainIndex]
        sourceTarget = y_train.iloc[trainIndex]

        #=========================
        sourceData.reset_index(drop=True,inplace=True)
        sourceTarget.reset_index(drop=True,inplace=True)
        #=========================

        #----------
        #Checking:
        assert sourceData.shape[0]==len(trainIndex)
        assert sourceData.shape[1] == X_train.shape[1]
        assert sourceTarget.shape[0] == len(trainIndex)
        #----------
        
        confScores,model = ICPtrainComputeCalibSetConfScores(sourceData,X_calib,sourceTarget,y_calib,ml_alg,non_conformity_scaling,global_random_seed,nrTrees)
        
        array_used_to_compute_meanCalibConfScores = np.add(array_used_to_compute_meanCalibConfScores,confScores)
        
        SCP_dict_of_model[indexSrc] = model

        old_trainIndex = trainIndex[:]
        
        trainIndex = randIndex[splitLen * (indexSrc + 1):splitLen * (indexSrc + 2)]

        if not 0==no_extra_indices:
            if not indexSrc == (n_source-2):
                pass
            else:
                trainIndex += randIndex[-no_extra_indices:]
    
    #---------------
    #Checking:
    assert randIndex == all_train_indices,f'randIndex={randIndex} vs. all_train_indices={all_train_indices}'
    #---------------
        
        
    meanCalibConfScores = array_used_to_compute_meanCalibConfScores / n_source
    
    confScores = pd.Series(meanCalibConfScores)
    
    return confScores,SCP_dict_of_model

def applyConformalRegressionToNewCmpds_SCP(confScores,SCP_dict_of_model,X_test,sig_level=0.32,ml_alg="RandomForestRegressor",nrTrees=100,non_conformity_scaling="exp(stdev of tree predictions)"):
    
    n_source = max(list(SCP_dict_of_model.keys()))+1
    
    #meanTestPred = np.zeros(len(y_test)) #RLMR: This does not actually seem necessary?
    
    if "RandomForestRegressor" == ml_alg and "exp(stdev of tree predictions)" == non_conformity_scaling:
        all_iterations_y_test_std = np.zeros(X_test.shape[0])#len(y_test))
    else:
        raise Exception("Conformal regression is only currently supported for Random Forest regression and non-coformity scaling using exp(stdev of tree predictions)")
    
    # Run the ICP regression for n_sources of division
    testPred_list = []
    
    #y_test_list = []
    
    #y_test_std_list = []
    
    for indexSrc in range(0, n_source):
        
        model = SCP_dict_of_model[indexSrc]
        
        nth_test_predictions = model.predict(X_test)
        
        #meanTestPred = np.add(meanTestPred, testPred)
        
        testPred_list.append(nth_test_predictions)#testPred)
        
        #y_test_list.append(y_test.tolist())
        
        if "RandomForestRegressor" == ml_alg and "exp(stdev of tree predictions)" == non_conformity_scaling:
            
            y_test_std = icp.compute_y_std(X=X_test, numberTrees=nrTrees,RF_reg_model=model)
            
            #y_test_std_list.append(y_test_std)
            
            all_iterations_y_test_std  = np.add(all_iterations_y_test_std , y_test_std)
        else:
            raise Exception("Conformal regression is only currently supported for Random Forest regression and non-coformity scaling using exp(stdev of tree predictions)")

    # Averaging the predictions according to n_sources
    
    if "RandomForestRegressor" == ml_alg and "exp(stdev of tree predictions)" == non_conformity_scaling:
        mean_y_test_std = all_iterations_y_test_std  / n_source
        
        lamda_test = icp.lamda_exp_std(mean_y_test_std, w=1)
    else:
        raise Exception("Conformal regression is only currently supported for Random Forest regression and non-coformity scaling using exp(stdev of tree predictions)")
    
    testPred = np.mean(testPred_list, axis=0).tolist()
    
    intervals = icp.computeInterval(confScores, testPred, sig_level, lamda_test)
    
    return testPred,intervals


def getNativeMLRegModel(train_x,train_y,global_random_seed,ml_alg="RandomForestRegressor",nrTrees=100):
    
    if "RandomForestRegressor" == ml_alg:
        model = icp.fit_RF(X_train=train_x, y_train=train_y,nrTrees=nrTrees, global_random_seed=global_random_seed)
    else:
        raise Exception("Conformal regression is only currently supported for Random Forest regression")
    
    return model

def checkIntervalsContainPreds(testPred,intervals):
    if monitor_time:
        start = time.time()
    
    #==============================
    assert isinstance(testPred,list),f'type(testPred)={type(testPred)}'
    assert len(testPred) == len(intervals),"len(testPred.tolist()) = {} vs. len(intervals) = {}".format(len(testPred.tolist()),len(intervals))
    #=============================
    
    for i in range(0,len(intervals)):
        lower = intervals[i,0]
        
        upper = intervals[i,1]
        
        pred = testPred[i]
        
        assert upper >= lower,"lower = {} upper = {}".format(lower,upper)
        assert pred >= lower and pred <= upper,"pred = {} lower = {} upper = {}".format(pred,lower,upper)
    
    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start)
        del task,end,start

def applyNativeMLRegModelToNewCmpds(X_test,model,ml_alg="RandomForestRegressor",uncertainty_alg="ENS",percentile=68,calc_preds_once=True,ignore_user_warning=False):
    
    testPred = model.predict(X_test)
    
    if "RandomForestRegressor" == ml_alg and "ENS" == uncertainty_alg:
        if calc_preds_once:
            intervals = rfPredInts_ens(ensemble_model=model,instances_X=X_test,confidence=percentile,pre_calc_ensemble_preds=testPred,ignore_user_warning=ignore_user_warning)
        else:
            intervals = rfPredInts_ens(ensemble_model=model,instances_X=X_test,confidence=percentile,ignore_user_warning=ignore_user_warning)
    elif "RandomForestRegressor" == ml_alg and "PI" == uncertainty_alg:
        intervals = rfPredInts(model=model, X=X_test, y_pred=testPred, ci_percentage=percentile,ignore_user_warning=ignore_user_warning)
    else:
        raise Exception("Native ML regression prediction intervals is only currently supported for Random Forest regression and ENS or PI approaches!")
    
    return testPred.tolist(),intervals


def getRegModellingAndCalibrationResults(uncertainty_method,non_conformity_scaling,ml_alg,train_inc_calib_x,train_inc_calib_y,global_random_seed,number_of_scp_splits=5,number_of_acp_splits=100,icp_calib_fraction=consistent_default_calib_fraction,nrTrees=100,stratified=True):
    
    if monitor_time:
        start = time.time()

    #-------------------------
    assert isinstance(train_inc_calib_x,pd.DataFrame),f'type(train_inc_calib_x)={type(train_inc_calib_x)}'
    assert isinstance(train_inc_calib_y,pd.Series),f'type(train_inc_calib_y)={type(train_inc_calib_y)}'
    #-------------------------

    modelling_calib_res_dict = {}
    
    if "Native" == uncertainty_method:
        native_model = getNativeMLRegModel(train_x=train_inc_calib_x,train_y=train_inc_calib_y,ml_alg=ml_alg,nrTrees=nrTrees,global_random_seed=global_random_seed)
        
        modelling_calib_res_dict['native_model'] = native_model
    
    elif "ICP" == uncertainty_method:
        confScores,icp_model = getConformalRegressionModelsPlusCalibDetails_ICP(train_inc_calib_x=train_inc_calib_x,train_inc_calib_y=train_inc_calib_y,calib_fraction=icp_calib_fraction,ml_alg=ml_alg,nrTrees=nrTrees,non_conformity_scaling=non_conformity_scaling,stratified=stratified,global_random_seed=global_random_seed)
        
        modelling_calib_res_dict['icp_confScores'] = confScores
        modelling_calib_res_dict['icp_model'] = icp_model
    
    elif "ACP" == uncertainty_method:
        ACP_dict_of_confScores,ACP_dict_of_model = getConformalRegressionModelsPlusCalibDetails_ACP(train_inc_calib_x=train_inc_calib_x,train_inc_calib_y=train_inc_calib_y,number_of_calib_splits=number_of_acp_splits,ml_alg=ml_alg,nrTrees=nrTrees,non_conformity_scaling=non_conformity_scaling,stratified=stratified,global_random_seed=global_random_seed,calib_fraction=icp_calib_fraction)
        
        modelling_calib_res_dict['ACP_dict_of_confScores'] = ACP_dict_of_confScores
        modelling_calib_res_dict['ACP_dict_of_model'] = ACP_dict_of_model
    
    elif "SCP" == uncertainty_method:
        confScores,SCP_dict_of_model = getConformalRegressionModelsPlusCalibDetails_SCP(train_inc_calib_x=train_inc_calib_x,train_inc_calib_y=train_inc_calib_y,number_of_train_splits=number_of_scp_splits,calib_fraction=icp_calib_fraction,ml_alg=ml_alg,nrTrees=nrTrees,non_conformity_scaling=non_conformity_scaling,stratified=stratified,global_random_seed=global_random_seed)
        
        modelling_calib_res_dict['scp_confScores'] = confScores
        modelling_calib_res_dict['SCP_dict_of_model'] = SCP_dict_of_model
        
    else:
        raise Exception('Unexpected uncertainty_method={}'.format(uncertainty_method))
    
    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start)
        del task,end,start
    
    return modelling_calib_res_dict




def getRegTestPredictionsPlusIntervals(uncertainty_method,non_conformity_scaling,ml_alg,nrTrees,modelling_calib_res_dict,test_x,sig_level,native_uncertainty_alg_variant,check_pred_intervals=True):
    
    if monitor_time:
        start = time.time()

    #--------------------
    assert isinstance(test_x,pd.DataFrame),f'type(test_x)={type(test_x)}'
    #--------------------

    if "Native" == uncertainty_method:
        conf_interval_percent = int(round(100*(1-sig_level),2))
        
        test_predictions,test_prediction_intervals = applyNativeMLRegModelToNewCmpds(X_test=test_x,model=modelling_calib_res_dict['native_model'],ml_alg=ml_alg,uncertainty_alg=native_uncertainty_alg_variant,percentile=conf_interval_percent,calc_preds_once=None)
    
    elif "ICP" == uncertainty_method:
        test_predictions,test_prediction_intervals = applyConformalRegressionToNewCmpds_ICP(confScores=modelling_calib_res_dict['icp_confScores'],model=modelling_calib_res_dict['icp_model'],X_test=test_x,sig_level=sig_level,ml_alg=ml_alg,nrTrees=nrTrees,non_conformity_scaling=non_conformity_scaling)
    
    elif "ACP" == uncertainty_method:
        test_predictions,test_prediction_intervals = applyConformalRegressionToNewCmpds_ACP(ACP_dict_of_confScores=modelling_calib_res_dict['ACP_dict_of_confScores'],ACP_dict_of_model=modelling_calib_res_dict['ACP_dict_of_model'],X_test=test_x,sig_level=sig_level,ml_alg=ml_alg,nrTrees=nrTrees,non_conformity_scaling=non_conformity_scaling)
    
    elif "SCP" == uncertainty_method:
        
        test_predictions,test_prediction_intervals = applyConformalRegressionToNewCmpds_SCP(confScores=modelling_calib_res_dict['scp_confScores'],SCP_dict_of_model= modelling_calib_res_dict['SCP_dict_of_model'],X_test=test_x,sig_level=sig_level,ml_alg=ml_alg,nrTrees=nrTrees,non_conformity_scaling=non_conformity_scaling)
        
    else:
        raise Exception('Unexpected uncertainty_method={}'.format(uncertainty_method))
    
    if check_pred_intervals:
        checkIntervalsContainPreds(test_predictions,test_prediction_intervals)
    
    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start)
        del task,end,start

    return test_predictions,test_prediction_intervals

def getTestPredictionsAndPredictionIntervalsForAllSignificanceLevels(all_sig_levels_considered_to_compute_ECE,uncertainty_method,train_inc_calib_x,train_inc_calib_y,test_x,non_conformity_scaling,ml_alg,global_random_seed,number_of_scp_splits,number_of_acp_splits,icp_calib_fraction,nrTrees,native_uncertainty_alg_variant,allow_for_acp_predictions_could_be_inconsistent=False,debug=False,switch_on_monitor_time=False):
    global monitor_time
    monitor_time = switch_on_monitor_time

    if monitor_time:
        start = time.time()
    
    #=======================================
    sig_level_2_prediction_intervals = {}
    sig_level_2_test_predictions = {} #For ACP, the prediction could depend upon the prediction interval, and hence the significance level, as this is set to the median of the intervals!
    
    modelling_calib_res_dict = getRegModellingAndCalibrationResults(uncertainty_method,non_conformity_scaling,ml_alg,train_inc_calib_x,train_inc_calib_y,global_random_seed,number_of_scp_splits,number_of_acp_splits,icp_calib_fraction,nrTrees)
    
    last_test_predictions = None

    for sig_level in all_sig_levels_considered_to_compute_ECE:
        
        test_predictions,test_prediction_intervals = getRegTestPredictionsPlusIntervals(uncertainty_method,non_conformity_scaling,ml_alg,nrTrees,modelling_calib_res_dict,test_x,sig_level,native_uncertainty_alg_variant)
        
        if monitor_time:
            check_start = time.time()

        #========================================
        if not last_test_predictions is None:
            if not (allow_for_acp_predictions_could_be_inconsistent and "ACP" == uncertainty_method): #This could happen if the prediction intervals were made infinitely large
                assert last_test_predictions == test_predictions,f"uncertainty_method={uncertainty_method}, sig_level={sig_level}, last_test_predictions = {last_test_predictions}, test_predictions = {test_predictions}"
        last_test_predictions = test_predictions
        #========================================

        if monitor_time:
            check_end = time.time()

            check_task = f'check predictions consistency for significance-level = {sig_level}'

            basic_time_task(check_task,check_end,check_start)
            del check_task,check_end,check_start

        sig_level_2_test_predictions[sig_level] = test_predictions
        
        sig_level_2_prediction_intervals[sig_level] = test_prediction_intervals
    #========================================
    
    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start)
        del task,end,start

    return sig_level_2_test_predictions,sig_level_2_prediction_intervals
#=