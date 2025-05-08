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
#Copyright (c) Syngenta 2022-2023
#Contact richard.marchese_robinson [at] syngenta.com
#######################
import sys,re,os,functools,math
import numpy as np
import pandas as pd
#import cPickle as pickle
import pickle
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV
from copy import deepcopy
#======================================
from .venn_abers_single_probability import computeVennAbersClassOneLowerUpperProbs,computeSingleClassOneProb,computeCVAPeffectiveP0P1values
from ..utils.ML_utils import convertClassLabel,predictedBinaryClassFromProbClass1,getKFoldCVTrainOtherXY,probOK,unexpectedScore,getClass1ProbsRFModel,isRFClass1ScoreConsistent,getScoresForClass1
#======================================

def getIVAPInputForASingleCalibrationSet(class_1_scores,true_class_labels):
    #==========================
    #Convert types in case necessary:
    class_1_scores = list(class_1_scores)
    true_class_labels = list(true_class_labels)
    #============================
    #================
    #Checking
    if not len(class_1_scores) == len(true_class_labels): raise Exception('len(class_1_scores)=%d, len(true_class_labels)=%d' % (len(class_1_scores),len(true_class_labels)))
    #================
    
    calib_score_label_tuples = [(class_1_scores[i],true_class_labels[i]) for i in range(0,len(true_class_labels))]
    
    return calib_score_label_tuples

def getIVAPprobabilityEstimatesForASingleNewCmpd(calib_score_label_tuples,new_cmpd_class_1_score):
    
    current_lower_p_value,current_upper_p_value = computeVennAbersClassOneLowerUpperProbs(calib_score_label_tuples,new_cmpd_class_1_score)
    
    current_single_p_value = computeSingleClassOneProb(current_lower_p_value,current_upper_p_value)
    
    return current_single_p_value,current_lower_p_value,current_upper_p_value

def getCVAPestimatedProbabiltiesFromKfoldCrossValIVAPestimatedProbabilities(NewCmpdFoldNo2IVAPp0p1ValsDict):
    
    list_of_p0_values = []
    list_of_p1_values = []
    
    for current_k in NewCmpdFoldNo2IVAPp0p1ValsDict.keys():
        p0 = NewCmpdFoldNo2IVAPp0p1ValsDict[current_k]['p0']
        list_of_p0_values.append(p0)
        
        p1 = NewCmpdFoldNo2IVAPp0p1ValsDict[current_k]['p1']
        list_of_p1_values.append(p1)
        
    effective_p0,effective_p1 = computeCVAPeffectiveP0P1values(list_of_p0_values,list_of_p1_values)
    
    p = computeSingleClassOneProb(current_lower_p_value=effective_p0,current_upper_p_value=effective_p1)
    
    return p,effective_p1,effective_p0

def convertClassLabels(y_array,class_1,check_binary_classification=True):
    #######################
    #Some initial checks:
    unique_class_labels = list(set(y_array.tolist()))
    if not class_1 in unique_class_labels: raise Exception("class_1=%s is not found in the class labels, which have these unique values=%s" % (class_1,str(unique_class_labels)))
    
    if check_binary_classification:
        if not 2 == len(unique_class_labels): raise Exception(str(unique_class_labels))
    #######################
    
    y_as_1s_and_0s = pd.Series(np.array([convertClassLabel(raw_class_label,class_1) for raw_class_label in y_array.tolist()]))
    
    return y_as_1s_and_0s

def specifyNativeMLModel(ml_alg,rand_seed,non_default_hp_to_search=None,hp_opt_cv=2,hp_opt_metric="balanced_accuracy"): #hp_opt_cv=2 is based upon conclusions of Baumann & Baumann 2014 (https://jcheminf.biomedcentral.com/articles/10.1186/s13321-014-0047-1)
    
    if 'RandomForestClassifier' == ml_alg:
        ml = RandomForestClassifier(random_state=rand_seed)#,n_estimators=n_trees,class_weight=imbalanced_data_approach)
        
        if non_default_hp_to_search is None:
            parameters = {}
            ##############################
            #This is based upon Mervin et al. (2020) [https://pubs.acs.org/doi/full/10.1021/acs.jcim.0c00476]
            #However, save for the class_weight = balanced, which can be expected to be generally useful for typically imbalanced cheminformatics datasets, the hyperparameters are the same as SciKit-Learn defaults, i.e. little evidence of tuning!
            parameters['n_estimators']=[101] #set number of trees to an odd number, rather than 100 in Mervin et al., to avoid classification ties
            parameters['max_depth'] = [None] #"auto" appears to be deprecated https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            parameters['max_features']=["sqrt"] #"auto" appears to be deprecated, as of version 1.1, and "If “auto”, then max_features=sqrt(n_features)" https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            parameters["class_weight"] = ["balanced"]
            ###########################
        else:
            parameters = non_default_hp_to_search
    else:
        raise Exception('The following ML algorithm is not currently supported: %s' % ml_alg)
    
    clf = GridSearchCV(estimator=ml,param_grid=parameters,cv=hp_opt_cv,scoring=hp_opt_metric)
    
    return clf

def commonInitStepsForIVAPandCVAP(train_inc_calib_y,ml_alg,check_binary_classification,class_1,rand_seed,non_default_hp_to_search=None,hp_opt_cv=2,hp_opt_metric="balanced_accuracy"):
    
    train_inc_calib_y_as_1s_and_0s = convertClassLabels(y_array=train_inc_calib_y,class_1=class_1,check_binary_classification=check_binary_classification)
    
    clf = specifyNativeMLModel(ml_alg=ml_alg,rand_seed=rand_seed,non_default_hp_to_search=non_default_hp_to_search,hp_opt_cv=hp_opt_cv,hp_opt_metric=hp_opt_metric)
    
    return clf,train_inc_calib_y_as_1s_and_0s

def singleTrainCalibSplit(train_inc_calib_x,train_inc_calib_y_as_1s_and_0s,rand_seed,ivap_calib_fraction=0.2):
    splitter = StratifiedShuffleSplit(n_splits=1,random_state=rand_seed,test_size=ivap_calib_fraction)
    
    split_count = 0
    
    for train_indices,calib_indices in splitter.split(train_inc_calib_x,train_inc_calib_y_as_1s_and_0s):
        split_count += 1
        
        train_x = train_inc_calib_x.iloc[train_indices,:]
        train_y = train_inc_calib_y_as_1s_and_0s.iloc[train_indices]
        calib_x = train_inc_calib_x.iloc[calib_indices,:]
        calib_y = train_inc_calib_y_as_1s_and_0s.iloc[calib_indices]
    
    #------------------------
    if not 1 == split_count: raise Exception('split_count should equal 1 for singleTrainCalibSplit(...)! split_count=%d' % split_count)
    #-----------------------
    
    return train_x,train_y,calib_x,calib_y

def getIVAPmodelPlusCalibDetails(train_inc_calib_x,train_inc_calib_y,class_1,rand_seed,ml_alg='RandomForestClassifier',check_binary_classification=True,ivap_calib_fraction=0.2):
    
    clf,train_inc_calib_y_as_1s_and_0s = commonInitStepsForIVAPandCVAP(train_inc_calib_y,ml_alg,check_binary_classification,class_1,rand_seed)
    
    train_x,train_y,calib_x,calib_y = singleTrainCalibSplit(train_inc_calib_x,train_inc_calib_y_as_1s_and_0s,rand_seed,ivap_calib_fraction)
    
    clf.fit(train_x,train_y)
    
    IVAP_model = clf
    
    calib_scores_for_class_1 = getScoresForClass1(model=IVAP_model,data_x=calib_x,ml_alg=ml_alg)
    
    IVAP_calib_score_label_tuples = getIVAPInputForASingleCalibrationSet(class_1_scores=calib_scores_for_class_1,true_class_labels=calib_y)
    
    return IVAP_calib_score_label_tuples,IVAP_model

def applyIVAPtoTestSet(IVAP_calib_score_label_tuples,IVAP_model,test_x,test_ids,ml_alg,bin_class_pred_prob_thresh=0.5):
    
    test_scores_for_class_1 = getScoresForClass1(model=IVAP_model,data_x=test_x,ml_alg=ml_alg)
    
    #================================
    #Checking:
    if not len(test_scores_for_class_1) == len(test_ids): raise Exception('len(test_scores_for_class_1) =%d,len(test_ids)=%d' % (len(test_scores_for_class_1),len(test_ids)))
    if not len(test_ids)==len(set(test_ids)): raise Exception('These test_ids are not unique - %s' % str(test_ids))
    #===================================
    
    IVAP_test_id_to_pred_class_class_1_prob_p1_p0 = {}#defaultdict(dict)
    
    for i in range(0,len(test_scores_for_class_1)):
        prob,p0,p1 = getIVAPprobabilityEstimatesForASingleNewCmpd(calib_score_label_tuples=IVAP_calib_score_label_tuples,new_cmpd_class_1_score=test_scores_for_class_1[i])
        
        #===========================
        #Checking:
        if not (all([probOK(p) for p in [prob,p0,p1]]) and p0 < p1 and prob > p0 and prob < p1):
            raise Exception('prob = %f, p0=%f, p1=%f' % (prob,p0,p1))
        #===========================
        
        IVAP_test_id_to_pred_class_class_1_prob_p1_p0[test_ids[i]] = {}
        
        
        IVAP_test_id_to_pred_class_class_1_prob_p1_p0[test_ids[i]]['Class_1_prob'] = prob
        IVAP_test_id_to_pred_class_class_1_prob_p1_p0[test_ids[i]]['p0'] = p0
        IVAP_test_id_to_pred_class_class_1_prob_p1_p0[test_ids[i]]['p1'] = p1
        IVAP_test_id_to_pred_class_class_1_prob_p1_p0[test_ids[i]]['Predicted_Binary_Class'] = predictedBinaryClassFromProbClass1(prob_class_1=prob,thresh=bin_class_pred_prob_thresh)
    
    
    return IVAP_test_id_to_pred_class_class_1_prob_p1_p0

def getCVAPmodelPlusCalibDetails(train_inc_calib_x,train_inc_calib_y,class_1,rand_seed,ml_alg,no_cvap_folds=5,check_binary_classification=True,calib_set_name='Calibration'):
    
    clf,train_inc_calib_y_as_1s_and_0s = commonInitStepsForIVAPandCVAP(train_inc_calib_y,ml_alg,check_binary_classification,class_1,rand_seed)
    
    fold2TrainCalibXY = getKFoldCVTrainOtherXY(data_x=train_inc_calib_x,data_y=train_inc_calib_y_as_1s_and_0s,other_name=calib_set_name,no_folds=no_cvap_folds,rand_seed=rand_seed)
    
    CVAP_models_per_fold = {}
    
    CVAP_calib_score_label_tuples_per_fold = {}
    
    for fold in fold2TrainCalibXY.keys():
        
        try:
            m = clf.best_estimator_
            raise Exception('We should not be modifying clf directly!')
        except AttributeError:
            pass
        
        fold_model = deepcopy(clf)
        
        train_x = fold2TrainCalibXY[fold]['Train_x']
        train_y = fold2TrainCalibXY[fold]['Train_y']
        calib_x = fold2TrainCalibXY[fold]['%s_x' % calib_set_name]
        calib_y = fold2TrainCalibXY[fold]['%s_y' % calib_set_name]
        
        fold_model.fit(train_x,train_y)

        #=======================
        #Just for checking the fit worked
        m = fold_model.best_estimator_
        del m
        #=========================
        
        CVAP_models_per_fold[fold] = fold_model
        
        calib_scores_for_class_1 = getScoresForClass1(model=fold_model,data_x=calib_x,ml_alg=ml_alg)
        
        CVAP_calib_score_label_tuples_per_fold[fold] = getIVAPInputForASingleCalibrationSet(class_1_scores=calib_scores_for_class_1,true_class_labels=calib_y)
    
    
    return CVAP_calib_score_label_tuples_per_fold,CVAP_models_per_fold

def getCVAPfoldSpecificP0andP1Vals(CVAP_calib_score_label_tuples_per_fold,CVAP_models_per_fold,test_x,test_ids,ml_alg):
    
    testId2FoldNo2IVAPp0p1ValsDict = defaultdict(functools.partial(defaultdict,dict))
    
    for fold in CVAP_calib_score_label_tuples_per_fold.keys():
        calib_score_label_tuples = CVAP_calib_score_label_tuples_per_fold[fold]
        
        model = CVAP_models_per_fold[fold]
        
        IVAP_test_id_to_pred_class_class_1_prob_p1_p0 = applyIVAPtoTestSet(IVAP_calib_score_label_tuples=calib_score_label_tuples,IVAP_model=model,test_x=test_x,test_ids=test_ids,ml_alg=ml_alg)
        
        for cmpd_id in test_ids:
            for prob_limit in ['p0','p1']:
                testId2FoldNo2IVAPp0p1ValsDict[cmpd_id][fold][prob_limit] = IVAP_test_id_to_pred_class_class_1_prob_p1_p0[cmpd_id][prob_limit]
    
    return testId2FoldNo2IVAPp0p1ValsDict

def applyCVAPtoTestSet(CVAP_calib_score_label_tuples_per_fold,CVAP_models_per_fold,test_x,test_ids,ml_alg,bin_class_pred_prob_thresh=0.5):
    
    testId2FoldNo2IVAPp0p1ValsDict = getCVAPfoldSpecificP0andP1Vals(CVAP_calib_score_label_tuples_per_fold,CVAP_models_per_fold,test_x,test_ids,ml_alg)
    
    CVAP_test_id_to_pred_class_class_1_prob_p1_p0 = {}#defaultdict(dict)
    
    for cmpd_id in test_ids:
        
        CVAP_test_id_to_pred_class_class_1_prob_p1_p0[cmpd_id] = {}
        
        NewCmpdFoldNo2IVAPp0p1ValsDict = testId2FoldNo2IVAPp0p1ValsDict[cmpd_id]
        
        p,effective_p1,effective_p0 = getCVAPestimatedProbabiltiesFromKfoldCrossValIVAPestimatedProbabilities(NewCmpdFoldNo2IVAPp0p1ValsDict)
        
        CVAP_test_id_to_pred_class_class_1_prob_p1_p0[cmpd_id]['p0'] = effective_p0
        CVAP_test_id_to_pred_class_class_1_prob_p1_p0[cmpd_id]['p1'] = effective_p1
        CVAP_test_id_to_pred_class_class_1_prob_p1_p0[cmpd_id]['Class_1_prob'] = p
        CVAP_test_id_to_pred_class_class_1_prob_p1_p0[cmpd_id]['Predicted_Binary_Class'] = predictedBinaryClassFromProbClass1(prob_class_1=p,thresh=bin_class_pred_prob_thresh)
    
    return CVAP_test_id_to_pred_class_class_1_prob_p1_p0 

def runIVAPandOrCVAPWorkflow(train_inc_calib_x,train_inc_calib_y,test_x,train_ids,test_ids,ml_alg,rand_seed,class_1,test_y=None,run_ivap=True,run_cvap=True,no_cvap_folds=5,ivap_calib_fraction=0.2):
    if run_ivap:
    
        IVAP_calib_score_label_tuples,IVAP_model = getIVAPmodelPlusCalibDetails(train_inc_calib_x=train_inc_calib_x,train_inc_calib_y=train_inc_calib_y,class_1=class_1,rand_seed=rand_seed,ml_alg=ml_alg,ivap_calib_fraction=ivap_calib_fraction)
        
        if not test_x is None:
            IVAP_test_id_to_pred_class_class_1_prob_p1_p0 = applyIVAPtoTestSet(IVAP_calib_score_label_tuples,IVAP_model,test_x,test_ids,ml_alg)
        else:
            IVAP_test_id_to_pred_class_class_1_prob_p1_p0 = None
    else:
        IVAP_calib_score_label_tuples = None
        IVAP_model = None
        IVAP_test_id_to_pred_class_class_1_prob_p1_p0 = None
    
    if run_cvap:
    
        CVAP_calib_score_label_tuples_per_fold,CVAP_models_per_fold = getCVAPmodelPlusCalibDetails(train_inc_calib_x=train_inc_calib_x,train_inc_calib_y=train_inc_calib_y,class_1=class_1,rand_seed=rand_seed,ml_alg=ml_alg,no_cvap_folds=no_cvap_folds)
        
        if not test_x is None:
            CVAP_test_id_to_pred_class_class_1_prob_p1_p0 = applyCVAPtoTestSet(CVAP_calib_score_label_tuples_per_fold,CVAP_models_per_fold,test_x,test_ids,ml_alg)
        else:
            CVAP_test_id_to_pred_class_class_1_prob_p1_p0 = None
    else:
        CVAP_calib_score_label_tuples_per_fold = None
        CVAP_models_per_fold = None
        CVAP_test_id_to_pred_class_class_1_prob_p1_p0 = None
    
    return IVAP_calib_score_label_tuples,IVAP_model,IVAP_test_id_to_pred_class_class_1_prob_p1_p0,CVAP_calib_score_label_tuples_per_fold,CVAP_models_per_fold,CVAP_test_id_to_pred_class_class_1_prob_p1_p0


