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
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,GridSearchCV
#-----------------------------------
from .IVAP_CVAP_workflow_functions import commonInitStepsForIVAPandCVAP,getScoresForClass1,predictedBinaryClassFromProbClass1
from ..utils.basic_utils import findDups as reportDuplicates
#------------------------------------

def buildNativeMLModel(train_inc_calib_x,train_inc_calib_y,ml_alg,class_1,rand_seed,check_binary_classification=True,non_default_hp_to_search=None,hp_opt_cv=2,hp_opt_metric="balanced_accuracy"):
    
    clf,train_inc_calib_y_as_1s_and_0s = commonInitStepsForIVAPandCVAP(train_inc_calib_y,ml_alg,check_binary_classification,class_1,rand_seed,non_default_hp_to_search,hp_opt_cv,hp_opt_metric)
    
    train_x = train_inc_calib_x
    train_y = train_inc_calib_y_as_1s_and_0s
    
    clf.fit(train_x,train_y)
    
    model = clf
    
    return model

def getPredictedClassesAndClass1ProbsForNativeModel(model,ml_alg,data_x,always_normalized=True):
    
    probs_of_class_1 = getScoresForClass1(model,data_x,ml_alg,always_normalized=always_normalized)
    
    if always_normalized or "RandomForestClassifier" == ml_alg:
        predicted_classes = [predictedBinaryClassFromProbClass1(prob_class_1=p) for p in probs_of_class_1]
    else:
        raise Exception('Unrecognised scenario: ml_alg=%s' % ml_alg)
    
    return predicted_classes,probs_of_class_1

def getId2PredictedClassAndClass1ProbsForNativeModel(model,ml_alg,data_x,data_ids):
    #=============================
    if not type([])==type(data_ids): raise Exception('type(data_ids)=%s' % str(type(data_ids)))
    if not len(data_ids)==data_x.shape[0]: raise Exception('len(data_ids)=%d,data_x.shape[0]=%d' % (len(data_ids),data_x.shape[0]))
    if not len(data_ids)==len(set(data_ids)): raise Exception('Duplicates in data_ids=%s' % str(reportDuplicates(data_ids)))
    #=============================
    
    predicted_classes,probs_of_class_1 = getPredictedClassesAndClass1ProbsForNativeModel(model,ml_alg,data_x)
    
    id_to_pred_class_class_1_prob = defaultdict(dict)
    
    for index in range(0,len(data_ids)):
        id_to_pred_class_class_1_prob[data_ids[index]]['Class_1_prob'] = probs_of_class_1[index]
        id_to_pred_class_class_1_prob[data_ids[index]]['Predicted_Binary_Class'] = predicted_classes[index]
    
    return id_to_pred_class_class_1_prob

def getNativeModelAndTestId2PredClassAndClass1Prob(train_inc_calib_x,train_inc_calib_y,ml_alg,rand_seed,class_1,test_x=None,test_ids=None):
    model = buildNativeMLModel(train_inc_calib_x,train_inc_calib_y,ml_alg,class_1,rand_seed)
    
    if not test_x is None:
    
        native_test_id_to_pred_class_class_1_prob = getId2PredictedClassAndClass1ProbsForNativeModel(model,ml_alg,data_x=test_x,data_ids=test_ids)
    else:
        native_test_id_to_pred_class_class_1_prob = None
        
    return model,native_test_id_to_pred_class_class_1_prob
