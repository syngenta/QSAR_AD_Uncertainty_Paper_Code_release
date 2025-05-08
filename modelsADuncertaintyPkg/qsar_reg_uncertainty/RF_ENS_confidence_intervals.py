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
############################################################
import numpy as np
import pandas as pd
from scipy import stats #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html : C% confidence interval of of a zero mean centered distribution with standard deviation SD is given by stats.norm.interval(confidence=C,loc=0,scale=SD)
from ..utils.ML_utils import get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance
from ..utils.basic_utils import get_pandas_df_row_as_df

def ensemblePredictionVarianceAndPrediction(ensemble_model,instances_X,do_not_compute_preds=False,ignore_user_warning=False):
    ##############################################
    #c.f. Wang et al. (2021) [ENS uncertainty estimates], this may be considered to be an estimate of the epistemic contribution to the prediction error variance (the epistemic uncertainty)
    ##############################################
    #===================
    assert isinstance(instances_X,pd.DataFrame),type(instances_X)
    assert instances_X.index.tolist()==list(range(instances_X.shape[0])),instances_X.index.tolist()
    #===================
    
    
    if not do_not_compute_preds:
        ensemble_preds = []
    else:
        ensemble_preds = None
    
    ensemble_vars = []
    
    list_of_all_trees = [tree for tree  in ensemble_model.estimators_]
    
    for index in instances_X.index.tolist():
        
        
        test_instance_row_x = get_pandas_df_row_as_df(pandas_df=instances_X,row_index=index)
        
        
        if not do_not_compute_preds:
            ensemble_preds.append(ensemble_model.predict(test_instance_row_x))
        
        
        base_model_predictions = [get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance(tree=list_of_all_trees[tree_index],ensemble=ensemble_model,tree_index=tree_index,test_X=test_instance_row_x,expected_type_of_test_X=pd.DataFrame,want_class_probs=False,ignore_user_warning=ignore_user_warning) for tree_index in range(0,len(list_of_all_trees))]
        
        ensemble_vars.append(np.std(base_model_predictions)**2)
    
    return ensemble_vars,ensemble_preds

def estimateConfidenceIntervalOfErrors(ensemble_vars,confidence=68):
    
    #========================
    assert confidence >=0 and confidence <=100,confidence
    #========================
    
    ci_list = []
    
    for index in range(0,len(ensemble_vars)):
        var = ensemble_vars[index]
        
        ci = stats.norm.interval(alpha=confidence/100.0,loc=0,scale=np.sqrt(var)) #confidence interval values should be in range 0 - 1: https://stackoverflow.com/questions/55091757/r-python-confidence-interval
        ci = [round(v,3) for v in list(ci)]
        #------------------
        assert ci[0] <= 0 and ci[1] >=0,"ci = {}".format(ci)
        assert abs(ci[0])==abs(ci[1]),"ci = {}".format(ci) #Before rounding: AssertionError: ci = [-0.7750916473263539, 0.7750916473263543]
        #------------------
        
        ci_list.append(ci)
    
    return ci_list

def getPredIntervals(ensemble_preds,ci_list):
    intervals = np.zeros((len(ensemble_preds), 2))
    
    for index in range(0,len(ensemble_preds)):
        ci = ci_list[index]
        #--------------------
        assert type([])==type(ci),"type(ci) = {}".format(type(ci))
        assert 2 == len(ci)
        assert ci[0] <= ci[1]
        #--------------------
        
        for i in range(0,2):
            intervals[index, i] = ensemble_preds[index] + ci[i] #mean centered distribution will have positive and negative ci upper and lower bounds respectively!
        
    
    return intervals

def estimateConfidenceIntervalOfExperimentalValues(ensemble_model,instances_X,confidence=68,pre_calc_ensemble_preds=None,ignore_user_warning=False):
    
    ensemble_vars,ensemble_preds = ensemblePredictionVarianceAndPrediction(ensemble_model,instances_X,do_not_compute_preds=(not pre_calc_ensemble_preds is None),ignore_user_warning=ignore_user_warning)
    
    if ensemble_preds is None:
        ensemble_preds = pre_calc_ensemble_preds
    
    
    ci_list = estimateConfidenceIntervalOfErrors(ensemble_vars,confidence=confidence)
    
    intervals = getPredIntervals(ensemble_preds,ci_list)
    
    return intervals
