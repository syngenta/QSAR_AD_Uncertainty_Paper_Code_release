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
#Copyright (c)  2020-2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#Parts of this code were written by Zied Hosni, whilst working on a Syngenta funded collaboration at the University of Sheffield
########################
import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
import matplotlib
matplotlib.use('Agg')
from py import test
import seaborn as sns
#----------------------
from . import reg_perf_pred_stats as RegPred
from .reg_Uncertainty_metrics import estimated_variance_from_CP,compute_validity_efficiency,compute_ENCE,compute_ECE,index2SigLevel,sigLevels,getKeyFromValue,compute_Spearman_rank_correlation_between_intervals_width_and_residuals
from .perf_measure import ErrorRate
from ..utils.graph_utils import add_identity
from .enforce_minimum_no_instances import size_of_inputs_for_stats_is_big_enough,get_no_instances
from ..utils.ML_utils import getTestYFromTestIds as getSubsetYFromSubsetIds
#----------------------



def getIntervalsWidthOfInterest(sig_level_2_prediction_intervals,sig_level_of_interest):
    intervals = sig_level_2_prediction_intervals[sig_level_of_interest]
    
    widths_for_intervals_of_interest = abs(intervals[:, 0] - intervals[:, 1])
    
    return widths_for_intervals_of_interest

def getAllerrorRate_s(sig_level_2_prediction_intervals,y_test,index2SigLevel=index2SigLevel,sigLevels=sigLevels):
    #copied and adapted from ZH calibration_plots_nestedCV.py:
    
    errorRate_s = np.zeros(len(sigLevels))
    
    
    
    for i in index2SigLevel.keys():#range(len(sigLevels)):
        
        corresponding_sig_level = index2SigLevel[i]
        
        #assert i == (100*(corresponding_sig_level-0.01)),"i = {} vs. corresponding_sig_level = {}".format(i,corresponding_sig_level)
        
        try:
            intervals = sig_level_2_prediction_intervals[corresponding_sig_level]
        except KeyError:
            errorRate_s[i] = None
            continue
        
        try:
            errorRate_s[i] = ErrorRate(intervals=intervals, testLabels=y_test)
        except KeyError as err:
            print('intervals = {}'.format(intervals))
            print('y_test = {}'.format(y_test))
            raise Exception(err)
    
    return errorRate_s

def checkAndPrepRegMetricsInputs(y_test, testPred):
    
    #--------------------------------
    assert type(y_test) in [type(np.array([0.5])),type(pd.Series([0.5])),type([])],type(y_test)
    assert type(testPred) in [type(np.array([0.5])),type(pd.Series([0.5])),type([])],type(testPred)
    
    if not type([]) == type(testPred):
        testPred = testPred.tolist()
    
    if not type([]) == type(y_test):
        y_test = y_test.tolist()
    #--------------------------------
    
    return y_test,testPred

def getAllRegPredMetrics(y_test, testPred):
    
    y_test,testPred = checkAndPrepRegMetricsInputs(y_test, testPred)
    
    #================
    e = y_test#.tolist()
    
    p = testPred#.tolist()
    #==================
    
    rmse = RegPred.rmse(e,p)
    
    MAD = RegPred.MAD(e,p)
    
    R2 = RegPred.coeffOfDetermination(e,p)
    
    
    Pearson = RegPred.PearsonCoeff(e,p)
    
    Pearson_Pval_one_tail = RegPred.PearsonCoeffPval(e,p)
    
    
    Spearman = RegPred.SpearmanCoeff(e,p)
    
    Spearman_Pval_one_tail = RegPred.SpearmanCoeffPval(e,p)
    
    return rmse,MAD,R2,Pearson,Pearson_Pval_one_tail,Spearman,Spearman_Pval_one_tail

def getRelevantRegUncertainyMetrics(y_test, testPred,sig_level_2_prediction_intervals,sig_level_of_interest,min_no_samples_per_bin_for_ENCE=20):
     
    sig_levels_for_ECE = list(sig_level_2_prediction_intervals.keys())

    y_test,testPred = checkAndPrepRegMetricsInputs(y_test, testPred)
    
    #=================================
    assert type(0.5) == type(sig_level_of_interest),type(sig_level_of_interest)
    assert sig_level_of_interest > 0 and sig_level_of_interest < 1,sig_level_of_interest
    #=================================
    
    widths_for_intervals_of_interest = getIntervalsWidthOfInterest(sig_level_2_prediction_intervals,sig_level_of_interest)
    
    estimated_variance = estimated_variance_from_CP(intervals_width=widths_for_intervals_of_interest)
    errorRate_s = getAllerrorRate_s(sig_level_2_prediction_intervals,y_test)
    
    validity, efficiency = compute_validity_efficiency(error_rate=errorRate_s[getKeyFromValue(index2SigLevel,sig_level_of_interest)], intervals_width=widths_for_intervals_of_interest)
        
    ENCE = compute_ENCE(y_test, testPred, estimated_variance,min_no_samples_per_bin=min_no_samples_per_bin_for_ENCE)
    
    ECE_new = compute_ECE(error_rate_s=errorRate_s,sig_levels=sig_levels_for_ECE)
    
    scc = compute_Spearman_rank_correlation_between_intervals_width_and_residuals(y_true=y_test,y_pred=testPred,intervals_width=widths_for_intervals_of_interest)
    
    return validity, efficiency,ECE_new,ENCE,errorRate_s,scc

def plot_x_vs_y_with_error_bars(x,y,x_err_vals,y_err_vals,min_val,max_val,x_axis_label,y_axis_label,plot_title,plot_file_name,offset):
    #------------------------
    if not isinstance(x,list): raise Exception(f'type(x)={type(x)}')
    if not isinstance(y,list): raise Exception(f'type(y)={type(y)}')
    if not len(x)==len(y): raise Exception(f'len(x)={len(x)} vs. len(y)={len(y)}')
    #-----------------------
    
    sns.set_style("white")

    figure = pp.figure()
    
    #----------------------------------------------------
    #https://matplotlib.org/stable/api/figure_api.html?highlight=add_subplot#matplotlib.figure.Figure.add_subplot
    nrows = 1
    ncols=1
    index=1 #We just want one plot in the figure 
    axes_obj = figure.add_subplot(nrows,ncols,index)
    
    #------------------------------------------------
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
    #Scatter plot:
    axes_obj.scatter(x=x,y=y)
    #------------------------------------------------

    #------------------------------------------------
    #Add error bars:
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html
    if not (x_err_vals is None and y_err_vals is None):
        axes_obj.errorbar(x=x,y=y,xerr=x_err_vals,yerr=y_err_vals,ms=1,fmt='o')

    #------------------------------------------------
    
    #----------------------------------------------------
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
    #Add Y=X line for comparison:
    
    
    axes_obj.plot([min_val,max_val],[min_val,max_val],lw=2)
	#----------------------------------------------------
    
    #----------------------------------------------------
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html
    #Plot axis limits and labels:
    axes_obj.set(xlabel=x_axis_label,ylabel=y_axis_label,xlim=(min_val,max_val),ylim=(min_val,max_val),title=plot_title)
    #---------------------------------------------------

    pp.savefig(plot_file_name, transparent=True)
    pp.clf()

def convert_PIs_into_lower_upper_err_bars(testPred,sig_level_2_prediction_intervals,sig_level_of_interest):
    #################
    #Need to match requirements for xerr,yerr described here: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html
    #################
    prediction_intervals = sig_level_2_prediction_intervals[sig_level_of_interest]

    lower_vals = prediction_intervals[:, 0].tolist()
    upper_vals = prediction_intervals[:,1].tolist()
    
    pred_vals = np.array(testPred).tolist()

    upper_errs = [(upper_vals[i]-pred_vals[i]) for i in range(len(pred_vals))]
    lower_errs = [(pred_vals[i]-lower_vals[i]) for i in range(len(pred_vals))]

    assert all([(v>=0) for v in upper_errs])
    assert all([(v>=0) for v in lower_errs])

    potentially_asymmetric_err_bars_2d_array =  np.transpose(np.column_stack((lower_errs,upper_errs)))
    

    assert 2 == potentially_asymmetric_err_bars_2d_array.shape[0],potentially_asymmetric_err_bars_2d_array.shape[0]
    assert len(pred_vals) == potentially_asymmetric_err_bars_2d_array.shape[1],potentially_asymmetric_err_bars_2d_array.shape[1]

    return potentially_asymmetric_err_bars_2d_array,lower_vals,upper_vals

def plotPredPlusEstimatedErrorBarsVsExperi(y_test, testPred,sig_level_2_prediction_intervals,sig_level_of_interest,plot_file_name,plot_title,assume_PIs_are_symmetric=True,experi_err_sizes=None,offset=0.2):
    
    #===============================
    x = np.array(y_test).tolist()
    y = np.array(testPred).tolist()
    
    x_axis_label = 'Experimental'
    y_axis_label = 'Predicted'

    if not sig_level_2_prediction_intervals is None:
        y_err_vals,y_lower_vals,y_upper_vals = convert_PIs_into_lower_upper_err_bars(testPred,sig_level_2_prediction_intervals,sig_level_of_interest)
    else:
        print('Information on prediction intervals was not supplied!')
        y_err_vals = None
    
    x_err_vals = experi_err_sizes

    if experi_err_sizes is None:
        print('Experimental errors are unknown, hence being ignored!')
    #===============================
    
    #-------------------------------------------------
    #Set plot limits
    
    combined_vals = []

    if not x_err_vals is None:
        if not isinstance(x_err_vals,list): raise Exception(f'type(x_err_vals)={type(x_err_vals)}, e.g. asymemetric errors, is not currently supported!')
        combined_vals += [(x[i]+x_err_vals[i]) for i in range(len(x))]
        combined_vals += [(x[i]-x_err_vals[i]) for i in range(len(x))]
    
    if not y_err_vals is None:
        combined_vals += y_lower_vals
        combined_vals += y_upper_vals

    min_val = min(combined_vals)-offset
    max_val = max(combined_vals)+offset
    #-----------------------------------------------------

    plot_x_vs_y_with_error_bars(x,y,x_err_vals,y_err_vals,min_val,max_val,x_axis_label,y_axis_label,plot_title,plot_file_name,offset)


def getSigLevel2SubsetPredictionIntervals(test_ids,sig_level_2_prediction_intervals,subset_test_ids):
    
    subset_sig_level_2_prediction_intervals = {}
    
    for sig_level in sig_level_2_prediction_intervals.keys():
        test_prediction_intervals = sig_level_2_prediction_intervals[sig_level]
        
        subset_sig_level_2_prediction_intervals[sig_level] = getSubsetYFromSubsetIds(test_ids,test_prediction_intervals,subset_test_ids)#.tolist()
    
    
    return subset_sig_level_2_prediction_intervals

def computeAllRegMetrics(test_y,test_predictions,sig_level_of_interest,sig_level_2_prediction_intervals,smallest_no_cmpds_to_compute_stats=2,min_no_samples_per_bin_for_ENCE=20):
    
    no_compounds = get_no_instances(test_y)
    
    if size_of_inputs_for_stats_is_big_enough(test_y,test_predictions,limit=smallest_no_cmpds_to_compute_stats):
    
        rmse,MAD,R2,Pearson,Pearson_Pval_one_tail,Spearman,Spearman_Pval_one_tail = getAllRegPredMetrics(y_test=test_y, testPred=test_predictions)
        
        validity, efficiency,ECE_new,ENCE,errorRate_s,scc = getRelevantRegUncertainyMetrics(y_test=test_y, testPred=test_predictions,sig_level_2_prediction_intervals=sig_level_2_prediction_intervals,sig_level_of_interest=sig_level_of_interest,min_no_samples_per_bin_for_ENCE=min_no_samples_per_bin_for_ENCE)
    else:
        
        rmse,MAD,R2,Pearson,Pearson_Pval_one_tail,Spearman,Spearman_Pval_one_tail,validity, efficiency,ECE_new,ENCE,errorRate_s,scc = [None]*13
    
    return rmse,MAD,R2,Pearson,Pearson_Pval_one_tail,Spearman,Spearman_Pval_one_tail,validity, efficiency,ECE_new,ENCE,errorRate_s,no_compounds,scc

def map_all_regression_stats_onto_default_names(rmse, MAD, R2, Pearson, Pearson_Pval_one_tail, Spearman, Spearman_Pval_one_tail, validity, efficiency, ECE_new, ENCE, no_compounds,scc):
    
    dict_of_current_stats = {}
    
    dict_of_current_stats['rmse'] = rmse
    dict_of_current_stats['MAD'] = MAD
    dict_of_current_stats['R2'] = R2
    dict_of_current_stats['Pearson'] = Pearson
    dict_of_current_stats['Pearson_Pval_one_tail'] = Pearson_Pval_one_tail
    dict_of_current_stats['Spearman'] = Spearman
    dict_of_current_stats['Spearman_Pval_one_tail'] = Spearman_Pval_one_tail
    dict_of_current_stats['validity'] = validity
    dict_of_current_stats['efficiency'] = efficiency
    dict_of_current_stats['ECE_new'] = ECE_new
    dict_of_current_stats['ENCE'] = ENCE
    dict_of_current_stats['no. compounds'] = no_compounds
    dict_of_current_stats['SCC-Uncertainty'] = scc
    
    return dict_of_current_stats
