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
#The function ebc_evaluation(...) was copied from the following repository: https://github.com/wangdingyan/HybridUQ
#[More specifically, this function was taken from this file: https://github.com/wangdingyan/HybridUQ/blob/c141a4bec0e716a12444f7e9ab0d7c975df93184/chemprop/plot/confidence.py#L115]
#Edits made, where necessary, by Zied Hosni and Richard Marchese Robinson
#Most notably, ebc_evaluation(...) was renamed compute_ENCE(...), the intermediate steps were broken into separate functions for unit-testing and some variable names were changed to try and make the code easier to follow.
#The original copyright and license information for that repository is reported below
##########################################################
# MIT License

# Copyright (c) 2020 Wengong Jin, Kyle Swanson, Kevin Yang, Regina Barzilay, Tommi Jaakkola

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
#===================================================
from ..utils.basic_utils import isInteger,getKeyFromValue,numpy_array_contains_zeros
from .reg_perf_pred_stats import SpearmanCoeff,compute_residuals
#===================================================

#==========================================
#sigLevels = np.linspace(0.01, 1.00, 100) copied from ZH compute_ECE(...) function and other adaptations made here to promote consistency in for-loops linking indices and significance levels to ECE calculation:
#from collections import OrderedDict #Probably do not need with Python 3.7 and above: https://realpython.com/python-ordereddict/

def getPercentageSigLevel(sig_level,round_dp=0):
    perc = round(100*sig_level,round_dp)
    
    return perc

def check_Global_index2SigLevel():
    
    old_i = -1
    for i in index2SigLevel.keys():
        
        assert i > old_i,"i = {} vs. old_i = {}".format(i,old_i)
        
        old_i = i
        
        sig_level = index2SigLevel[i]
        
        assert isInteger(getPercentageSigLevel(sig_level)),"sig_level = {}".format(sig_level)

def defineGlobalSigLevelsAndCorrespondingIndices():
    
    sigLevels = np.array([0.0,0.1,0.2,0.32,0.4,0.5,0.6,0.7,0.8,0.9])
    
    assert len(sigLevels.tolist()) == len(set(sigLevels.tolist()))
    
    index2SigLevel = dict(zip([i for i in range(len(sigLevels))],sigLevels.tolist()))
    
    return index2SigLevel,sigLevels

index2SigLevel,sigLevels = defineGlobalSigLevelsAndCorrespondingIndices()

check_Global_index2SigLevel()

#============================================

def estimated_variance_from_CP(intervals_width):
    #================================
    #This calculation only makes sense if these conformal regression prediction intervals were computed at a significance level of 32% and the following assumptions hold.
    #If we can assume a normal prediction error distribution, 68% would lie +-sigma of the mean.  Variance = sigma**2
    #If conformal prediction gives perfectly efficient prediction intervals then the prediction intervals at significance level 32% would correspond to a confidence interval on the true values of 68%.
    #===============================

    estimated_variance = [(i/2)**2 for i in intervals_width]
    
    return estimated_variance

def compute_ECE(error_rate_s,sig_levels=sigLevels,index2SigLevel=index2SigLevel):
    
    assert len(sig_levels) <= len(list(index2SigLevel.keys())),f'len(sig_levels)={len(sig_levels)}'
    #assert 100 == len(list(index2SigLevel.keys())),f'len(list(index2SigLevel.keys()))={len(list(index2SigLevel.keys()))}'
    if not len(sig_levels) == len(list(index2SigLevel.keys())):
        print(f'WARNING: You have chosen to compute ECE with only {len(sig_levels)} significance levels.')

    ECE_tot_list = []
    
    for expected_error_rate in sig_levels:
        i = getKeyFromValue(index2SigLevel,expected_error_rate)
        #==========================================
        if pd.isna(error_rate_s[i]):
            raise Exception(f'We cannot be missing an error rate for expected_error_rate={expected_error_rate}')
        #===========================================
        ECE_tot_list.append(abs(error_rate_s[i]-expected_error_rate)) #In Wang et al. (2021) [https://doi.org/10.1186/s13321-021-00551-x], this is defined as abs(accuracy(confidence level at i%) - i/100) = abs(accuracy(confidence level at i%) - expected_accuracy(confidence level at i%)) = abs((1.0 - accuracy(confidence level at i%)) - (1.0 - expected_accuracy(confidence level at i%))) = abs(error_rate(confidence level at i%) - expected_error_rate(confidence level at i%))
        
    
    ECE = sum(ECE_tot_list)/len(sig_levels)
    assert (not pd.isna(ECE)) and (ECE>0) , "ECE={} should be positive. \n error_rate_s = {} \n sig_levels = {}".format(ECE,error_rate_s,sig_levels)


    return ECE

def compute_validity_efficiency(error_rate, intervals_width):

    new_validity = 1-error_rate
    new_efficiency = np.mean(intervals_width)

    assert new_efficiency>=0 , "new_efficiency=%f should be positive" % (new_efficiency)
    assert new_validity>=0 , "new_validity=%f should be positive" % (new_validity) # I added that the validity can be 0
    assert new_validity<=1 , "new_validity=%f should be lower than 1" % (new_validity)


    return new_validity, new_efficiency



def compute_Spearman_rank_correlation_between_intervals_width_and_residuals(y_true,y_pred,intervals_width):
    residuals = compute_residuals(y_true,y_pred)

    spearman_coeff = SpearmanCoeff(e=residuals.tolist(),p=intervals_width.tolist())

    return spearman_coeff

def organize_variables_as_required_for_ENCE(y_true, y_pred, var_estimated):

    data = np.vstack((y_true, y_pred, var_estimated)).T
    data = data[data[:, 2].argsort()]  # Sort the data according to the var_estimated
    # data = data[::-1]
    # var_estimated_ordered = data[:,2].T.squeeze()

    return data

def bin_data_for_ENCE(data,min_no_samples_per_bin=20):

    if len(data) >= min_no_samples_per_bin:
        no_bins = len(data) // min_no_samples_per_bin
    else:
       no_bins = 1
    
    bins = np.array_split(data, no_bins, axis=0)
        
    return bins

def get_precursor_info_for_ENCE(y_true,
                   y_pred,
                   estimated_variance,min_no_samples_per_bin=20):
    
    var_estimated = estimated_variance
    
    data = organize_variables_as_required_for_ENCE(y_true, y_pred, var_estimated)

    bins = bin_data_for_ENCE(data,min_no_samples_per_bin=min_no_samples_per_bin)

    list_of_RMSE_values_for_each_y_pred_bin = []
    list_of_root_mean_squared_estimated_variance_in_prediction = []
    
    for i in range(len(bins)):
        rmse = np.sqrt(mse(bins[i][:, 0], bins[i][:, 1]))
        error = np.sqrt(np.nanmean(bins[i][:, 2]))
        list_of_RMSE_values_for_each_y_pred_bin.append(rmse)
        list_of_root_mean_squared_estimated_variance_in_prediction.append(error)
    
    array_of_RMSE_values_for_each_y_pred_bin = np.array(list_of_RMSE_values_for_each_y_pred_bin)
    
    array_of_root_mean_squared_estimated_variance_in_prediction = np.array(list_of_root_mean_squared_estimated_variance_in_prediction)

    return array_of_RMSE_values_for_each_y_pred_bin,array_of_root_mean_squared_estimated_variance_in_prediction

def compute_ENCE(y_true,
                   y_pred,
                   estimated_variance,min_no_samples_per_bin=20):
    
    y_true, y_pred, estimated_variance = np.array(y_true), np.array(y_pred), np.array(estimated_variance)
    if (y_pred.shape == y_true.shape == estimated_variance.shape) is False:
        raise Exception("The lists or arrays of experimental y (y_true), predicted y (y_pred) and the estimated variance in the prediction (estimated_variance) must all be the same length!")

    if any(np.isinf(estimated_variance)):
        ENCE = None
        print(f'WARNING: ENCE has been set to None with testPred={y_pred}, estimated_variance={estimated_variance}')
        return ENCE

    array_of_RMSE_values_for_each_y_pred_bin,array_of_root_mean_squared_estimated_variance_in_prediction = get_precursor_info_for_ENCE(y_true,
                   y_pred,
                   estimated_variance,min_no_samples_per_bin=min_no_samples_per_bin)

    RMSE_PER_BIN = array_of_RMSE_values_for_each_y_pred_bin
    EXPECTED_TYPICAL_PREDICTION_ERROR_PER_BIN = array_of_root_mean_squared_estimated_variance_in_prediction

    # compute expected normalized calibration error
    ENCE = np.nanmean((abs(RMSE_PER_BIN - EXPECTED_TYPICAL_PREDICTION_ERROR_PER_BIN)) / EXPECTED_TYPICAL_PREDICTION_ERROR_PER_BIN)
    
    #==================
    assert ENCE>0 , "ENCE=%f should be positive" % (ENCE)
    if np.isinf(ENCE): assert numpy_array_contains_zeros(EXPECTED_TYPICAL_PREDICTION_ERROR_PER_BIN),f'EXPECTED_TYPICAL_PREDICTION_ERROR_PER_BIN={EXPECTED_TYPICAL_PREDICTION_ERROR_PER_BIN}'
    #=================

    return ENCE
