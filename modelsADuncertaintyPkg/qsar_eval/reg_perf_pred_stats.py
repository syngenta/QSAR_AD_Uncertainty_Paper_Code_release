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
#Copyright (c) 2020-2023 Syngenta
#Contact richard.marchese_robinson@syngenta.com
#Adapted from pred_perf_metrics_including_MAD.py
#pred_perf_metrics_including_MAD.py was retrieved from https://dx.doi.org/10.5281/zenodo.3477986 on 24/04/20
#######################
####################################
#https://opensource.org/licenses/BSD-3-Clause (last accessed 26/11/17)
#Copyright (c) 2016-2019 University of Leeds
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
####################################
import scipy
import scipy.stats
import numpy as np
from sklearn import metrics
import math
from pytest import approx
from ..utils.basic_utils import finite_value_could_not_be_computed

def checkExperiPredictionsLists(e,p):
    assert type([]) == type(e)
    assert type([]) == type(p)
    assert 0 == len([v for v in e if not type(0.5)==type(v)])#, "len(probs_of_all_classes_in_order_for_all_instances)=%d, len(experi_class_labels)=%d" % (v,v)
    #del v #Syngenta
    assert 0 == len([v for v in p if not type(0.5)==type(v)])
    #del v #Syngenta
    assert not 0 == len(e),"Regression metrics will fail if you supply an empty list of experimental values!"
    assert len(e) == len(p),"Regression metrics will fail if you supply a list of predicted values which is not equal in length to the list of experimental values!"

def parseExperiPredLists(e,p):
    checkExperiPredictionsLists(e,p)
    return np.asarray(e),np.asarray(p)

def rmse(e,p,check_consistency=True):
    if any(np.isinf(p)):
        return np.inf
    
    e,p = parseExperiPredLists(e,p)
    
    rmse = scipy.sqrt(sum(((e-p)**2))/len(e))
    
    if check_consistency:
        y_test = e
        testPred = p
        alt_rmse = math.sqrt(metrics.mean_squared_error(y_test, testPred))
        
        assert rmse == approx(alt_rmse),"rmse = {}, alt_rmse={}".format(rmse,alt_rmse)
    
    return rmse

def MAD(e,p,check_consistency=True):
    if any(np.isinf(p)):
        return np.inf
    
    e,p = parseExperiPredLists(e,p)


    mad = sum(abs(e-p))/len(e)
    
    if check_consistency:
        alt_mad = metrics.mean_absolute_error(y_true=e, y_pred=p)
        
        assert mad == approx(alt_mad),"mad = {}, alt_mad={}".format(mad,alt_mad)
    
    return mad


def coeffOfDetermination(e,p,check_consistency=True):
    if any(np.isinf(p)):
        return -np.inf
    
    e,p = parseExperiPredLists(e,p)
    
    r2 = 1 - (sum((e-p)**2)/sum((e-scipy.mean(e))**2))
    
    if check_consistency:
        y_test = e
        testPred = p
        alt_r2 = metrics.r2_score(y_test, testPred)
        
        if not finite_value_could_not_be_computed(r2): 
            assert r2 == approx(alt_r2),"r2 = {}, alt_r2={}".format(r2,alt_r2)
        else:
            r2 = None
            if not finite_value_could_not_be_computed(alt_r2):
                print(f'WARNING: Setting r2=None, when alt_r2={alt_r2},experimental values={e},predicted values={p}')
    
    
    return r2

def PearsonCoeff(e,p):
    if any(np.isinf(p)):
        return None
    
    e,p = parseExperiPredLists(e,p)
    
    val = scipy.stats.pearsonr(e,p)[0]
    
    return val

def convertTwoTailPvalue(two_tailed_p_value,coeff,alternative):
    
    if "greater" == alternative:
        if coeff > 0:
            val = two_tailed_p_value/2.0
        else:
            val = 1 - (two_tailed_p_value/2.0)
    elif "less" == alternative:
        if coeff <= 0:
            val = two_tailed_p_value/2.0
        else:
            val = 1 - (two_tailed_p_value/2.0)
    elif "different" == alternative:
        val = two_tailed_p_value
    else:
        raise Exception("Unrecognised alternative hypothesis:%s" % alternative)
    
    return val

def PearsonCoeffPval(e,p,alternative="greater"):
    if any(np.isinf(p)):
        return None

    e,p = parseExperiPredLists(e,p)
    
    coeff,two_tailed_p_value = scipy.stats.pearsonr(e,p)
    
    val = convertTwoTailPvalue(two_tailed_p_value=two_tailed_p_value,coeff=coeff,alternative=alternative)
    
    if len(e) <= 500:
        print('Warning: this Pearson correlation coefficient p-value might be unreliable! \n See the documentation: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html#scipy.stats.pearsonr \n') #Syngenta
    
    return val

def SpearmanCoeff(e,p):
    e,p = parseExperiPredLists(e,p)
    
    val = scipy.stats.spearmanr(e,p)[0]
    
    return val

def SpearmanCoeffPval(e,p,alternative="greater"):
    e,p = parseExperiPredLists(e,p)
    
    coeff,two_tailed_p_value = scipy.stats.spearmanr(e,p)
    
    val = convertTwoTailPvalue(two_tailed_p_value=two_tailed_p_value,coeff=coeff,alternative=alternative)
    
    if len(e) <= 500:
        print('Warning: this Spearman correlation coefficient p-value might be unreliable! \n See the documentation: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html#scipy.stats.spearmanr \n') #Syngenta
    
    return val

def compute_residuals(y_true,y_pred):
    return np.abs(np.array(y_true)-np.array(y_pred))
