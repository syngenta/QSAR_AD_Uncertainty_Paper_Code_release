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
#Copyright (c) 2020 Syngenta
#Contact richard.marchese_robinson@syngenta.com
#######################
import sys,os,re
import numpy as np
import scipy as sp
from sklearn.isotonic import IsotonicRegression
from collections import defaultdict
import pickle

dir_of_this_file = os.path.dirname(os.path.abspath(__file__))
#top_dir = os.path.sep.join(dir_of_this_file.split(os.path.sep)[:-1])
from .VennABERS import ScoresToMultiProbs
from ..utils.basic_utils import geometricMean

##########################

def checkP0P1(p0,p1,calib_score_label_tuples=None,test_set_class_1_score=None):
    for p in [p0,p1]:
        assert p >= 0,str(p)
        assert p <= 1, str(p)
    try:
        assert p1 >= p0,"p0=%f,p1=%f. Corresponding inputs (if provided): calib_score_label_tuples=%s,test_set_class_1_score=%s" % (p0,p1,str(calib_score_label_tuples),str(test_set_class_1_score)) #Commented whilst running 'CVAP_extra_train' calculations, as this guarantee may not hold for this pseudo-CVAP approach?
    except AssertionError as err:
        fail_dict = {}
        fail_dict['p0'] = p0
        fail_dict['p1'] = p1
        fail_dict['calib_score_label_tuples'] = calib_score_label_tuples
        fail_dict['test_set_class_1_score'] = test_set_class_1_score
        f_o = open(os.path.sep.join([dir_of_this_file,"checkP0P1_fail_dict.pickle"]),'wb')
        try:
            pickle.dump(fail_dict,f_o)
        finally:
            f_o.close()
        print(err)
        sys.exit(1)
    
def checkP0P1calcInputs(calib_score_label_tuples,test_set_class_1_score):
    #-------------------------------
    assert type([]) == type(calib_score_label_tuples),type(calib_score_label_tuples)
    assert type(0.5) == type(test_set_class_1_score),type(test_set_class_1_score)
    #-------------------------------
    
    calib_scores = [t[0] for t in calib_score_label_tuples]
    calib_labels = [t[1] for t in calib_score_label_tuples]
    
    assert 0 == len([s for s in calib_scores if not type(0.5)==type(s)]),calib_scores
    assert 0 == len([l for l in calib_labels if not (0 == l or 1 == l)]),calib_labels
    assert all([isinstance(v,int) for v in calib_labels]),calib_labels
    
    return calib_scores,calib_labels

def convertTestScoreToP0AndP1(calib_score_label_tuples,test_set_class_1_score):
    
    checkP0P1calcInputs(calib_score_label_tuples,test_set_class_1_score)
    
    p0_array,p1_array = ScoresToMultiProbs(calib_score_label_tuples,[test_set_class_1_score])
    assert 1 == len(p0_array.tolist())
    assert 1 == len(p1_array.tolist())
    p0 = float(p0_array.tolist()[0])
    p1 = float(p1_array.tolist()[0])
    
    checkP0P1(p0,p1,calib_score_label_tuples,test_set_class_1_score)
    
    return p0,p1
##########################

def computeVennAbersClassOneLowerUpperProbs(calib_score_label_tuples,test_set_class_1_score):
    
    current_list_of_test_set_class_1_scores = [test_set_class_1_score]
    
    p0,p1 = convertTestScoreToP0AndP1(calib_score_label_tuples,test_set_class_1_score)
    
    checkP0P1(p0,p1,calib_score_label_tuples,test_set_class_1_score)
    
    current_lower_p_value = p0
    current_upper_p_value = p1
    
    
    return current_lower_p_value,current_upper_p_value

def computeSingleClassOneProb(current_lower_p_value,current_upper_p_value):
    #--------------------------------------------------
    p0 = current_lower_p_value
    p1 = current_upper_p_value
    #checkP0P1(p0,p1,calib_score_label_tuples,test_set_class_1_score) #Is it possible for the CVAP effective p0 to sometimes be higher than the effective p1, even if the individual values are compliant???
    #---------------------------------------------------    
    
    
    p_single =  p1/(1.0 - p0 + p1) #c.f. IVAP formula for binary classification single probability prediction in Manokhin (2017), "Multi-class probabilistic classifcation using inductive and cross Venn-Abers predictors"
    
    #------------------------------------------------
    assert p_single >= 0,str(p_single)
    assert p_single <= 1, str(p_single)
    #------------------------------------------------
    
    return p_single

def computeCVAPeffectiveP0P1values(list_of_p0_values,list_of_p1_values):
    #########################
    #c.f. Vovk et al. (2015) "Large-scale probabilistic predictors with and without guarantees of validity"
    #c.f. Manokhin (2017), "Multi-class probabilistic classifcation using inductive and cross Venn-Abers predictors"
    ########################
    #----------------------------
    assert type([])==type(list_of_p0_values),type(list_of_p0_values)
    assert type([])==type(list_of_p1_values),type(list_of_p1_values)
    assert 0 == len([v for v in list_of_p0_values if not type(0.5) == type(v)]),[type(v) for v in list_of_p0_values]
    assert 0 == len([v for v in list_of_p1_values if not type(0.5) == type(v)]),[type(v) for v in list_of_p1_values]
    assert len(list_of_p0_values) == len(list_of_p1_values)
    assert len(list_of_p0_values) > 1
    for index in range(0,len(list_of_p0_values)):
        checkP0P1(p0=list_of_p0_values[index],p1=list_of_p1_values[index])
    #---------------------------
    
    
    list_of_1_minus_p0_values = [(1-p0) for p0 in list_of_p0_values]
    
    effective_p1 = geometricMean(list_of_p1_values)
    effective_p0 = 1-geometricMean(list_of_1_minus_p0_values)
    
    # try:
        # checkP0P1(p0=effective_p0,p1=effective_p1) #Is it possible for the CVAP effective p0 to sometimes be higher than the effective p1, even if the individual values are compliant???
    # except AssertionError:
        # print('Warning! The effective p0 and p1 values are not compliant with p1 > p0, even though the individual values are!')
    
    current_lower_p_value = effective_p0
    current_upper_p_value = effective_p1
    
    return current_lower_p_value,current_upper_p_value
