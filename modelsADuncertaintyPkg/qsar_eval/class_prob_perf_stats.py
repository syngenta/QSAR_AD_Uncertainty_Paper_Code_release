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
#Copyright (c)  2020-2022 Syngenta
#Contact richard.marchese_robinson@syngenta.com
#######################
#===========================================
import sys,re,glob,os,getopt,functools
import pandas as pd
import numpy as np
from collections import defaultdict
from .class_pred_perf_stats import getWeight
from sklearn.metrics import brier_score_loss #c.f. Mervin et al. (2020) [https://pubs.acs.org/doi/full/10.1021/acs.jcim.0c00476]
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
#from pred_perf_metrics_including_MAD import SpearmanCoeff
#==========================================

def computeVariantBrierLossForKclasses(pred_class_prob_tuples,experi_class_labels):
    #================================
    #For two classes, this variation is equivalent to the variation used here, for individual classes, to compute the stratified Brier score: Collell et al. (2018) [http://dx.doi.org/10.1016/j.neucom.2017.08.035])
    #For binary classification, p_inactive = (1-p_active))
    #
    #[1] In Collell et al. (2018), they would compute the following for the active and inactive classes:
    #active: (1- p_active)^2
    #inactive: (1 - p_inactive)^2 = (p_active)^2
    #
    #[2] What we compute:
    
    #[2] p_active vs. 1 or 0 (predicted active - correct or incorrect prediction) OR p_inactive vs. 1 or 0 (predicted inactive  - correct or incorrect prediction): Brier loss:
    #
    #active AND predicted active: (1 - p_active)^2
    #active AND predicted inactive: (0 - (1-p_active))^2 = (1 - p_active)^2
    #
    #inactive AND predicted inactive: (1 - (1-p_active))^2 = (p_active)^2
    #inactive AND predicted active: (0 - p_active)^2 = (p_active)^2
    #
    #================================
    #*********************************
    assert type([]) == type(pred_class_prob_tuples),str(type(pred_class_prob_tuples))
    assert type([])==type(experi_class_labels),str(type(experi_class_labels))
    assert len(pred_class_prob_tuples)==len(experi_class_labels),"len(pred_class_prob_tuples)=%d,len(experi_class_labels)=%d" % (len(pred_class_prob_tuples),len(experi_class_labels))
    #*********************************
    numeric_classes = []
    prob_vals = []
    for index in range(0,len(pred_class_prob_tuples)):
        pred_class = pred_class_prob_tuples[index][0]
        prob = pred_class_prob_tuples[index][1]
        experi_class = experi_class_labels[index]

        if pred_class == experi_class: #This should be adapted for ordinal classes!
            numeric_classes.append(1)
        else:
            numeric_classes.append(0)
        
        prob_vals.append(prob)
    
    return np.sum((np.array(prob_vals)-np.array(numeric_classes))**2)/len(numeric_classes)

def getProbOfClass1(numeric_pred_class,prob_of_pred_class):
    if 1 == numeric_pred_class:
        return prob_of_pred_class
    elif 0 == numeric_pred_class:
        return (1-prob_of_pred_class)
    else:
        raise Exception('Numeric predicted class=%d' % numeric_pred_class)
    

def check_computeVariantBrierLossForKclasses():
    examples_dict = defaultdict(dict)
    
    ###########################
    #[1] Very basic example taken from here: https://www.statisticshowto.com/brier-score/
    
    examples_dict[1]['pred_class_prob_tuples'] = [('rain',0.9)]
    examples_dict[1]['experi_class_labels'] = ['rain']
    examples_dict[1]['expected'] = 0.010
    
    ###########################
    
    ############################
    #[2] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
    
    examples_dict[2]['pred_class_prob_tuples'] = [('spam',0.9),("ham",0.9),("ham",0.8),("spam",0.7)] #class_1 probabilities = [0.1, 0.9, 0.8, 0.3]. class_1 = "ham" . class_1 probability > 0.5 assumed to correspond to a prediction of class_1.
    examples_dict[2]['experi_class_labels'] = ["spam", "ham", "ham", "spam"]
    examples_dict[2]['expected'] = 0.03750 #SciKit-Learn documentation truncates the results!
    #>>> y_true = np.array([0, 1, 1, 0])
    #>>> y_prob = np.array([0.1, 0.9, 0.8, 0.3])
    #>>> brier_score_loss(y_true, y_prob)
    #0.03749999999999999
    
    ############################
    
    ############################
    #[3] Example [2] ADAPTED to add one incorrect prediction at the end
    
    examples_dict[3]['pred_class_prob_tuples'] = [('spam',0.9),("ham",0.9),("ham",0.8),("spam",0.7),("ham",0.55)] #class_1 probabilities = [0.1, 0.9, 0.8, 0.3]. class_1 = "ham" . class_1 probability > 0.5 assumed to correspond to a prediction of class_1.
    examples_dict[3]['experi_class_labels'] = ["spam", "ham", "ham", "spam","spam"]
    examples_dict[3]['expected'] = 0.0905 #((0.0375*4)+(0.55-0)**2)/5
    
    ############################
    
    for eg in examples_dict.keys():
        print('check_computeVariantBrierLossForKclasses(): Checking example %d' % eg)
        
        pred_class_prob_tuples = examples_dict[eg]['pred_class_prob_tuples']
        experi_class_labels = examples_dict[eg]['experi_class_labels']
        
        res = computeVariantBrierLossForKclasses(pred_class_prob_tuples,experi_class_labels)
        
        assert round(res,4) == examples_dict[eg]['expected'],"example %d: res=%f,expected=%f" % (eg,res,examples_dict[eg]['expected'])
        
        #============================================
        if 2 >= len(set(examples_dict[eg]['experi_class_labels'])):
            print('check_computeVariantBrierLossForKclasses(): example %d: Also checking consistency for binary classification with sklearn.metrics.brier_score_loss!' % eg) #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
            
            classLabel2Binary = dict(zip(list(set(examples_dict[eg]['experi_class_labels'])),[1,0]))
            
            binary_class_labels = [classLabel2Binary[cl] for cl in examples_dict[eg]['experi_class_labels']]
            
            print('binary_class_labels')
            
            class_1_probs = [getProbOfClass1(numeric_pred_class=classLabel2Binary[t[0]],prob_of_pred_class=t[1]) for t in examples_dict[eg]['pred_class_prob_tuples']]
            
            alt_res = brier_score_loss(np.array(binary_class_labels),np.array(class_1_probs))
            
            assert round(res,4)==round(alt_res,4),"res=%f,alt_res=%f" % (res,alt_res)
            
        #============================================
        
        print('check_computeVariantBrierLossForKclasses(): CHECKED example %d' % eg)

#check_computeVariantBrierLossForKclasses()

def probsSumOK(ps,tolerance=0.02):
    ########################
    #We expect the sum of probability estimates to be one.
    #However, numerical rounding issues might lead to minor deviations.
    ########################
    
    if abs(sum(ps)-1) <= tolerance:
        return True
    else:
        return False

def checkOriginalScoreCalcInputs(probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,experi_class_labels,debug=False):
    #**********************
    assert type([]) == type(probs_of_all_classes_in_order_for_all_instances),str(type(probs_of_all_classes_in_order_for_all_instances))
    assert 0 == len([e for e in probs_of_all_classes_in_order_for_all_instances if not type([])==type(e)]),"probs_of_all_classes_in_order_for_all_instances=%s" % str(probs_of_all_classes_in_order_for_all_instances)
    assert 0 == len([ps for ps in probs_of_all_classes_in_order_for_all_instances if not probsSumOK(ps)]),"probs_of_all_classes_in_order_for_all_instances=%s" % str(probs_of_all_classes_in_order_for_all_instances)
    assert type([]) == type(class_labels_in_order),str(type(class_labels_in_order))
    assert type([])==type(experi_class_labels),str(type(experi_class_labels))
    assert len(probs_of_all_classes_in_order_for_all_instances) == len(experi_class_labels),"len(probs_of_all_classes_in_order_for_all_instances)=%d, len(experi_class_labels)=%d" % (len(probs_of_all_classes_in_order_for_all_instances),len(experi_class_labels))
    experi_labels_not_expected = [c for c in experi_class_labels if not c in class_labels_in_order] #18/03/21: cannot see full error message in PP!
    assert 0 == len(experi_labels_not_expected),"experi_labels_not_expected=%s,class_labels_in_order=%s,experi_class_labels=%s" % (str(experi_labels_not_expected),str(class_labels_in_order),str(experi_class_labels)) #18/03/21
    if debug:
        print("class_labels_in_order", class_labels_in_order)
        print("set(class_labels_in_order)", set(class_labels_in_order))
        print("len(class_labels_in_order)", len(class_labels_in_order))
        print("len(set(class_labels_in_order))", len(set(class_labels_in_order)))
    assert len(class_labels_in_order) == len(set(class_labels_in_order)),str(class_labels_in_order)
    #**********************

def oneHotKeyEncodeTrueClassValues(class_labels_in_order,experi_class_labels):
    one_hot_key_encoded_values_of_all_experi_classes_in_order = []
    for ec in experi_class_labels:
        one_hot_key_list = []
        for c in class_labels_in_order:
            if ec == c: #This should be adapted for ordinal classes!
                one_hot_key_list.append(1)
            else:
                one_hot_key_list.append(0)
        one_hot_key_encoded_values_of_all_experi_classes_in_order.append(one_hot_key_list)
    return one_hot_key_encoded_values_of_all_experi_classes_in_order

def computeOriginalBrierLossForKclasses(probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,experi_class_labels):
    #==================================
    #c.f. Manokhin 2017 (“Multi-class probabilistic classification using inductive and cross Venn-Abers predictors”)
    #https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    #https://www.wikiwand.com/en/Brier_score#/Original_definition_by_Brier
    #Brier (1950) "Verification of Forecasts Expressed in Terms of Probability", Monthly Weather Review, 78, 1-3
    #==================================
    
    checkOriginalScoreCalcInputs(probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,experi_class_labels)
    
    N = len(probs_of_all_classes_in_order_for_all_instances)
    #R = len(class_labels_in_order)
    
    one_hot_key_encoded_values_of_all_experi_classes_in_order = oneHotKeyEncodeTrueClassValues(class_labels_in_order,experi_class_labels)
    
    return np.sum((np.array(probs_of_all_classes_in_order_for_all_instances)-np.array(one_hot_key_encoded_values_of_all_experi_classes_in_order))**2)/N #Sums over both both axes, i.e. instances (total = N) and classes (total = R)

def check_computeOriginalBrierLossForKclasses():
    examples_dict = defaultdict(dict)
    
    ###################################
    #Example (1) is taken from the original publication (Table 1):
    #Brier (1950) "Verification of Forecasts Expressed in Terms of Probability", Monthly Weather Review, 78, 1-3
    ####################################
    
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'] = []
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'].append([0.7,0.3])
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'].append([0.9,0.1])
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'].append([0.8,0.2])
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'].append([0.4,0.6])
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'].append([0.2,0.8])
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'].append([0,1])
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'].append([0,1])
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'].append([0,1])
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'].append([0,1])
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'].append([0.1,0.9])
    
    examples_dict[1]['class_labels_in_order'] = ['rain','no-rain']
    
    examples_dict[1]['experi_class_labels'] = ['no-rain','rain','rain','rain'] + 6*['no-rain']
    
    examples_dict[1]['expected'] = 0.19
    
    ######################################
    
    for eg in examples_dict.keys():
        print('check_computeOriginalBrierLossForKclasses: checking example %d' % eg)
        
        probs_of_all_classes_in_order_for_all_instances = examples_dict[eg]['probs_of_all_classes_in_order_for_all_instances']
        class_labels_in_order = examples_dict[eg]['class_labels_in_order']
        experi_class_labels = examples_dict[eg]['experi_class_labels']
        
        res = computeOriginalBrierLossForKclasses(probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,experi_class_labels)
        
        assert round(res,2) == examples_dict[eg]['expected'],"example %d: res=%f,expected=%f" % (eg,res,examples_dict[eg]['expected'])
        
        print('check_computeOriginalBrierLossForKclasses: CHECKED example %d' % eg)
    
#check_computeOriginalBrierLossForKclasses()

def computeWijValuesForAllInstances(class_labels_in_order,experi_class_labels):
    
    w_ij_values_for_all_instances = []
    for ec in experi_class_labels:
        class_j = ec
        
        w_ij_list = []
        
        for c in class_labels_in_order:
            class_i = c
            
            w_ij = getWeight(class_i,class_j,class_labels_in_order)
            
            w_ij_list.append(w_ij)
        w_ij_values_for_all_instances.append(w_ij_list)
    return w_ij_values_for_all_instances

def computeLogLoss(probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,experi_class_labels,ordinal_variant=False,focus_on_predicted_class=False,eps=10**-15):
    #====================================================================
    #For a given instance with a given class (j), the log loss contribution for an individual class (i), for which a probability (p_i) was estimated, is given by l_ij:
    #ll_ij = -w_ij*log(p_i) , where w_ij is a weight
    #c.f. Manokhin 2017 (“Multi-class probabilistic classification using inductive and cross Venn-Abers predictors”), the standard log loss sets w_ij as follows:
    #w_ij = 1 (if the instance belongs to class i, i.e. if i = j) or w_ij = 0 (if the instance belongs to any other class, i.e. if i != j)
    #The log loss contributions are summed for all classes, to give the log loss for an individual instance.
    #The arithmetic mean of the log loss across all instances is reported as the log loss for the set of instances.
    #
    #Non-standard options (not mutually exclusive):
    #[1] ordinal_variant=True (default=False):
    #Here, a variation on the standard calculation is introduced, to account for ordinal classes, by adjusting the values of w_ij c.f. Ben David 2008 ("Comparison of classification accuracy using Cohen’s Weighted Kappa")
    #[2] focus_on_predicted_class=True (default=False):
    #Rather than being concerned with the entire probability distribution, we should arguably be most concerned with the probability estimated for the predicted class, i.e. the class with the highest estimated probability.
    #Hence, log loss contributions arising from probabilities other than the predicted class (highest p_i) would be ignored. 
    #If only the probability for the predicted class is available, the other probabilities could be assigned dummy values of zero. 
    #*However*, a problem with this idea is that incorrect predictions would make no contributions to the log loss (ordinal_variant=False), or limited contributions (ordinal_variant=True), hence this would be misleading! 
    #--------------------------------------------------------------------
    #The use of eps to avoid numerical error when p_i = 0 is based upon the scikit-learn implementation [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss]
    #====================================================================
    
    if focus_on_predicted_class: raise Exception('This is actually not a good idea for the log loss score!')
    
    checkOriginalScoreCalcInputs(probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,experi_class_labels)
    
    N = len(probs_of_all_classes_in_order_for_all_instances)
    
    if not ordinal_variant:
        w_ij_values_for_all_instances = oneHotKeyEncodeTrueClassValues(class_labels_in_order,experi_class_labels)
    else:
        w_ij_values_for_all_instances = computeWijValuesForAllInstances(class_labels_in_order,experi_class_labels)
    
    log_loss_values_for_all_instances = []
    for instance_index in range(0,len(probs_of_all_classes_in_order_for_all_instances)):
        
        log_loss_for_instance_j = 0
        
        for class_index in range(0,len(class_labels_in_order)):
            
            w_ij = w_ij_values_for_all_instances[instance_index][class_index]
            
            p_i = probs_of_all_classes_in_order_for_all_instances[instance_index][class_index]
            
            l_ij = -w_ij*np.log(p_i+eps)
            
            log_loss_for_instance_j += l_ij
        
        log_loss_values_for_all_instances.append(log_loss_for_instance_j)
    
    return sum(log_loss_values_for_all_instances)/N

def check_computeLogLoss():
    
    examples_dict = defaultdict(dict)
    ############################
    
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'] = [[0.1,0.3,0.5,0.1]]
    examples_dict[1]['class_labels_in_order'] = ['1','2','3','4']
    examples_dict[1]['experi_class_labels'] = ['1']
    examples_dict[1]['ordinal_variant'] = False
    
    examples_dict[1]['expected'] = -np.log(0.1+10**-15) #2.3
    
    sklearn_estimate = log_loss(y_true=['1'], y_pred=[[0.1,0.3,0.5,0.1]], eps=1e-15, normalize=True, sample_weight=None, labels=['1','2','3','4'])
    assert round(sklearn_estimate,2) == round(examples_dict[1]['expected'],2),str(sklearn_estimate)
    
    ###########################
    
    ############################
    
    examples_dict[2]['probs_of_all_classes_in_order_for_all_instances'] = [[0.1,0.3,0.5,0.1]]
    examples_dict[2]['class_labels_in_order'] = ['1','2','3','4']
    examples_dict[2]['experi_class_labels'] = ['3']
    examples_dict[2]['ordinal_variant'] = False
    
    examples_dict[2]['expected'] = -np.log(0.5+10**-15)
    
    ###########################
    
    ############################
    
    examples_dict[3]['probs_of_all_classes_in_order_for_all_instances'] = [[0.1,0.3,0.5,0.1]]
    examples_dict[3]['class_labels_in_order'] = ['1','2','3','4']
    examples_dict[3]['experi_class_labels'] = ['3']
    examples_dict[3]['ordinal_variant'] = True
    
    examples_dict[3]['expected'] = -np.log(0.5+10**-15) + (-np.log(0.1+10**-15)*(1-(abs(3-1)/3))) + (-np.log(0.3+10**-15)*(1-(abs(3-2)/3))) + (-np.log(0.1+10**-15)*(1-(abs(3-4)/3)))
    
    ###########################
    
    examples_dict[4]['probs_of_all_classes_in_order_for_all_instances'] = [[0.6,0.4]]
    examples_dict[4]['class_labels_in_order'] = ['1','2']
    examples_dict[4]['experi_class_labels'] = ['2']
    examples_dict[4]['ordinal_variant'] = True
    
    examples_dict[4]['expected'] = -np.log(0.4+10**-15)
    
    ###########################
    
    for eg in examples_dict.keys():
        print('check_computeLogLoss: checking example %d' % eg)
        
        probs_of_all_classes_in_order_for_all_instances = examples_dict[eg]['probs_of_all_classes_in_order_for_all_instances']
        class_labels_in_order = examples_dict[eg]['class_labels_in_order']
        experi_class_labels = examples_dict[eg]['experi_class_labels']
        ordinal_variant = examples_dict[eg]['ordinal_variant']
        
        res = computeLogLoss(probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,experi_class_labels,ordinal_variant)
        
        assert round(res,2) == round(examples_dict[eg]['expected'],2),"example %d: res=%f,expected=%f" % (eg,res,examples_dict[eg]['expected'])
        
        #---------------------------------------
        if 2 == len(class_labels_in_order):
            if ordinal_variant:
                alt_res = computeLogLoss(probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,experi_class_labels,False)
                
                assert res == alt_res,"example %d: res=%f,alt_res=%f" % (eg,res,alt_res)
        #---------------------------------------
        
        print('check_computeLogLoss: CHECKED example %d' % eg)

#check_computeLogLoss()

def compute_Stratified_ProbabilisticLossForKclasses(predictions,class_labels_in_order,experi_class_labels,type_of_loss,debug=False):
    #################################################
    #c.f. Collell et al. (2018) [http://dx.doi.org/10.1016/j.neucom.2017.08.035]
    #Note, the 'standard' formulation of the Brier score is used in Collell et al. (2018), but this idea can be generalized. 
    #################################################
    #*********************************
    assert type([]) == type(predictions),str(type(predictions))
    assert type([])==type(experi_class_labels),str(type(experi_class_labels))
    assert len(predictions)==len(experi_class_labels),"len(predictions)=%d,len(experi_class_labels)=%d" % (len(predictions),len(experi_class_labels))
    experi_labels_not_expected = [c for c in experi_class_labels if not c in class_labels_in_order] #18/03/21: cannot see full error message in PP!
    assert 0 == len(experi_labels_not_expected),"experi_labels_not_expected=%s,class_labels_in_order=%s,experi_class_labels=%s" % (str(experi_labels_not_expected),str(class_labels_in_order),str(experi_class_labels)) #18/03/21: cannot see full error message in PP!
    #*********************************
    
    all_class_specific_scores = []
    
    for class_label in class_labels_in_order:
        #-----------------------------
        subset_indices = [i for i in range(0,len(experi_class_labels)) if class_label == experi_class_labels[i]]
        
        subset_experi_class_labels = [experi_class_labels[i] for i in subset_indices]
        subset_predictions = [predictions[i] for i in subset_indices]
        if debug:
            print("subset_indices",subset_indices)
            print("subset_predictions", subset_predictions)
        #-----------------------------
        
        if 'original_Brier' == type_of_loss:
            
            subset_probs_of_all_classes_in_order_for_all_instances = subset_predictions
            
            loss = computeOriginalBrierLossForKclasses(subset_probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,subset_experi_class_labels)
        
        elif 'variant_Brier' == type_of_loss:
            subset_pred_class_prob_tuples = subset_predictions
            
            loss = computeVariantBrierLossForKclasses(subset_pred_class_prob_tuples,subset_experi_class_labels)
        
        elif 'computeLogLoss' == type_of_loss:
            
            subset_probs_of_all_classes_in_order_for_all_instances = subset_predictions
            
            ordinal_variant = False
            
            loss = computeLogLoss(subset_probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,subset_experi_class_labels,ordinal_variant)
        
        elif 'computeLogLoss_ordinal' == type_of_loss:
            
            subset_probs_of_all_classes_in_order_for_all_instances = subset_predictions
            
            ordinal_variant = True
            
            loss = computeLogLoss(subset_probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,subset_experi_class_labels,ordinal_variant)
        
        else:
            raise Exception('Unrecognised loss function: %s' % type_of_loss)
        
        all_class_specific_scores.append(loss)
    
    return sum(all_class_specific_scores)/len(all_class_specific_scores)

def check_compute_Stratified_ProbabilisticLossForKclasses():
    examples_dict = defaultdict(dict)
    
    ##########################################
    examples_dict[1]['predictions'] = []
    examples_dict[1]['predictions'].append([0.7,0.3])
    examples_dict[1]['predictions'].append([0.9,0.1])
    examples_dict[1]['predictions'].append([0.8,0.2])
    examples_dict[1]['predictions'].append([0.4,0.6])
    examples_dict[1]['predictions'].append([0.2,0.8])
    examples_dict[1]['predictions'].append([0,1])
    examples_dict[1]['predictions'].append([0,1])
    examples_dict[1]['predictions'].append([0,1])
    examples_dict[1]['predictions'].append([0,1])
    examples_dict[1]['predictions'].append([0.1,0.9])
    
    examples_dict[1]['class_labels_in_order'] = ['rain','no-rain']
    
    examples_dict[1]['experi_class_labels'] = ['no-rain','rain','rain','rain'] + 6*['no-rain']
    
    examples_dict[1]['type_of_loss'] = 'original_Brier'
    
    examples_dict[1]['expected'] = 0.21
    #################################################
    
    examples_dict[2]['predictions'] = [('rain',0.9),('rain',0.9),('no-rain',0.6)]
    examples_dict[2]['experi_class_labels'] = ['rain','no-rain','no-rain']
    examples_dict[2]['type_of_loss'] = 'variant_Brier'
    
    examples_dict[2]['expected'] = 0.25
    
    examples_dict[2]['class_labels_in_order'] = ['rain','no-rain']
    
    ###########################
    
    examples_dict[3]['predictions'] = [('A',0.9),('B',0.9),('B',0.6),('C',0.4),('C',0.6)]
    examples_dict[3]['experi_class_labels'] = ['A','B','C','C','C']
    examples_dict[3]['type_of_loss'] = 'variant_Brier'
    
    examples_dict[3]['expected'] = 0.10
    
    examples_dict[3]['class_labels_in_order'] = ['A','B','C']
    
    ###########################
    
    ##########################################
    examples_dict[4]['predictions'] = []
    examples_dict[4]['predictions'].append([0.1,0.3,0.5,0.1])
    examples_dict[4]['predictions'].append([0.1,0.3,0.5,0.1])
    examples_dict[4]['predictions'].append([0.1,0.3,0.5,0.1])
    examples_dict[4]['predictions'].append([0.1,0.3,0.5,0.1])
    examples_dict[4]['predictions'].append([0.1,0.3,0.5,0.1])
    
    examples_dict[4]['class_labels_in_order'] = ['1','2','3','4']
    
    examples_dict[4]['experi_class_labels'] = ['1','1','3','2','4']
    
    examples_dict[4]['type_of_loss'] = 'computeLogLoss'
    
    examples_dict[4]['expected'] = sum([-np.log(0.1+10**-15),-np.log(0.5+10**-15),-np.log(0.3+10**-15),-np.log(0.1+10**-15)])/4
    #################################################
    
    ###########################
    
    for eg in examples_dict.keys():
        print('check_compute_Stratified_ProbabilisticLossForKclasses: checking example %d' % eg)
        
        predictions = examples_dict[eg]['predictions']
        class_labels_in_order = examples_dict[eg]['class_labels_in_order']
        experi_class_labels = examples_dict[eg]['experi_class_labels']
        type_of_loss = examples_dict[eg]['type_of_loss']
        print('type_of_loss=%s' % type_of_loss)
        
        res = compute_Stratified_ProbabilisticLossForKclasses(predictions,class_labels_in_order,experi_class_labels,type_of_loss)
        
        assert round(res,2) == round(examples_dict[eg]['expected'],2),"example %d: res=%f,expected=%f" % (eg,res,examples_dict[eg]['expected'])
        
        print('check_compute_Stratified_ProbabilisticLossForKclasses: CHECKED example %d' % eg)

#check_compute_Stratified_ProbabilisticLossForKclasses()


def renormalize(probs_which_should_sum_to_one):
    assert probsSumOK(probs_which_should_sum_to_one) 
    #However, it appears that roc_auc_score(...) is particularly senstiive to even small deviations of the sum from one!
    sum_val = sum(probs_which_should_sum_to_one)
    return [(p/sum_val) for p in probs_which_should_sum_to_one]

def bitEncodeBinaryLabel(label,first_class):
    if label == first_class:
        return 1
    else:
        return 0

def computePairwiseAverageAuc(probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,experi_class_labels):
    ######################################
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    #For the multi-class case, we choose multi_class='ovo' (average of one vs. one for all possible pairwise combinations), which is insensitive to class imbalance when average='macro' (unweighted mean of AUC values)
    #I *think* this corresponds to calculating the value defined in equation (7) of Hand et al. (2001):
    #Hand et al. (2001). "A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems". Machine Learning, 45(2), 171-186.
    ######################################

    checkOriginalScoreCalcInputs(probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,experi_class_labels)
    
    if len(class_labels_in_order) > 2:
        
        renorm_probs_of_all_classes_in_order_for_all_instances = [renormalize(ps) for ps in probs_of_all_classes_in_order_for_all_instances]

        return roc_auc_score(y_true=np.array(experi_class_labels), y_score=np.array(renorm_probs_of_all_classes_in_order_for_all_instances), average='macro',multi_class='ovo', labels=np.array(class_labels_in_order))
    else:
        
        assert len(class_labels_in_order) == 2
        
        probs_of_first_class = [p[0] for p in probs_of_all_classes_in_order_for_all_instances]
        
        first_class = class_labels_in_order[0]
        
        bit_encoded_experi_class_labels = [bitEncodeBinaryLabel(label,first_class) for label in experi_class_labels]
        
        return roc_auc_score(y_true=np.array(bit_encoded_experi_class_labels), y_score=np.array(probs_of_first_class))

def check_computePairwiseAverageAuc():
    
    examples_dict = defaultdict(dict)
    ############################
    
    examples_dict[1]['probs_of_all_classes_in_order_for_all_instances'] = [[0.4,0.3,0.2,0.1],[0.3,0.4,0.2,0.1],[0.3,0.2,0.4,0.1],[0.1,0.2,0.3,0.4]]
    examples_dict[1]['class_labels_in_order'] = ['1','2','3','4']
    examples_dict[1]['experi_class_labels'] = ['1','2','3','4']
    
    examples_dict[1]['expected'] = 1
    
    ###########################
    
    ############################
    
    examples_dict[2]['probs_of_all_classes_in_order_for_all_instances'] = [[0.8,0.2],[0.2,0.8],[0.8,0.2],[0.2,0.8],[0.2,0.8]]
    examples_dict[2]['class_labels_in_order'] = ['A','B']
    examples_dict[2]['experi_class_labels'] = ['A','B','A','B','B']
    
    examples_dict[2]['expected'] = 1
    
    ###########################
    
    ############################
    #>>> roc_auc_score(np.array(ytrue),np.array(yprob))
    #0.6666666666666667
    #>>> ytrue
    #[0, 1, 0, 1, 1, 1, 0]
    #>>> yprob
    #[1, 0.8, 0.2, 0.58, 0.87, 0.99, 0.2]
    
    examples_dict[3]['probs_of_all_classes_in_order_for_all_instances'] = [[1.0,0.0],[0.80,0.20],[0.20,0.80],[0.58,0.42],[0.87,0.13],[0.99,0.01],[0.20,0.80]]
    examples_dict[3]['class_labels_in_order'] = ['A','B']
    examples_dict[3]['experi_class_labels'] = ['B','A','B','A','A','A','B']
    
    examples_dict[3]['expected'] = 0.67
    ##############################
    
    for eg in examples_dict.keys():
        print('check_computePairwiseAverageAuc: checking example %d' % eg)
        
        probs_of_all_classes_in_order_for_all_instances = examples_dict[eg]['probs_of_all_classes_in_order_for_all_instances']
        class_labels_in_order = examples_dict[eg]['class_labels_in_order']
        experi_class_labels = examples_dict[eg]['experi_class_labels']
        
        
        res = computePairwiseAverageAuc(probs_of_all_classes_in_order_for_all_instances,class_labels_in_order,experi_class_labels)
        
        assert round(res,2) == round(examples_dict[eg]['expected'],2),"example %d: res=%f,expected=%f" % (eg,res,examples_dict[eg]['expected'])
        
        print('check_computePairwiseAverageAuc: CHECKED example %d' % eg)

#check_computePairwiseAverageAuc()
