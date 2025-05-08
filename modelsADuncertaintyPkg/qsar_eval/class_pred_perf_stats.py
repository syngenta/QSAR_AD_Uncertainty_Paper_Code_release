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
#Copyright (c) 2020-2022 Syngenta
#Contact richard.marchese_robinson@syngenta.com
#######################
#print('Initialising environment and defining functions')
#===========================================
import sys,re,glob,os,getopt,functools
import pandas as pd
import numpy as np
from collections import defaultdict
#==========================================
def recall(class_val,predicted_classes,experi_class_labels):
    
    total_experi_for_this_class = len([c for c in experi_class_labels if class_val == c])
    
    if not 0 == total_experi_for_this_class:
        
        total_correct_for_this_class = len([i for i in range(0,len(experi_class_labels)) if (experi_class_labels[i]==class_val and predicted_classes[i]==class_val)])
        
        stat_val = total_correct_for_this_class/total_experi_for_this_class
    else:
        stat_val = np.inf
    
    return stat_val

def classSize(class_val,experi_class_labels):
    return len([c for c in experi_class_labels if c == class_val])

def precision(class_val,predicted_classes,experi_class_labels):
    
    total_predicted_for_this_class = len([c for c in predicted_classes if class_val == c])
    
    if not 0 == total_predicted_for_this_class:
        
        total_correct_for_this_class = len([i for i in range(0,len(experi_class_labels)) if (experi_class_labels[i]==class_val and predicted_classes[i]==class_val)])
        
        stat_val = total_correct_for_this_class/total_predicted_for_this_class
    else:
        stat_val = np.inf
    
    return stat_val

def BalancedAccuracy(all_recall_vals):
    
    stat_val = sum(all_recall_vals)/len(all_recall_vals)
    
    return stat_val

def getWeight(class_i,class_j,class_labels_in_order):
    ########################
    #Currently implementing a linear weighting scheme as per equation (7) in the following reference:
    #Arie Ben-David
    #"Comparison of classification accuracy using Cohen’s Weighted Kappa"
    #Expert Systems with Applications
    #Volume 34, Issue 2, February 2008, Pages 825-832
    #https://doi.org/10.1016/j.eswa.2006.10.022
    #######################################################################
    
    #============================
    assert type(class_labels_in_order) == type([]),"%s" % str(type(class_labels_in_order))
    assert len(class_labels_in_order)==len(set(class_labels_in_order)),"%s" % str(class_labels_in_order)
    assert class_i in class_labels_in_order,"class_i=%s,class_labels_in_order=%s" % (class_i,str(class_labels_in_order))
    assert class_j in class_labels_in_order,"class_j=%s,class_labels_in_order=%s" % (class_j,str(class_labels_in_order))
    #============================
    
    i = class_labels_in_order.index(class_i)+1
    j = class_labels_in_order.index(class_j)+1
    
    w_ij = 1 - (abs(i - j)/(len(class_labels_in_order)-1))
    
    assert type(w_ij) == type(0.5)
    assert w_ij <= 1 and w_ij >= 0,"%f" % w_ij
    
    return w_ij

def weighted_kappa(predicted_classes,experi_class_labels,class_labels_in_order,unweighted=False):
    #######################################################################
    #Here, I propose computing a Weighted Kappa per equations (6) in the following reference:
    #Arie Ben-David
    #"Comparison of classification accuracy using Cohen’s Weighted Kappa"
    #Expert Systems with Applications
    #Volume 34, Issue 2, February 2008, Pages 825-832
    #https://doi.org/10.1016/j.eswa.2006.10.022
    #######################################################################
    
    #===================================
    assert type(experi_class_labels) == type([])
    assert type(experi_class_labels) == type(predicted_classes)
    assert len(predicted_classes) == len(experi_class_labels),"len(predicted_classes)= %d, len(experi_class_labels) = %d"  % (len(predicted_classes),len(experi_class_labels))
    experi_labels_not_expected = [c for c in experi_class_labels if not c in class_labels_in_order] #18/03/21: cannot see full error message in PP!
    assert 0 == len(experi_labels_not_expected),"experi_labels_not_expected=%s,class_labels_in_order=%s,experi_class_labels=%s" % (str(experi_labels_not_expected),str(class_labels_in_order),str(experi_class_labels)) #18/03/21: cannot see full error message in PP!
    #===================================
    
    weighted_agreement = 0
    weighted_agreement_expected_due_to_chance = 0
    
    for class_i in class_labels_in_order:
        for class_j in class_labels_in_order:
            try:
                P_i = len([c for c in experi_class_labels if c == class_i])/len(experi_class_labels)
            except ZeroDivisionError:
                continue
            P_j = len([c for c in predicted_classes if c == class_j])/len(experi_class_labels)
            P_ij = len([index for index in range(0,len(experi_class_labels)) if (experi_class_labels[index] == class_i and predicted_classes[index] == class_j)])/len(experi_class_labels)
            
            #-------------------------------
            for p_name in ['P_i','P_j','P_ij']:
                if 'P_i' == p_name:
                    prob = P_i
                elif 'P_j' == p_name:
                    prob = P_j
                elif 'P_ij' == p_name:
                    prob = P_ij
                else:
                    raise Exception('p_name=%s')
                    
                assert type(prob) == type(0.5)
                assert prob <= 1 and prob >= 0,"p_name=%s,value=%f" % (p_name,prob)
            #-------------------------------
            
            if not unweighted:
                w_ij = getWeight(class_i,class_j,class_labels_in_order)
            else:
                if class_i == class_j:
                    w_ij = 1
                else:
                    w_ij = 0
            
            weighted_agreement += (w_ij*P_ij)
            
            weighted_agreement_expected_due_to_chance += (w_ij*P_i*P_j)
            
    stat_val = (weighted_agreement - weighted_agreement_expected_due_to_chance) / (1 - weighted_agreement_expected_due_to_chance)
    
    return stat_val

def check_weighted_kappa():
    #######################################################################
    #Here, I propose checking this based upon the input and output (linear weighting) given in Table 9 of the following reference:
    #Arie Ben-David
    #"Comparison of classification accuracy using Cohen’s Weighted Kappa"
    #Expert Systems with Applications
    #Volume 34, Issue 2, February 2008, Pages 825-832
    #https://doi.org/10.1016/j.eswa.2006.10.022
    #######################################################################
    
    print('Checking linear weighted Kappa')
    
    table_7_predicted_classes = []
    table_7_experi_class_labels = []
    
    table_7_predicted_classes += ['Excellent']*350
    table_7_experi_class_labels += ['Excellent']*250
    table_7_experi_class_labels += ['Good']*0
    table_7_experi_class_labels += ['Average']*0
    table_7_experi_class_labels += ['Bad']*100
    
    table_7_predicted_classes += ['Good']*250
    table_7_experi_class_labels += ['Excellent']*0
    table_7_experi_class_labels += ['Good']*250
    table_7_experi_class_labels += ['Average']*0
    table_7_experi_class_labels += ['Bad']*0
    
    table_7_predicted_classes += ['Average']*250
    table_7_experi_class_labels += ['Excellent']*0
    table_7_experi_class_labels += ['Good']*0
    table_7_experi_class_labels += ['Average']*250
    table_7_experi_class_labels += ['Bad']*0
    
    table_7_predicted_classes += ['Bad']*350 #There must be a typo in Table 7!
    table_7_experi_class_labels += ['Excellent']*100
    table_7_experi_class_labels += ['Good']*0
    table_7_experi_class_labels += ['Average']*0
    table_7_experi_class_labels += ['Bad']*250
    
    
    table_7_classes_in_order = ['Excellent','Good','Average','Bad']
    
    table_7_linear_weighted_kappa_val = weighted_kappa(table_7_predicted_classes,table_7_experi_class_labels,table_7_classes_in_order)
    
    assert 0.623 == round(table_7_linear_weighted_kappa_val,3),"%f" % table_7_linear_weighted_kappa_val
    
    print('CHECKED linear weighted Kappa')

#check_weighted_kappa()
#==============================================================
def check_kappa():
    #######################################################################
    #Here, I propose checking this based upon the input and output (linear weighting) given in Table 9 of the following reference:
    #Arie Ben-David
    #"Comparison of classification accuracy using Cohen’s Weighted Kappa"
    #Expert Systems with Applications
    #Volume 34, Issue 2, February 2008, Pages 825-832
    #https://doi.org/10.1016/j.eswa.2006.10.022
    #######################################################################
    
    print('Checking Kappa')
    
    table_7_predicted_classes = []
    table_7_experi_class_labels = []
    
    table_7_predicted_classes += ['Excellent']*350
    table_7_experi_class_labels += ['Excellent']*250
    table_7_experi_class_labels += ['Good']*0
    table_7_experi_class_labels += ['Average']*0
    table_7_experi_class_labels += ['Bad']*100
    
    table_7_predicted_classes += ['Good']*250
    table_7_experi_class_labels += ['Excellent']*0
    table_7_experi_class_labels += ['Good']*250
    table_7_experi_class_labels += ['Average']*0
    table_7_experi_class_labels += ['Bad']*0
    
    table_7_predicted_classes += ['Average']*250
    table_7_experi_class_labels += ['Excellent']*0
    table_7_experi_class_labels += ['Good']*0
    table_7_experi_class_labels += ['Average']*250
    table_7_experi_class_labels += ['Bad']*0
    
    table_7_predicted_classes += ['Bad']*350 #There must be a typo in Table 7!
    table_7_experi_class_labels += ['Excellent']*100
    table_7_experi_class_labels += ['Good']*0
    table_7_experi_class_labels += ['Average']*0
    table_7_experi_class_labels += ['Bad']*250
    
    
    table_7_classes_in_order = ['Excellent','Good','Average','Bad']
    
    table_7_linear_kappa_val = weighted_kappa(table_7_predicted_classes,table_7_experi_class_labels,table_7_classes_in_order,unweighted=True)
    
    assert 0.776 == round(table_7_linear_kappa_val,3),"%f" % table_7_linear_kappa_val
    
    print('CHECKED Kappa')

#check_kappa()
