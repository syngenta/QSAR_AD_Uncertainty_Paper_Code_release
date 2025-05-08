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
#Copyright (c)  2020-2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#Where indicated (see ZH comments), functions and lines of code were copied from files prepared by Zied Hosni (University of Sheffield), whilst working on a Syngenta funded project.
#######################
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score,brier_score_loss
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pytest import approx
#------------------------
from .class_prob_perf_stats import computeVariantBrierLossForKclasses,computePairwiseAverageAuc,compute_Stratified_ProbabilisticLossForKclasses #For two categories, these should be redundant, given the above imports from sklearn: computeVariantBrierLossForKclasses,computePairwiseAverageAuc
from .class_pred_perf_stats import recall,precision,BalancedAccuracy,weighted_kappa,classSize #Some of these should be redundant, given the above imports from sklearn: all of these functions
from .reg_perf_pred_stats import rmse, MAD, coeffOfDetermination, PearsonCoeff, PearsonCoeffPval,SpearmanCoeff,SpearmanCoeffPval,SpearmanCoeffPval
from ..utils.ML_utils import predictedBinaryClassFromProbClass1
from .enforce_minimum_no_instances import size_of_inputs_for_stats_is_big_enough,get_no_instances
#-------------------------

def computeRecallAndPrecision(class_val,experi_class_labels,y_pred,check_consistency=True):
    
    predicted_classes = y_pred
    
    precision_val = precision(class_val,predicted_classes,experi_class_labels)
    
    recall_val = recall(class_val,predicted_classes,experi_class_labels)
    
    if check_consistency:
        #We assume two classes only here!
        alt_precision,alt_recall,alt_fbeta_score,alt_support = precision_recall_fscore_support(y_true=experi_class_labels,y_pred=y_pred,pos_label=class_val,average='binary')  #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        
        if not np.isinf(precision_val):
        
            assert precision_val==approx(alt_precision),"precision_val = {}, alt_precision={}".format(precision_val,alt_precision)
        
        else:
            #UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
            
            assert 0 == alt_precision,"alt_precision = {}".format(alt_precision)
            
            precision_val = None
        
        if not np.isinf(recall_val):
            assert recall_val==approx(alt_recall),"recall_val = {}, alt_recall={}".format(recall_val,alt_recall)
        
        else:
            assert 0 == alt_recall,"alt_recall = {}".format(alt_recall)
            
            recall_val = None
    
    return precision_val,recall_val

def classesAreOneOrZeroAndOneClassMissing(experi_class_labels):
    #------------------------
    class_labels_in_order = [1, 0]
    
    if not 0 == len([c for c in experi_class_labels if not c in class_labels_in_order]): raise Exception('Class labels should be 1 or 0! Unexpected class labels = {}'.format([c for c in experi_class_labels if not c in class_labels_in_order]))
    
    if 0 == len([c for c in experi_class_labels if 1 == c]) or 0 == len([c for c in experi_class_labels if 0 == c]):
        return True
    else:
        return False
    #------------------------

def computeBA(experi_class_labels, y_pred,check_consistency=True,only_1_0_classes=True):
    
    if only_1_0_classes:
        if classesAreOneOrZeroAndOneClassMissing(experi_class_labels):
            return None
    
    try:
        BA = balanced_accuracy_score(experi_class_labels, y_pred) #copied from ZH latest script
        
        if check_consistency:
            unique_classes = list(set(experi_class_labels))
            
            predicted_classes = y_pred
            
            all_recall_vals = [recall(class_val,predicted_classes,experi_class_labels) for class_val in unique_classes]
            
            alt_BA = BalancedAccuracy(all_recall_vals)
            
            assert BA==approx(alt_BA),"BA = {}, alt_BA={}".format(BA,alt_BA)
    except ZeroDivisionError as err:
        BA = None
    
    return BA

def computeMCC(experi_class_labels, y_pred):
    if not 1 == len(set(experi_class_labels)):
        mcc = matthews_corrcoef(experi_class_labels, y_pred)
    else:
        mcc = None
    
    return mcc

def computeAUC_TwoCategories(experi_class_labels, class_1_probs_in_order,check_consistency=True):
    
    if classesAreOneOrZeroAndOneClassMissing(experi_class_labels):
        return None
    
    class_labels_in_order = [1, 0]
    
    try:
        auc = roc_auc_score(experi_class_labels, class_1_probs_in_order) #copied from ZH latest script
    
        if check_consistency:
        
            probs_of_all_classes_in_order_for_all_instances = [[p,(1-p)] for p in class_1_probs_in_order]
        
            alt_auc = computePairwiseAverageAuc(probs_of_all_classes_in_order_for_all_instances, class_labels_in_order,experi_class_labels)
        
            assert auc==approx(alt_auc),"auc = {}, alt_auc={}".format(auc,alt_auc)
    except ZeroDivisionError:
        auc = None
    
    return auc

def computeKappa(experi_class_labels, y_pred,check_consistency=True):

    if classesAreOneOrZeroAndOneClassMissing(experi_class_labels):
        return None
    
    try:
        kappa = cohen_kappa_score(experi_class_labels, y_pred, labels=None, weights=None,sample_weight=None) #copied from ZH latest script
        
        if check_consistency:
            
            classes_in_order = list(set(experi_class_labels))
            classes_in_order.sort() #Order does not actually matter if not worried about ordinality, becuase not computing weighted Kapp
            
            alt_kappa = weighted_kappa(y_pred,experi_class_labels,classes_in_order,unweighted=True)
            
            
            assert kappa==approx(alt_kappa),"kappa = {}, alt_kappa={}".format(kappa,alt_kappa)
            
    except ZeroDivisionError:
        kappa = None
    
    return kappa

def getPredClassProb(class_1_prob, predicted_class): #copied from ZH extraFunctions.py
    assert predicted_class in [1, 0], predicted_class
    assert class_1_prob >= 0 and class_1_prob <= 1, class_1_prob
    if 1 == predicted_class:
        return class_1_prob
    else:
        return (1 - class_1_prob)

def computeBrier_TwoCategories(experi_class_labels, class_1_probs_in_order,check_consistency=True):
    #------------------------
    class_labels_in_order = [1, 0]
    
    if not 0 == len([c for c in experi_class_labels if not c in class_labels_in_order]): raise Exception('Unexpected class labels + {}'.format([c for c in experi_class_labels if not c in class_labels_in_order]))
    #------------------------
    
    
    brier = brier_score_loss(experi_class_labels, class_1_probs_in_order) #copied from ZH latest script
    
    if check_consistency:
        
        pred_class_prob_tuples = [(predictedBinaryClassFromProbClass1(prob_class_1,thresh=0.5),getPredClassProb(class_1_prob=prob_class_1,predicted_class=predictedBinaryClassFromProbClass1(prob_class_1,thresh=0.5))) for prob_class_1 in class_1_probs_in_order]
        
        alt_brier = computeVariantBrierLossForKclasses(pred_class_prob_tuples, experi_class_labels)
        
        assert brier==approx(alt_brier),"brier = {}, alt_brier={}".format(brier,alt_brier)
    
    return brier

def computeStratifiedBrier_TwoCategories(class_1_probs_in_order,experi_class_labels):
    
    #------------------------
    class_labels_in_order = [1, 0]
    
    if not 0 == len([c for c in experi_class_labels if not c in class_labels_in_order]): raise Exception('Unexpected class labels + {}'.format([c for c in experi_class_labels if not c in class_labels_in_order]))
    #------------------------
    
    pred_class_prob_tuples = [(predictedBinaryClassFromProbClass1(prob_class_1,thresh=0.5),getPredClassProb(class_1_prob=prob_class_1,predicted_class=predictedBinaryClassFromProbClass1(prob_class_1,thresh=0.5))) for prob_class_1 in class_1_probs_in_order]
    
    try:
        strat_brier = compute_Stratified_ProbabilisticLossForKclasses(pred_class_prob_tuples, class_labels_in_order, experi_class_labels,type_of_loss='variant_Brier') #copied from ZH latest script [checked wrt class_prob_perf_stats.py]
    except ZeroDivisionError:
        strat_brier = None
    
    return strat_brier

def get_experi_class_1_probs(class_1_probs_in_order,experi_class_labels,delta_prob):
    ObservedProbability = []
    #print("class_1_probs_in_order", class_1_probs_in_order)
    for item_i in class_1_probs_in_order:
        experi_class_1_in_delta_bin_count = 0
        experi_class_0_in_delta_bin_count = 0
        i = 0
        for item_j in class_1_probs_in_order:
            diff = abs(item_j - item_i)
            if diff < delta_prob:
                if experi_class_labels[i] == 1:
                    experi_class_1_in_delta_bin_count += 1
                elif experi_class_labels[i] == 0:
                    experi_class_0_in_delta_bin_count += 1
                else:
                    raise Exception("experi_class_labels[i] = {0}".format(experi_class_labels[i]))
            i += 1

        ObservedProbability.append(float(experi_class_1_in_delta_bin_count / (experi_class_1_in_delta_bin_count + experi_class_0_in_delta_bin_count)))

    return ObservedProbability

def computeDeltaCalibrationPlot(class_1_probs_in_order, delta_probs, experi_class_labels, path,include_Spearman_pvalue=False,skip_delta_plot_image=False): #copied and adapted ZH function from extraFunctions.py
    
    #------------------------
    class_labels_in_order = [1, 0]
    
    if not 0 == len([c for c in experi_class_labels if not c in class_labels_in_order]): raise Exception('Unexpected class labels + {}'.format([c for c in experi_class_labels if not c in class_labels_in_order]))
    #------------------------
    
    CalibMetrics = []
    for delta_prob in delta_probs:
        path_png = path + "delta_%s.png" % delta_prob
        
        ObservedProbability = get_experi_class_1_probs(class_1_probs_in_order,experi_class_labels,delta_prob)

        #import numpy as np
        # class_1_probs_in_order = [x for x in class_1_probs_in_order]
        expected_prob = [float(x) for x in class_1_probs_in_order]
        #print("expected_prob", expected_prob)
        #print("ObservedProbability", ObservedProbability)
        rmseCal = rmse(ObservedProbability,expected_prob)
        MADCal = MAD(ObservedProbability,expected_prob)
        coeffOfDeterminationCal = coeffOfDetermination(ObservedProbability,expected_prob)
        #print("class_1_probs_in_order", class_1_probs_in_order)
        #print("delta_probs", delta_probs)
        PearsonCoeffCal = PearsonCoeff(ObservedProbability,expected_prob)
        PearsonCoeffPvalCal = PearsonCoeffPval(ObservedProbability,expected_prob)
        SpearmanCoeffCal = SpearmanCoeff(ObservedProbability,expected_prob)
        if not include_Spearman_pvalue:
            CalibMetrics.append(
                [rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal])
        else:
            SpearmanCoeffPvalCal = SpearmanCoeffPval(ObservedProbability,expected_prob)
            
            CalibMetrics.append([rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal])
        
        if not skip_delta_plot_image:
            sns.set_style("white")
            
            sns.set_style("white")
            g = sns.jointplot(x=expected_prob, y=ObservedProbability, kind='scatter', color='royalblue')  # kind='reg'
            # ax.annotate(stats.pearsonr)
            #=======================
            r = PearsonCoeffCal#r, p = stats.pearsonr(ObservedProbability,expected_prob) #vs. ZH code, no need to compute these twice!
            p = PearsonCoeffPvalCal
            #=======================
            g.ax_joint.annotate(f'$\\rho = {r:.3f}, p = {p:.3f}, SPR = {SpearmanCoeffCal:.3f}$',
                                xy=(0.1, 0.9), xycoords='axes fraction',
                                ha='left', va='center',
                                bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'}, fontsize=10)
            expected_prob, ObservedProbability = zip(*sorted(zip(ObservedProbability,expected_prob)))
            g.ax_joint.plot(ObservedProbability,expected_prob)
            # pyplot.plot([0, 1], [0, 1], linestyle='--')
            # Draw a line of x=y
            x0, x1 = g.ax_joint.get_xlim()
            y0, y1 = g.ax_joint.get_ylim()
            lims = [1, 0]
            g.ax_joint.plot(lims, lims, '-r')
            
            g.set_axis_labels(xlabel='Expected Probability', ylabel='Observed probability', size=10)
            plt.tight_layout()
            # plt.scatter(ObservedProbability,expected_prob)
            #plt.title('Delta-Binned Calibration Plot: Delta={0:.3f}'.format(delta_prob)) #pyplot replaced with plt in these lines (vs. ZH code)
            plt.xlabel('Expected Probability')
            plt.ylabel('Observed probability')
            # pyplot.show()
            plt.savefig(path_png)
            plt.close()
    return (CalibMetrics)

def computeAllClassMetrics(test_y,predicted_y,probs_for_class_1,method,subset_name,output_dir,delta_for_calib_plot=0.05,smallest_no_cmpds_to_compute_stats=2):
    
    no_cmpds_1 = classSize(class_val=1,experi_class_labels=test_y)
    
    no_cmpds_0 = classSize(class_val=0,experi_class_labels=test_y)
        
    no_cmpds = (no_cmpds_0+no_cmpds_1)
        
    assert no_cmpds == len(test_y) and no_cmpds == len(predicted_y),"no_cmpds = {}, test_y = {}, predicted_y = {}".format(no_cmpds,test_y,predicted_y)
    
    if size_of_inputs_for_stats_is_big_enough(test_y,predicted_y,limit=smallest_no_cmpds_to_compute_stats):
        #---------------
        assert get_no_instances(predicted_y) == get_no_instances(probs_for_class_1),f'get_no_instances(predicted_y)={get_no_instances(predicted_y)} vs. get_no_instances(probs_for_class_1)={get_no_instances(probs_for_class_1)}'
        #----------------
    
        ba = computeBA(experi_class_labels=test_y, y_pred=predicted_y)
            
        mcc = computeMCC(experi_class_labels=test_y, y_pred=predicted_y)
        
        auc = computeAUC_TwoCategories(experi_class_labels=test_y, class_1_probs_in_order=probs_for_class_1)
        
        kappa = computeKappa(experi_class_labels=test_y, y_pred=predicted_y)
        
        precision_1,recall_1 = computeRecallAndPrecision(class_val=1,experi_class_labels=test_y, y_pred=predicted_y)
        
        precision_0,recall_0 = computeRecallAndPrecision(class_val=0,experi_class_labels=test_y, y_pred=predicted_y)
        
        brier = computeBrier_TwoCategories(experi_class_labels=test_y, class_1_probs_in_order=probs_for_class_1)
        
        strat_brier = computeStratifiedBrier_TwoCategories(class_1_probs_in_order=probs_for_class_1,experi_class_labels=test_y)
        
        CalibMetrics = computeDeltaCalibrationPlot(class_1_probs_in_order=probs_for_class_1, delta_probs=[delta_for_calib_plot], experi_class_labels=test_y, path=os.path.sep.join([output_dir,'{}_{}_Delta_Calibration_Plot'.format(subset_name,method)]),include_Spearman_pvalue=True)
        
        rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal = CalibMetrics[0]
    else:
        print(f'Too few compounds (less than {smallest_no_cmpds_to_compute_stats}) to compute statistics!')
        
        precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal = [None]*17
    
    return precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal,no_cmpds_1,no_cmpds_0,no_cmpds

def map_all_class_stats_onto_default_names(precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal,no_cmpds_1,no_cmpds_0,no_cmpds):
    dict_of_current_stats = {}
    
    dict_of_current_stats['ba'] = ba
    dict_of_current_stats['mcc'] = mcc
    dict_of_current_stats['auc'] = auc
    dict_of_current_stats['kappa'] = kappa
    dict_of_current_stats['precision_1'] = precision_1
    dict_of_current_stats['precision_0'] = precision_0
    dict_of_current_stats['recall_1'] = recall_1
    dict_of_current_stats['recall_0'] = recall_0
    dict_of_current_stats['brier'] = brier
    dict_of_current_stats['strat_brier'] = strat_brier
    dict_of_current_stats['rmse (cal)'] = rmseCal
    dict_of_current_stats['MAD (cal)'] = MADCal
    dict_of_current_stats['R2 (cal)'] = coeffOfDeterminationCal
    dict_of_current_stats['Pearson coeff (cal)'] = PearsonCoeffCal
    dict_of_current_stats['Pearson coeff p-value (cal)'] = PearsonCoeffPvalCal
    dict_of_current_stats['Spearman coeff (cal)'] = SpearmanCoeffCal
    dict_of_current_stats['Spearman coeff p-value (cal)'] = SpearmanCoeffPvalCal
    dict_of_current_stats['no. compounds (class 1)'] = no_cmpds_1
    dict_of_current_stats['no. compounds (class 0)'] = no_cmpds_0
    dict_of_current_stats['no. compounds'] = no_cmpds
    
    return dict_of_current_stats
