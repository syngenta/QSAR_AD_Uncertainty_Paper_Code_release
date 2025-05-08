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
##############################
#Copright (c) 2022-2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
###############################
import os,sys,re
##################################
dir_of_this_file = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname((dir_of_this_file))
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.qsar_eval import all_key_reg_stats_and_plots as RegEval
##################################
#=================================
#Selected default AD parameters:
ADMethod2Param = {}
ADMethod2Param['Tanimoto'] = 0.4 #distance threshold
consistent_k = 3
ADMethod2Param['dkNN'] = consistent_k
ADMethod2Param['RDN'] = consistent_k
ADMethod2Param['UNC'] = consistent_k
#=================================
all_global_random_seed_opts = [42,100,1,1000,657]
#=================================
#Parameters to be used for regression modelling and uncertainty estimation:
native_uncertainty_alg_variant="PI"
ml_alg_regression = "RandomForestRegressor"
nrTrees_regression=100
#other Random Forest regression modelling hyperparameters are defined in modelsADuncertaintyPkg.qsar_reg_uncertainty.RegressionICP: def fit_RF(...)
non_conformity_scaling="exp(stdev of tree predictions)"

icp_calib_fraction=0.25 #Previous applications of ACP to regression datasets in cheminformatics used this fraction, and we should be consistent.
number_of_acp_splits=20 #Previous applications of ACP to regression datasets have set this to 100-200. However, for computational efficiency, it may be appropriate to set this to 20, which has precedence for classification applications of ACP in cheminformatics.
number_of_scp_splits=5 #This is higher than some previous applications of SCP to regression, and is consistent with some applications of SCP to classifcation in cheminformatics.
sig_level_of_interest=0.32 #This of interest for all metrics other than ECE
all_sig_levels_considered_to_compute_ECE = RegEval.sigLevels.tolist()
#=================================
#Parameters to be used for classification modelling and uncertainty estimation:
ml_alg_classification = "RandomForestClassifier"
#no_cvap_folds is consistently defined by the default for the modelsADuncertaintyPkg.qsar_class_uncertainty.IVAP_CVAP_workflow_functions: def runIVAPandOrCVAPWorkflow(...):
#ivap_calib_fraction is also consistently defined by the default for the modelsADuncertaintyPkg.qsar_class_uncertainty.IVAP_CVAP_workflow_functions: def runIVAPandOrCVAPWorkflow(...):
#parameters['n_estimators'] (number of trees) is defined, along with other Random Forest classification modelling hypeparameters in modelsADuncertaintyPkg.IVAP_CVAP_workflow_functions: def specifyNativeMLModel(...):
#====================================
#Parameters to be used for evaluation of classification uncertainty estimates:
lit_precedent_delta_for_calib_plot=0.05
larger_delta_for_calib_plot=0.20
#=====================================
#-------------
raw2prettyMetricNames = {}
raw2prettyMetricNames['ba'] = 'Balanced Accuracy'
raw2prettyMetricNames['mcc'] = 'MCC'
raw2prettyMetricNames['kappa'] = 'Kappa'
raw2prettyMetricNames['auc'] = 'AUC'
raw2prettyMetricNames['precision_1'] = 'Precision (class 1)'
raw2prettyMetricNames['precision_0'] = 'Precision (class 0)'
raw2prettyMetricNames['recall_1'] = 'Recall (class 1)'
raw2prettyMetricNames['recall_0'] = 'Recall (class 0)'
raw2prettyMetricNames['strat_brier'] = 'Stratified Brier Score'
raw2prettyMetricNames['brier'] = 'Brier Score'
raw2prettyMetricNames['rmse (cal)'] = 'RMSE (cal)'
for name in ['MAD (cal)','R2 (cal)','no. compounds (class 1)','no. compounds (class 0)','no. compounds']:
    raw2prettyMetricNames[name] = name
raw2prettyMetricNames['Pearson coeff (cal)'] = 'Pearson coefficient (cal)'
raw2prettyMetricNames['Pearson coeff p-value (cal)'] = 'Pearson coefficient p-value (cal)'
raw2prettyMetricNames['Spearman coeff (cal)'] = 'Spearman coefficient (cal)'
raw2prettyMetricNames['Spearman coeff p-value (cal)'] = 'Spearman coefficient p-value (cal)'
raw2prettyMetricNames['rmse'] = 'RMSE'
raw2prettyMetricNames['MAD'] = 'MAD'
raw2prettyMetricNames['R2'] = 'R2'
raw2prettyMetricNames['Pearson'] = 'Pearson coefficient'
raw2prettyMetricNames['Pearson_Pval_one_tail'] = 'Pearson coefficient p-value'
raw2prettyMetricNames['Spearman'] = 'Spearman coefficient'
raw2prettyMetricNames['Spearman_Pval_one_tail'] = 'Spearman coefficient p-value'
raw2prettyMetricNames['validity'] = 'Validity'
raw2prettyMetricNames['efficiency'] = 'Efficiency'
raw2prettyMetricNames['ECE_new'] = 'ECE'
raw2prettyMetricNames['ENCE'] = 'ENCE'
raw2prettyMetricNames['SCC-Uncertainty'] = 'Spearman coefficient (PIs vs. residuals)'
#-------------
#Just focus on statistics used for overall assessments of winners:
#sub_1 = subset_1 = Inside applicability domain
#sub_2 = subset_2 = Outside applicability domain
regression_metric_to_expected_sub_1_minus_sub_2_sign = {'R2':1,'Pearson coefficient':1,'Spearman coefficient':1,'RMSE':-1,'Validity':1,'Efficiency':-1,'ENCE':-1,'ECE':-1,'Spearman coefficient (PIs vs. residuals)':1}
classifcation_metric_to_expected_sub_1_minus_sub_2_sign = {'Balanced Accuracy':1,'MCC':1,'AUC':1,'Kappa':1,'Stratified Brier Score':-1,'RMSE (cal)':-1,'R2 (cal)':1,'Pearson coefficient (cal)':1,'Spearman coefficient (cal)':1}
#-------------
assert all([key in raw2prettyMetricNames.values() for key in regression_metric_to_expected_sub_1_minus_sub_2_sign.keys()])
assert all([key in raw2prettyMetricNames.values() for key in classifcation_metric_to_expected_sub_1_minus_sub_2_sign.keys()])
#------------
classification_performance_metrics_for_winners_analysis = ['Balanced Accuracy','MCC','AUC','Kappa']
classification_uncertainty_metrics_for_winners_analysis = ['Stratified Brier Score','R2 (cal)','RMSE (cal)','Pearson coefficient (cal)','Spearman coefficient (cal)']
regression_performance_metrics_for_winners_analysis = ['R2','RMSE','Pearson coefficient','Spearman coefficient']
regression_uncertainty_metrics_for_winners_analysis = ['ENCE','ECE','Spearman coefficient (PIs vs. residuals)']+['Efficiency','Validity']
#------------
classification_stats_in_desired_order = ['no. compounds','no. compounds (class 1)','no. compounds (class 0)']
#Just focus on statistics used for overall assessments of winners:
classification_stats_in_desired_order += classification_performance_metrics_for_winners_analysis
classification_stats_in_desired_order += classification_uncertainty_metrics_for_winners_analysis
#Plus include class specific statistics at the end:
classification_stats_in_desired_order += ['Recall (class 1)','Recall (class 0)','Precision (class 1)','Precision (class 0)']
#Include any additional statistics we compute, in case they are of interest to some, at the end:
classification_stats_in_desired_order += ['Brier Score','MAD (cal)']
#Previous comments also apply to regression statistics:
regression_stats_in_desired_order = ['no. compounds']
regression_stats_in_desired_order += regression_performance_metrics_for_winners_analysis
regression_stats_in_desired_order += regression_uncertainty_metrics_for_winners_analysis
regression_stats_in_desired_order += ['MAD']
#----------------
assert all([s in raw2prettyMetricNames.values() for s in classification_stats_in_desired_order])

assert all([c in list(classifcation_metric_to_expected_sub_1_minus_sub_2_sign.keys()) for c in classification_performance_metrics_for_winners_analysis+classification_uncertainty_metrics_for_winners_analysis])
assert all([c in classification_performance_metrics_for_winners_analysis+classification_uncertainty_metrics_for_winners_analysis for c in  list(classifcation_metric_to_expected_sub_1_minus_sub_2_sign.keys())])

assert all([s in raw2prettyMetricNames.values() for s in regression_stats_in_desired_order])
assert all([c in list(regression_metric_to_expected_sub_1_minus_sub_2_sign.keys()) for c in regression_performance_metrics_for_winners_analysis+regression_uncertainty_metrics_for_winners_analysis])
assert all([c in regression_performance_metrics_for_winners_analysis+regression_uncertainty_metrics_for_winners_analysis for c in  list(regression_metric_to_expected_sub_1_minus_sub_2_sign.keys())])
#----------------

##############################
ad_params_col = "AD Method Parameters (k applicable to dkNN,RDN,UNC and t applicable to Tanimoto)"
endpoint_col = 'Endpoint'
test_set_type_col = 'Test Set Name (ignoring fold if applicable)'
fold_col = 'Fold (if applicable)'
rnd_seed_col = 'Random seed'
alg_col = 'Modelling Algorithm'
ad_col = 'AD Method'
ad_subset_col = 'AD Subset'
##############################
stats_metadata_cols_in_desired_order = [ad_params_col,endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col]

##############################
consistent_no_random_splits_for_AD_p_val=1000
##############################