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
import os,sys,time,pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import pickle
import shutil
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
top_scripts_dir = os.path.dirname(dir_of_this_script)
pkg_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
top_ds_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'SyngentaData','DT50_Updates'])
top_calc_dir = os.path.sep.join([top_ds_dir,'Calc'])
all_stats_pkl = os.path.sep.join([top_calc_dir,'DT50.All.Updates.Stats.pkl'])
all_raw_pkl = os.path.sep.join([top_calc_dir,'DT50.All.Updates.Raw.pkl'])
###################################
sys.path.append(top_scripts_dir)
from consistent_parameters_for_all_modelling_runs import ADMethod2Param, ml_alg_classification,lit_precedent_delta_for_calib_plot,larger_delta_for_calib_plot
from consistent_parameters_for_all_modelling_runs import all_global_random_seed_opts
ml_alg = ml_alg_classification
from recommended_defaults import ep_type_matched_to_default_AD_uncertainty_methods
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.qsar_class_uncertainty import define_native_ML_baseline as nml
from modelsADuncertaintyPkg.qsar_class_uncertainty import IVAP_CVAP_workflow_functions as vap
from modelsADuncertaintyPkg.qsar_eval import all_key_class_stats_and_plots as ClassEval
from modelsADuncertaintyPkg.qsar_class_uncertainty.parse_native_or_venn_abers_output import getProbsForClass1
from modelsADuncertaintyPkg.utils.load_example_datasets_for_code_checks import generateExampleDatasetForChecking_BCWD
from modelsADuncertaintyPkg.utils.ML_utils import getXandY,singleRandomSplit,makeYLabelsNumeric
from modelsADuncertaintyPkg.utils.ML_utils import predictedBinaryClassFromProbClass1
from modelsADuncertaintyPkg.utils.ML_utils import getTestYFromTestIds as getSubsetYFromSubsetIds
from modelsADuncertaintyPkg.utils.basic_utils import findDups,convertDefaultDictDictIntoDataFrame,neverEndingDefaultDict,findDups
from modelsADuncertaintyPkg.CheminformaticsUtils import chem_data_parsing_utils as ChemDataParser
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import findInsideOutsideADTestIds,getInsideOutsideADSubsets
#----------------------------------------------------------------------------
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.CheminformaticsUtils.chem_data_parsing_utils import getFpsDfAndDataDfSkippingFailedMols
from modelsADuncertaintyPkg.utils.ML_utils import checkTrainTestHaveUniqueDistinctIDs,makeIDsNumeric,prepareInputsForModellingAndUncertainty
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import getInputRequiredForModellingAndAD,getADSubsetTestIDsInOrder
#----------------------------------------------------------------------------
#Dataset globals:
id_col = 'CSN' 
smiles_col = 'Canonical_Smiles'
class_label_col = 'Scenario_4'
original_class_1 = '<100 days'
original_class_0 = '>100 days'
class_1 = original_class_1
class_0 = original_class_0

#-----------------------------------------------------------------------------
#Dataset requirements for each update:
min_no_cmpds_per_class = 25 #Meaning each test set would need at least 50 compounds c.f. Sheridan (2022)
#--------------------------------------
updates_in_order = ['T{}'.format(t) for t in range(1,7)]
update2DataFiles = defaultdict(dict)
for update_version in updates_in_order:
    update2DataFiles[update_version]['train_file'] = os.path.sep.join([top_ds_dir,update_version,'All_Cleaned_SED_Old_CSNs_{}.xlsx'.format(update_version)])
    update2DataFiles[update_version]['test_file'] = os.path.sep.join([top_ds_dir,update_version,'All_Cleaned_SED_New_CSNs_{}.xlsx'.format(update_version)])
#---------------------------------------
#=================================
selected_uncertainty_methods = [ep_type_matched_to_default_AD_uncertainty_methods['Classification']['uncertainty_method']]
selected_ad_methods = [ep_type_matched_to_default_AD_uncertainty_methods['Classification']['AD_method_name']]
#=================================


def updateClassLabels(row,class_label_col,class_1,original_class_1,original_class_0):
    
    orig_class_label = row[class_label_col]
    
    if orig_class_label == original_class_1:
        row[class_label_col] = class_1
    elif orig_class_label == original_class_0:
        pass
    else:
        raise Exception('Unrecognised original class label = {}'.format(orig_class_label))
    
    return row

def prepareClassData_OneFile(data_file,id_col,smiles_col,class_label_col,class_1,original_class_1,original_class_0):
    
    df = pd.read_excel(data_file,engine='openpyxl')
    
    df = df[[id_col,smiles_col,class_label_col]]
    
    df=df.dropna()
    
    df = df.apply(updateClassLabels, axis=1,args=(class_label_col,class_1,original_class_1,original_class_0))
    
    return df

def prepareClassData(train_file,test_file,id_col,smiles_col,class_label_col,class_1,original_class_1,original_class_0):
    
    for train_test in ['Train','Test']:
        if 'Train' == train_test:
            train_df = prepareClassData_OneFile(train_file,id_col,smiles_col,class_label_col,class_1,original_class_1,original_class_0)
        elif 'Test' == train_test:
            test_df = prepareClassData_OneFile(test_file,id_col,smiles_col,class_label_col,class_1,original_class_1,original_class_0)
        else:
            raise Exception('Unrecognised train_test label = {}'.format(train_test))
    
    return train_df,test_df
    


def populateDictWithStats(dict_of_stats,AD_method_name,method,AD_subset,precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal,no_cmpds_1,no_cmpds_0,no_cmpds):
    dict_of_current_stats = ClassEval.map_all_class_stats_onto_default_names(precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal,no_cmpds_1,no_cmpds_0,no_cmpds)
    
    for stat_name in dict_of_current_stats.keys():
        dict_of_stats[AD_method_name][method][AD_subset][stat_name] = dict_of_current_stats[stat_name]


def writeStatsFiles(dict_of_stats,calc_dir,selected_ad_methods,selected_uncertainty_methods,endpoint_name='DT50'):
    
    statsFilesDict = defaultdict(dict)
    
    for AD_method_name in selected_ad_methods:
        for method in selected_uncertainty_methods:
            dd = dict_of_stats[AD_method_name][method]
            
            df = convertDefaultDictDictIntoDataFrame(dd,col_name_for_first_key='AD Subset')
            
            stats_file = os.path.sep.join([calc_dir,'e={}_ad={}_m={}_Classification_Statistics.csv'.format(endpoint_name,AD_method_name,method)])
            
            statsFilesDict[AD_method_name][method] = stats_file
            
            df.to_csv(stats_file,index=False)
    
    return statsFilesDict


def testSetHasEnoughCompoundsPerClass(min_no_cmpds_per_class,test_df,class_label_col,class_1,original_class_0,test_y):
    
    if test_y is None:
        no_cmpds_in_class_1 = test_df[test_df[class_label_col].isin([class_1])].shape[0]
    
        no_cmpds_in_class_0 = test_df[test_df[class_label_col].isin([class_0])].shape[0]
    else:
        no_cmpds_in_class_1 = test_y.isin([class_1]).sum() #Only True elements counted when sum() applied to a boolean series!
    
        no_cmpds_in_class_0 = test_y.isin([class_0]).sum()
    
    if (no_cmpds_in_class_1 >= min_no_cmpds_per_class) and (no_cmpds_in_class_0 >= min_no_cmpds_per_class):
        return True
    else:
        return False

def updateDictOfRawInsideVsOutsideADResults(dict_of_raw_results,AD_method_name,method,AD_subset,subset_test_ids,subset_test_y,subset_probs_for_class_1,subset_predicted_y):
    
    dict_of_raw_results[AD_method_name][method][AD_subset]['subset_test_ids'] = subset_test_ids
    dict_of_raw_results[AD_method_name][method][AD_subset]['subset_test_y'] = subset_test_y
    dict_of_raw_results[AD_method_name][method][AD_subset]['subset_probs_for_class_1'] = subset_probs_for_class_1
    dict_of_raw_results[AD_method_name][method][AD_subset]['subset_predicted_y'] = subset_predicted_y

def modellingEvalAnalysisForOneModelVersion(update_version,selected_uncertainty_methods,selected_ad_methods,rand_seed,min_no_cmpds_per_class=min_no_cmpds_per_class,id_col=id_col,smiles_col=smiles_col,class_label_col=class_label_col,class_1=class_1,original_class_1=original_class_1,original_class_0=original_class_0):
    
    print(f'Running modellingEvalAnalysisForOneModelVersion(...) for update_version={update_version},random seed={rand_seed}')
    
    train_file = update2DataFiles[update_version]['train_file']
    
    test_file = update2DataFiles[update_version]['test_file']
    
    train_df,test_df = prepareClassData(train_file,test_file,id_col,smiles_col,class_label_col,class_1,original_class_1,original_class_0)
    
    #==================================
    #Should not be needed for SYN datasets:
    train_df = ChemDataParser.removeDuplicateMolsTakingAccountOfActivities(df=train_df,smiles_col=smiles_col,activity_col=class_label_col,unique_mol_id_col=id_col,type_of_activities="classification")
    
    test_df = ChemDataParser.removeDuplicateMolsTakingAccountOfActivities(df=test_df,smiles_col=smiles_col,activity_col=class_label_col,unique_mol_id_col=id_col,type_of_activities="classification")
    #===================================
    
    checkTrainTestHaveUniqueDistinctIDs(train_df,test_df,id_col)
    
    if not testSetHasEnoughCompoundsPerClass(min_no_cmpds_per_class,test_df,class_label_col,class_1,original_class_0,test_y=None):
        print('Skipping the rest of modellingEvalAnalysisForOneModelVersion(...) for update_version={}!'.format(update_version))
        return 0
    
    train_df = makeIDsNumeric(df=train_df,id_col=id_col)
    
    test_df = makeIDsNumeric(df=test_df,id_col=id_col)
    
    fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids = getInputRequiredForModellingAndAD(train_df,test_df,id_col,smiles_col,class_label_col,class_1)
    
    if not testSetHasEnoughCompoundsPerClass(min_no_cmpds_per_class,None,class_label_col,class_1,original_class_0,test_y):
        print('[After fingerprint calculation] Skipping the rest of modellingEvalAnalysisForOneModelVersion(...) for update_version={}!'.format(update_version))
        return 0
    
    
    calc_dir = os.path.sep.join([top_calc_dir,f'{update_version}_s={rand_seed}'])
    
    createOrReplaceDir(dir_=calc_dir)
    
    train_inc_calib_x,train_inc_calib_y,test_x = prepareInputsForModellingAndUncertainty(X_train_and_ids_df,train_y,X_test_and_ids_df,id_col)
    
    
    ########################
    #Whilst the IVAP,CVAP, native modelling code should perform this conversion internally, for the training set class labels, the classification modelling statistics code still assumes class labels are 1 and 0!
    #So, make this change here and ensure class_1 and class_0 are updated for consistency!
    train_inc_calib_y = makeYLabelsNumeric(train_inc_calib_y, class_1=class_1, class_0=class_0,return_as_pandas_series=True)
    test_y = makeYLabelsNumeric(test_y, class_1=class_1, class_0=class_0,return_as_pandas_series=True)
    new_class_1 = 1
    new_class_0 = 0
    #######################
    
    #==================
    #Redundant:
    del train_y
    #==================
    
    if 'IVAP' in selected_uncertainty_methods:
        run_ivap = True
    else:
        run_ivap = False
    
    if 'CVAP' in selected_uncertainty_methods:
        run_cvap = True
    else:
        run_cvap = False
    
    
    IVAP_calib_score_label_tuples,IVAP_model,IVAP_test_id_to_pred_class_class_1_prob_p1_p0,CVAP_calib_score_label_tuples_per_fold,CVAP_models_per_fold,CVAP_test_id_to_pred_class_class_1_prob_p1_p0 = vap.runIVAPandOrCVAPWorkflow(train_inc_calib_x,train_inc_calib_y,test_x,train_ids,test_ids,ml_alg,rand_seed,new_class_1,test_y,run_ivap=run_ivap,run_cvap=run_cvap)
    
    if 'Native' in selected_uncertainty_methods:
        model,native_test_id_to_pred_class_class_1_prob = nml.getNativeModelAndTestId2PredClassAndClass1Prob(train_inc_calib_x,train_inc_calib_y,ml_alg,rand_seed,new_class_1,test_x,test_ids)
        
        del model
    else:
        native_test_id_to_pred_class_class_1_prob = None
    
    dict_of_stats = neverEndingDefaultDict()
    
    dict_of_raw_results = neverEndingDefaultDict()
    
    for AD_method_name in selected_ad_methods:
        
        test_id_status_dict = findInsideOutsideADTestIds(X_train=X_train_and_ids_df, X_test=X_test_and_ids_df, fps_train=fps_train, fps_test=fps_test, threshold=ADMethod2Param[AD_method_name], id_col=id_col,rand_seed=rand_seed, endpoint_col=class_label_col, AD_method_name=AD_method_name,test_ids=test_ids, y_train=train_inc_calib_y, y_test=None, regression=False,class_1_label=new_class_1,class_0_label=new_class_0)
        
        for AD_subset in ['All','Inside','Outside']:
            
            subset_test_ids = getADSubsetTestIDsInOrder(AD_subset,test_id_status_dict,test_ids)
            
            for method in selected_uncertainty_methods:
            
                if not 0 == len(subset_test_ids):
                    #==================================================
                    subset_test_y = getSubsetYFromSubsetIds(test_ids,pd.Series(test_y),subset_test_ids).tolist()
                    
                    
                    subset_probs_for_class_1 = getProbsForClass1(method,IVAP_test_id_to_pred_class_class_1_prob_p1_p0,CVAP_test_id_to_pred_class_class_1_prob_p1_p0,native_test_id_to_pred_class_class_1_prob,subset_test_ids)
                    
                    subset_predicted_y = [predictedBinaryClassFromProbClass1(p) for p in subset_probs_for_class_1]
                    #===================================================
                    
                    updateDictOfRawInsideVsOutsideADResults(dict_of_raw_results,AD_method_name,method,AD_subset,subset_test_ids,subset_test_y,subset_probs_for_class_1,subset_predicted_y)
                    
                    #----------------------------------------------------
                    
                    precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal,no_cmpds_1,no_cmpds_0,no_cmpds = ClassEval.computeAllClassMetrics(test_y=subset_test_y,predicted_y=subset_predicted_y,probs_for_class_1=subset_probs_for_class_1,method=method,subset_name='{}_{}'.format(AD_method_name,AD_subset),output_dir=calc_dir,delta_for_calib_plot=lit_precedent_delta_for_calib_plot)
                else:
                
                    precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal = [None]*17
                    no_cmpds_1,no_cmpds_0,no_cmpds = [0]*3
                
                populateDictWithStats(dict_of_stats,AD_method_name,method,AD_subset,precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal,no_cmpds_1,no_cmpds_0,no_cmpds)
            
                
                
    statsFilesDict = writeStatsFiles(dict_of_stats,calc_dir,selected_ad_methods,selected_uncertainty_methods)
    
    
    print(f'RAN modellingEvalAnalysisForOneModelVersion(...) for update_version={update_version},random seed={rand_seed}')
    
    return {'statsFilesDict':statsFilesDict,'dict_of_raw_results':dict_of_raw_results}

def main():
    print('THE START')
    
    allStatsFilesDicts = defaultdict(dict)
    
    allRawResDicts = defaultdict(dict)
    
    createOrReplaceDir(dir_=top_calc_dir)
    
    
    for update_version in updates_in_order:

        for global_random_seed in all_global_random_seed_opts:
        
            ret_val = modellingEvalAnalysisForOneModelVersion(update_version,selected_uncertainty_methods,selected_ad_methods,rand_seed=global_random_seed)
        
            if 0 == ret_val:
                pass
            else:
                allStatsFilesDicts[update_version][f's={global_random_seed}'] = ret_val['statsFilesDict']
                
                allRawResDicts[update_version][f's={global_random_seed}'] = ret_val['dict_of_raw_results']
    
    print('Writing all statistics to {}'.format(all_stats_pkl))
    
    f_o = open(all_stats_pkl,'wb')
    try:
        pickle.dump(allStatsFilesDicts,f_o)
    finally:
        f_o.close()
        del f_o
    
    print('Writing all raw results to {}'.format(all_raw_pkl))
    
    f_o = open(all_raw_pkl,'wb')
    try:
        pickle.dump(allRawResDicts,f_o)
    finally:
        f_o.close()
        del f_o
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
