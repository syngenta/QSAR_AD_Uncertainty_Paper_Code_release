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
#Contact zied.hosni [at] syngenta.com
##############################

#====================================
import os,sys,re,copy
from collections import defaultdict
import pandas as pd,numpy as np
from pandas.testing import assert_frame_equal,assert_series_equal
from sklearn.ensemble import RandomForestRegressor
from pytest import approx
from modelsADuncertaintyPkg.qsar_eval.reg_perf_pred_stats import SpearmanCoeff
#===================================
this_dir = os.path.dirname(os.path.abspath(__file__))
top_dir=os.path.dirname(this_dir)
#===================================
#Import from the proto-package for testing:
sys.path.append(top_dir)

from modelsADuncertaintyPkg.utils.basic_utils import check_findDups,check_convertDefaultDictDictIntoDataFrame,check_flatten,check_isInteger,check_getKeyFromValue,neverEndingDefaultDict,check_geometricMean,returnDefDictOfLists,doubleDefaultDictOfLists,check_no_missing_values,insert_list_values_into_list_of_lists,add_elements_to_start_of_list_of_lists,get_pandas_df_row_as_df
from modelsADuncertaintyPkg.CheminformaticsUtils import chem_data_parsing_utils as ChemDataParser
from modelsADuncertaintyPkg.CheminformaticsUtils.standardize_mols import standardize
from modelsADuncertaintyPkg.qsar_AD.SingleTcDist1NNAD import coreTanimotoSplitting
from modelsADuncertaintyPkg.qsar_AD import UNC_like_adapted_AD_approach as UNC_like_AD
from modelsADuncertaintyPkg.qsar_AD import dk_NN_thresholds as dk_NN_AD
from modelsADuncertaintyPkg.qsar_AD import rdn_update_dk_NN_thresholds as rdn_AD
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import findInsideOutsideADTestIds,getInsideOutsideADSubsets,getInputRequiredForModellingAndAD
from modelsADuncertaintyPkg.qsar_eval.class_prob_perf_stats import check_computeVariantBrierLossForKclasses,check_computeOriginalBrierLossForKclasses,check_computeLogLoss,check_compute_Stratified_ProbabilisticLossForKclasses,check_computePairwiseAverageAuc
from modelsADuncertaintyPkg.qsar_eval.class_pred_perf_stats import check_weighted_kappa,check_kappa
from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import consistentlyDiscretizeEndpointValues,define_no_bins_for_regression_discretization
from modelsADuncertaintyPkg.utils.average_all_bioact_vals_for_same_mol_after_exclude_outliers import check_getMeanSd,check_getAveragePlusDeviationEstimateForBioactAfterRemovingOutliers
from modelsADuncertaintyPkg.utils.ML_utils import getTestYFromTestIds
from modelsADuncertaintyPkg.CheminformaticsUtils.similarity import compute_tanimoto_similarity
from modelsADuncertaintyPkg.qsar_reg_uncertainty.combine_pred_intervals import aggregateIntervals
from modelsADuncertaintyPkg.qsar_reg_uncertainty import funcsToApplyRegUncertaintyToNewDatasets as RegUncert
from modelsADuncertaintyPkg.utils.ML_utils import get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance
from modelsADuncertaintyPkg.qsar_reg_uncertainty.RegressionICP import compute_y_std
from modelsADuncertaintyPkg.qsar_eval.enforce_minimum_no_instances import get_no_instances
from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import computeAllRegMetrics
from modelsADuncertaintyPkg.utils import check_stratified_continuous_split
from modelsADuncertaintyPkg.utils.basic_utils import report_name_of_function_where_this_is_called
from modelsADuncertaintyPkg.utils.time_utils import times_of_repeated_many_function_calls
from modelsADuncertaintyPkg.qsar_reg_uncertainty.RF_prediction_intervals import pred_ints
from modelsADuncertaintyPkg.utils.ML_utils import compute_pred_intervals
from modelsADuncertaintyPkg.utils.basic_utils import numpy_array_contains_zeros
from modelsADuncertaintyPkg.qsar_eval.reg_Uncertainty_metrics import compute_ENCE, get_precursor_info_for_ENCE, organize_variables_as_required_for_ENCE,bin_data_for_ENCE
from modelsADuncertaintyPkg.qsar_eval.all_key_class_stats_and_plots import get_experi_class_1_probs
#=====================================

class TestRunner():
    
    def get_Train_Test_for_Checking_AD_Method_Correctly_Flags_Aliphatics_Out_Aromatics_In_For_Aromatics_Training_Set(self):
        train_set_smiles = ['c1ccccc1','c1ccccc1N','c1ccccc1O','c1cccc(C)c1N','c1ccccc1N(C)C','c1ccccc1NF']
        
        expected_testId2SmilesAndStatus = defaultdict(dict)
        
        expected_testId2SmilesAndStatus[1]['SMILES'] = 'c1ccccc1'
        expected_testId2SmilesAndStatus[1]['InsideAD'] = True
        
        expected_testId2SmilesAndStatus[2]['SMILES'] = 'c1ccccc1N'
        expected_testId2SmilesAndStatus[2]['InsideAD'] = True
        
        expected_testId2SmilesAndStatus[3]['SMILES'] = 'c1(C)ccccc1N'
        expected_testId2SmilesAndStatus[3]['InsideAD'] = True
        
        expected_testId2SmilesAndStatus[4]['SMILES'] = 'CCO'
        expected_testId2SmilesAndStatus[4]['InsideAD'] = False
        
        expected_testId2SmilesAndStatus[5]['SMILES'] = 'C=C'#'CCN' #This was inside the AD for a very high Tanimoto distance threshold (0.9)
        expected_testId2SmilesAndStatus[5]['InsideAD'] = False
        
        return train_set_smiles,expected_testId2SmilesAndStatus
    
    def get_Details_Required_for_Checking_AD_Method_Correctly_Flags_Aliphatics_Out_Aromatics_In_For_Aromatics_Training_Set(self,only_need_FPs_not_bit_vectors=True,smiles_col='SMILES',id_col='ID',regression=False):
        
        train_set_smiles,expected_testId2SmilesAndStatus = self.get_Train_Test_for_Checking_AD_Method_Correctly_Flags_Aliphatics_Out_Aromatics_In_For_Aromatics_Training_Set()
        
        fps_train = pd.DataFrame([ChemDataParser.getFPFromStandardisedMol(tr_smiles) for tr_smiles in train_set_smiles],columns=['fp']) #c.f. def TanimotoSplitting(...)
        
        test_ids_in_order = [id_ for id_ in expected_testId2SmilesAndStatus.keys()]
        
        fps_test = pd.DataFrame([ChemDataParser.getFPFromStandardisedMol(expected_testId2SmilesAndStatus[id_][smiles_col]) for id_ in test_ids_in_order],columns=['fp'])
        
        if not only_need_FPs_not_bit_vectors:
            #=================
            #These may need updating, as may size of training set if we hope to check RDN using this function (see def get_Train_Test_for_Checking_AD_Method_Correctly_Flags_Aliphatics_Out_Aromatics_In_For_Aromatics_Training_Set)
            if regression:
                print(f'Sanity checking AD method for a regression training set!')
                train_y = [(v-0.25) for v in range(0,len(train_set_smiles))]
            else:
                print(f'Sanity checking AD method for a classification training set!')
                train_y = ['Active','Inactive']*3
            
            assert len(train_y) == len(train_set_smiles),"len(endpoint_vals) = {} len(train_set_smiles) = {}".format(len(train_y),len(train_set_smiles))
            
            train_y = pd.Series(train_y)
            #=================
            
            train_smiles_df = pd.DataFrame({id_col:list(range(0,len(train_set_smiles))),smiles_col:train_set_smiles})#,endpoint_col:endpoint_vals})
            
            train_df_with_fps_too = ChemDataParser.addFPBitsColsToDfWithSMILES(df_with_smiles=train_smiles_df,smiles_col=smiles_col)
            
            X_train_and_ids_df = train_df_with_fps_too.drop([smiles_col],axis=1)
            
            test_smiles_df = pd.DataFrame({id_col:test_ids_in_order,smiles_col:[expected_testId2SmilesAndStatus[id_]['SMILES'] for id_ in test_ids_in_order]})
            
            test_df_with_fps_too = ChemDataParser.addFPBitsColsToDfWithSMILES(df_with_smiles=test_smiles_df,smiles_col=smiles_col)
            
            X_test_and_ids_df = test_df_with_fps_too.drop([smiles_col],axis=1)
        else:
            X_train_and_ids_df = None
            
            X_test_and_ids_df = None
            
            train_y = None
        
        return fps_train,fps_test,test_ids_in_order,X_train_and_ids_df,X_test_and_ids_df,train_y,expected_testId2SmilesAndStatus
    
    def test_Tanimoto_AD_Method_Correctly_Flags_Aliphatics_Out_Aromatics_In_For_Aromatics_Training_Set(self):
        
        
        fps_train,fps_test,test_ids_in_order,X_train_and_ids_df,X_test_and_ids_df,train_y,expected_testId2SmilesAndStatus = self.get_Details_Required_for_Checking_AD_Method_Correctly_Flags_Aliphatics_Out_Aromatics_In_For_Aromatics_Training_Set(only_need_FPs_not_bit_vectors=True)
        
        for threshold_percentile in range(20,100,10):
            threshold = threshold_percentile/100
            
            print('-'*50)
            print('Checking coreTanimotoSplitting(...) using distance threshold={}'.format(threshold))
            test_id_status_dict = coreTanimotoSplitting(fps_train, fps_test, threshold,test_ids_in_same_order_as_fps_test=pd.Series(test_ids_in_order))
            
            for id_ in test_ids_in_order:
                print('Checking for id={} ...'.format(id_))
                
                assert expected_testId2SmilesAndStatus[id_]['InsideAD'] == test_id_status_dict[id_]['InsideAD'],"distance threshold={},id_={}, expected_testId2SmilesAndStatus[id_]['InsideAD'] ={}, test_id_status_dict[id_]['InsideAD']={}".format(threshold,id_,expected_testId2SmilesAndStatus[id_]['InsideAD'],test_id_status_dict[id_]['InsideAD'])
            
            print('-'*50)
        
    def test_ChemDataParsingUtils_All_FP_Checks_with_SMILES_inputs(self):
        
        
        ChemDataParser.check_getFPFromStandardisedMol_BitsList_SimpleMol()
        
        ChemDataParser.check_getFPFromStandardisedMol_BitsList_SimpleMolSalt()
        
        ChemDataParser.check_getFPFromStandardisedMol_BitsList_ProblemMol()
        
        ChemDataParser.check_addFPBitsColsToDfWithSMILES()
        
        ChemDataParser.check_addFPBitsColsToDfWithProblemSMILES()
    
    def test_ChemDataParsingUtils_InChI_Checks(self):
        
        
        ChemDataParser.check_getInChIFromStandardisedMol_simple_smiles_salt()
    
    def test_UNC_like_AD_approach(self):
        
        
        UNC_like_AD.check_UNC_like_AD_approach()
    
    def test_dk_NN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations(self):
        
        
        dk_NN_AD.check_dk_NN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations()
    
    def test_rdn_thresholds_only_consider_k_neighbours_for_final_threshold_calculations(self):
        
        
        rdn_AD.check_RDN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations()
    
    def test_rdn_thresholds_only_consider_k_neighbours_for_final_threshold_calculations_regression(self):
        
        
        rdn_AD.check_RDN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations_regression()
    
    def test_findDups(self): 
        
        check_findDups()
    
    def test_convertDefaultDictDictIntoDataFrame(self):
        
        check_convertDefaultDictDictIntoDataFrame()
    
    def test_flatten(self):
        
        check_flatten()
    
    def test_getInsideOutsideADSubsets(self):
        
        
        #-------------------------------
        test_id2ADStatus = defaultdict(dict)
        
        test_id2ADStatus['M1']['InsideAD'] = True
        test_id2ADStatus['M2']['InsideAD'] = False
        test_id2ADStatus['M3']['InsideAD'] = True
        #--------------------------------
        
        id_col = 'ID'
        
        #--------------------------------
        test_set_df = pd.DataFrame({id_col:['M1','M2','M3'],'x1':[1,0,0]})
        #--------------------------------
        
        expected_test_insideAD_df = pd.DataFrame({id_col:['M1','M3'],'x1':[1,0]})
        
        expected_test_outsideAD_df = pd.DataFrame({id_col:['M2'],'x1':[0]})
        #--------------------------------
        
        test_insideAD_df, test_outsideAD_df = getInsideOutsideADSubsets(test_set_df, id_col, test_id2ADStatus)
        
        #----------------------------------
        test_insideAD_df.reset_index(drop=True, inplace=True)
        
        test_outsideAD_df.reset_index(drop=True, inplace=True)
        
        #----------------------------------
        assert_frame_equal(expected_test_insideAD_df,test_insideAD_df)
        
        assert_frame_equal(expected_test_outsideAD_df,test_outsideAD_df)
        
    def check_findInsideOutsideADTestIds_Generic(self,AD_method_name,threshold,non_default_expected_testId2SmilesAndStatus=None,regression=False):
        
        fps_train,fps_test,test_ids_in_order,X_train_and_ids_df,X_test_and_ids_df,train_y,expected_testId2SmilesAndStatus = self.get_Details_Required_for_Checking_AD_Method_Correctly_Flags_Aliphatics_Out_Aromatics_In_For_Aromatics_Training_Set(only_need_FPs_not_bit_vectors=False,regression=regression)
        
        #=============================
        if not non_default_expected_testId2SmilesAndStatus is None:
            expected_testId2SmilesAndStatus = non_default_expected_testId2SmilesAndStatus
        #=============================
        
        id_col = 'ID'
        
        #==============================
        #if not 'RDN' == AD_method_name:
        #    regression = False
        #else:
        #    regression = True
        #==============================
        
        test_id_status_dict = findInsideOutsideADTestIds(X_train=X_train_and_ids_df, X_test=X_test_and_ids_df, fps_train=fps_train, fps_test=fps_test, threshold=threshold, id_col=id_col,rand_seed=42, endpoint_col='Endpoint', AD_method_name=AD_method_name,test_ids=test_ids_in_order, y_train=train_y, y_test=None, regression=regression,class_1_label='Active')
        
        
        
        #======================================
        for id_ in test_ids_in_order:
            print('Checking for id={} ...'.format(id_))
            
            try:
                assert expected_testId2SmilesAndStatus[id_]['InsideAD'] == test_id_status_dict[id_]['InsideAD'],"threshold (AD method specific meaning)={},id_={}, expected_testId2SmilesAndStatus[id_]['InsideAD'] ={}, test_id_status_dict[id_]['InsideAD']={}".format(threshold,id_,expected_testId2SmilesAndStatus[id_]['InsideAD'],test_id_status_dict[id_]['InsideAD'])
            except KeyError as err:
                #Debugging:
                print('test_id_status_dict = {}'.format(test_id_status_dict))
                print('X_test_and_ids_df = {}'.format(X_test_and_ids_df))
                print('X_test_and_ids_df.astype(int) = {}'.format(X_test_and_ids_df.astype(int)))
                print('X_test_and_ids_df[id_col].tolist() = {}'.format(X_test_and_ids_df[id_col].tolist()))
                print('X_test_and_ids_df[id_col].astype(int).tolist() = {}'.format(X_test_and_ids_df[id_col].astype(int).tolist()))
                raise err
        #======================================
        
    def test_findInsideOutsideADTestIds_Tanimoto(self):
        
        
        self.check_findInsideOutsideADTestIds_Generic(AD_method_name='Tanimoto',threshold=0.35,non_default_expected_testId2SmilesAndStatus=None)
    
    def test_findInsideOutsideADTestIds_UNC(self):
        
        
        k_NN = 5
        
        #-------------------------------------------
        non_default_expected_testId2SmilesAndStatus = None
        # non_default_expected_testId2SmilesAndStatus = defaultdict(dict) #copied and pasted, then edited - where indicated - from above #could have tried copy.deepcopy(...) then edit as well, but more code re-organization would be required!
        
        # non_default_expected_testId2SmilesAndStatus[1]['SMILES'] = 'c1ccccc1'
        # non_default_expected_testId2SmilesAndStatus[1]['InsideAD'] = True
        
        # non_default_expected_testId2SmilesAndStatus[2]['SMILES'] = 'c1ccccc1N'
        # non_default_expected_testId2SmilesAndStatus[2]['InsideAD'] = True
        
        # non_default_expected_testId2SmilesAndStatus[3]['SMILES'] = 'c1(C)ccccc1N'
        # non_default_expected_testId2SmilesAndStatus[3]['InsideAD'] = False #Actually, this appears to mask another error due to inconsistent IDs - but keeping uncommented for now for debuggin! #non-default (changed from above): it is OK for this method to be more stringent than simple Tanimoto method!
        
        # non_default_expected_testId2SmilesAndStatus[4]['SMILES'] = 'CCO'
        # non_default_expected_testId2SmilesAndStatus[4]['InsideAD'] = False
        
        # non_default_expected_testId2SmilesAndStatus[5]['SMILES'] = 'C=C'
        # non_default_expected_testId2SmilesAndStatus[5]['InsideAD'] = False
        # #-------------------------------------------------------
        
        self.check_findInsideOutsideADTestIds_Generic(AD_method_name='UNC',threshold=k_NN,non_default_expected_testId2SmilesAndStatus=non_default_expected_testId2SmilesAndStatus)
    
    def test_findInsideOutsideADTestIds_dkNN(self):
        
        
        k_NN = 5
        
        #-------------------------------------------
        non_default_expected_testId2SmilesAndStatus = None
        # non_default_expected_testId2SmilesAndStatus = defaultdict(dict) #copied and pasted, then edited - where indicated - from above #could have tried copy.deepcopy(...) then edit as well, but more code re-organization would be required!
        
        # non_default_expected_testId2SmilesAndStatus[1]['SMILES'] = 'c1ccccc1'
        # non_default_expected_testId2SmilesAndStatus[1]['InsideAD'] = True
        
        # non_default_expected_testId2SmilesAndStatus[2]['SMILES'] = 'c1ccccc1N'
        # non_default_expected_testId2SmilesAndStatus[2]['InsideAD'] = True
        
        # non_default_expected_testId2SmilesAndStatus[3]['SMILES'] = 'c1(C)ccccc1N'
        # non_default_expected_testId2SmilesAndStatus[3]['InsideAD'] = False #Actually, this appears to mask another error due to inconsistent IDs - but keeping uncommented for now for debuggin! #non-default (changed from above): it is OK for this method to be more stringent than simple Tanimoto method!
        
        # non_default_expected_testId2SmilesAndStatus[4]['SMILES'] = 'CCO'
        # non_default_expected_testId2SmilesAndStatus[4]['InsideAD'] = False
        
        # non_default_expected_testId2SmilesAndStatus[5]['SMILES'] = 'C=C'
        # non_default_expected_testId2SmilesAndStatus[5]['InsideAD'] = False
        # #-------------------------------------------------------
        
        self.check_findInsideOutsideADTestIds_Generic(AD_method_name='dkNN',threshold=k_NN,non_default_expected_testId2SmilesAndStatus=non_default_expected_testId2SmilesAndStatus)
    
    def test_findInsideOutsideADTestIds_RDN(self):
        
        
        k_NN = 5
        
        #-------------------------------------------
        non_default_expected_testId2SmilesAndStatus = None
        # non_default_expected_testId2SmilesAndStatus = defaultdict(dict) #copied and pasted, then edited - where indicated - from above #could have tried copy.deepcopy(...) then edit as well, but more code re-organization would be required!
        
        # non_default_expected_testId2SmilesAndStatus[1]['SMILES'] = 'c1ccccc1'
        # non_default_expected_testId2SmilesAndStatus[1]['InsideAD'] = True
        
        # non_default_expected_testId2SmilesAndStatus[2]['SMILES'] = 'c1ccccc1N'
        # non_default_expected_testId2SmilesAndStatus[2]['InsideAD'] = True
        
        # non_default_expected_testId2SmilesAndStatus[3]['SMILES'] = 'c1(C)ccccc1N'
        # non_default_expected_testId2SmilesAndStatus[3]['InsideAD'] = False #Actually, this appears to mask another error due to inconsistent IDs - but keeping uncommented for now for debuggin! #non-default (changed from above): it is OK for this method to be more stringent than simple Tanimoto method!
        
        # non_default_expected_testId2SmilesAndStatus[4]['SMILES'] = 'CCO'
        # non_default_expected_testId2SmilesAndStatus[4]['InsideAD'] = False
        
        # non_default_expected_testId2SmilesAndStatus[5]['SMILES'] = 'C=C'
        # non_default_expected_testId2SmilesAndStatus[5]['InsideAD'] = False
        # #-------------------------------------------------------
        
        self.check_findInsideOutsideADTestIds_Generic(AD_method_name='RDN',threshold=k_NN,non_default_expected_testId2SmilesAndStatus=non_default_expected_testId2SmilesAndStatus)
    
    def test_findInsideOutsideADTestIds_RDN_regression(self):
        
        
        k_NN = 5
        
        #-------------------------------------------
        non_default_expected_testId2SmilesAndStatus = None
        #---------------------------------------------

        
        self.check_findInsideOutsideADTestIds_Generic(AD_method_name='RDN',threshold=k_NN,non_default_expected_testId2SmilesAndStatus=non_default_expected_testId2SmilesAndStatus,regression=True)
    
    def test_Qn(self):
        
        
        dk_NN_AD.check_Qn()
    
    def test_isInteger(self):
        
        
        check_isInteger()
    
    def test_getKeyFromValue(self):
        
        
        
        check_getKeyFromValue()
    
    def test_computeVariantBrierLossForKclasses(self):
        
        
        
        
        check_computeVariantBrierLossForKclasses()
    
    def test_computeOriginalBrierLossForKclasses(self):
        
        
        
        check_computeOriginalBrierLossForKclasses()
        
    def test_computeLogLoss(self):
        
        
        
        check_computeLogLoss()
    
    def test_compute_Stratified_ProbabilisticLossForKclasses(self):
        
        
        
        check_compute_Stratified_ProbabilisticLossForKclasses()
    
    def test_computeStratifiedBrier_TwoCategories_only_one_experimental_class(self):
        from modelsADuncertaintyPkg.qsar_eval.all_key_class_stats_and_plots import computeStratifiedBrier_TwoCategories

        class_1_probs_in_order = [0.9,0.3,0.8]
        experi_class_labels = [1,1,1]

        res = computeStratifiedBrier_TwoCategories(class_1_probs_in_order,experi_class_labels)

        assert pd.isna(res),res

    def test_computeStratifiedBrier_TwoCategories_fails_with_unexpected_class(self):
        from modelsADuncertaintyPkg.qsar_eval.all_key_class_stats_and_plots import computeStratifiedBrier_TwoCategories

        fails = False

        class_1_probs_in_order = [0.9,0.3,0.8]
        experi_class_labels = [1,1,2]

        try:
            res = computeStratifiedBrier_TwoCategories(class_1_probs_in_order,experi_class_labels)
        except Exception:
            fails = True
        
        assert fails

    
    def test_computePairwiseAverageAuc(self):
        
        
        
        check_computePairwiseAverageAuc()
    
    def test_weighted_kappa(self):
        
        
        
        check_weighted_kappa()
    
    def test_kappa(self):
        
        
        
        check_kappa()
    
    def test_neverEndingDefaultDict(self):
        
        
        
        ddi = neverEndingDefaultDict()
        
        #ddi['a']['b']['c'] = 5 #Using this causes the following error to occur at the next line: "TypeError: 'int' object does not support item assignment"
        ddi['a']['b']['c']['d'] = 500
    
    def test_geometricMean(self):
        
        
        
        check_geometricMean()
    
    def test_getTestYFromTestIds_series_indices_not_default(self):
        
        
        
        dataset_ids = [1,2,3,4,5]
        
        test_ids = [2,4,5]
        
        dataset_y = pd.Series([1.5,1.8,-0.9,0.9,0.68,2.5])[1:6]
        
        expected_test_y = pd.Series([-0.9,0.68,2.5])
        
        test_y = getTestYFromTestIds(dataset_ids,dataset_y,test_ids)
        
        assert expected_test_y.equals(test_y),"test_y={}".format(test_y)
    
    def test_getTestYFromTestIds_series_indices_re_arranged(self):
        
        
        
        dataset_ids = [1,2,3,4,5]
        
        test_ids = [2,4,5]
        
        dataset_y = pd.Series([1.5,1.8,-0.9,0.9,0.68,2.5])[1:6]

        dataset_y.index=[5,2,3,4,1]
        
        expected_test_y = pd.Series([-0.9,0.68,2.5])
        
        test_y = getTestYFromTestIds(dataset_ids,dataset_y,test_ids)
        
        assert expected_test_y.equals(test_y),"test_y={}".format(test_y)
    
    def test_getTestYFromTestIds_numpy_2d_array(self):
        
        dataset_ids=[2,30,3,1]
        test_ids=[30,1]
        
        nrTestCases=4
        intervals = np.zeros((nrTestCases,  2))
        for k in range(0,nrTestCases):
            intervals[k,0] = -k
            intervals[k,1] = k
        
        dataset_y=intervals
        
        expected_test_y = np.array([[-1.,  1.],[-3.,  3.]])
        
        test_y = getTestYFromTestIds(dataset_ids,dataset_y,test_ids)
        
        assert np.array_equal(expected_test_y,test_y),"test_y={}".format(test_y)
    
    def test_assert_works(self):
        #Python can be run with a flag which means assert statements are ignored! Check this isn't the case!
        
        
        try:
            assert 1==2
            raise Exception('This assert statement should have failed! Otherwise, all assert statements need replacing with if ....: raise Exception(...)!')
        except AssertionError as err:
            pass
    
    def test_getAveragePlusDeviationEstimateForBioactAfterRemovingOutliers(self):
        
        
        check_getAveragePlusDeviationEstimateForBioactAfterRemovingOutliers()
    
    def test_getMeanSd(self):
        
        
        check_getMeanSd()
    
    def test_removeDuplicateMolsTakingAccountOfActivities(self):
        
        
        ChemDataParser.check_removeDuplicateMolsTakingAccountOfActivities()
    
    def test_getInChIFromMol(self):
        
        
        ChemDataParser.check_getInChIFromMol()
    
    def test_consistentlyDiscretizeEndpointValues(self):
        
        
        examples_dict = defaultdict(dict)
        
        #---------------------------
        examples_dict[1]['y_set_two_list'] = [-19.458093636448474, -16.31871112199646, -12.552059409315117, -10.54941385468599, -9.155664497936975, -8.640219860095506, -8.052181220813377, -3.865585892480138, -2.484520304963123, 0.18809399784200753, 0.7299287564658243, 4.467204688213438, 4.56389077248358, 6.964780125808418, 11.056090568429347, 12.398423109488188, 13.248245903357773, 14.136629812813975, 29.297235447750822, 34.73594131857662]
        
        examples_dict[1]['y_set_one_list'] = [-12.262465184828052, -3.3635781991822906, -2.732455189245883, -0.5769849432117597, 0.9788448564114522, 1.1367136974286423, 4.429012647802734, 5.9832260159059185, 6.917764085416956, 15.954900037736373]
        
        examples_dict[1]['expected_y_set_one_discretized'] = [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0]
        
        examples_dict[1]['expected_y_set_two_discretized'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0] #It appears that the first and last bin edges are redundant?
        
        #---------------------------
        
        for eg in examples_dict.keys():
            print('Checking example {}'.format(eg))
            
            y_set_one_discretized,y_set_two_discretized,discretizer = consistentlyDiscretizeEndpointValues(examples_dict[eg]['y_set_one_list'],examples_dict[eg]['y_set_two_list'])
            
            if not examples_dict[eg]['expected_y_set_one_discretized'] == y_set_one_discretized: raise Exception('example {}: expected_y_set_one_discretized={} vs. y_set_one_discretized={} [discretizer.bin_edges_ = {}]'.format(eg,examples_dict[eg]['expected_y_set_one_discretized'],y_set_one_discretized,discretizer.bin_edges_))
            
            if not examples_dict[eg]['expected_y_set_two_discretized'] == y_set_two_discretized: raise Exception('example {}: expected_y_set_two_discretized={} vs. y_set_two_discretized={} [discretizer.bin_edges_ = {}]'.format(eg,examples_dict[eg]['expected_y_set_two_discretized'],y_set_two_discretized,discretizer.bin_edges_))
            
            print('CHECKED example {}'.format(eg))
    
    def test_consistentlyDiscretizeEndpointValues_with_non_diverse_set(self):
        examples_dict = defaultdict(dict)
        
        #---------------------------
        examples_dict[1]['y_set_two_list'] = [-19.458093636448474, -16.31871112199646, -12.552059409315117, -10.54941385468599, -9.155664497936975, -8.640219860095506, -8.052181220813377, -3.865585892480138, -2.484520304963123, 0.18809399784200753, 0.7299287564658243, 4.467204688213438, 4.56389077248358, 6.964780125808418, 11.056090568429347, 12.398423109488188, 13.248245903357773, 14.136629812813975, 29.297235447750822, 34.73594131857662]
        
        examples_dict[1]['y_set_one_list'] = [-19.458093636448474]*3 #[discretizer.bin_edges_ = [array([-inf,  inf])]]
        
        examples_dict[1]['expected_y_set_one_discretized'] = [0.0]*3
        
        examples_dict[1]['expected_y_set_two_discretized'] = [0.0]*(len(examples_dict[1]['y_set_two_list']))
        
        #---------------------------
        
        for eg in examples_dict.keys():
            print('Checking example {}'.format(eg))

            ####################
            #c.f. function calls inside assess_stat_sig_shif_metrics.py:def getReadyForStratRandSplit(dict_of_raw_subset_results,strat_rand_split_y_name,type_of_modelling,subset_1_name,subset_2_name,default_no_bins_for_reg_strat=5):
            default_no_bins_for_reg_strat=5
            
            computed_n_bins = define_no_bins_for_regression_discretization(examples_dict[eg]['y_set_one_list'],default=default_no_bins_for_reg_strat)

            y_set_one_discretized,y_set_two_discretized,discretizer = consistentlyDiscretizeEndpointValues(examples_dict[eg]['y_set_one_list'],examples_dict[eg]['y_set_two_list'],n_bins=computed_n_bins)
            #####################

            if not examples_dict[eg]['expected_y_set_one_discretized'] == y_set_one_discretized: raise Exception('example {}: expected_y_set_one_discretized={} vs. y_set_one_discretized={} [discretizer.bin_edges_ = {}]'.format(eg,examples_dict[eg]['expected_y_set_one_discretized'],y_set_one_discretized,discretizer.bin_edges_))
            
            if not examples_dict[eg]['expected_y_set_two_discretized'] == y_set_two_discretized: raise Exception('example {}: expected_y_set_two_discretized={} vs. y_set_two_discretized={} [discretizer.bin_edges_ = {}]'.format(eg,examples_dict[eg]['expected_y_set_two_discretized'],y_set_two_discretized,discretizer.bin_edges_))
            
            print('CHECKED example {}'.format(eg))

    
    def test_doubleDefaultDictOfLists(self):
        
        
        d = doubleDefaultDictOfLists()
        
        d['a']['b'].append(2)
        d['a']['b'].append(1)
        
        if not [2,1] == d['a']['b']: raise Exception("d['a']['b']={}".format(d['a']['b']))
        
        #d['x']['y']['z'].append(0)
        
        #if not [0] == d['x']['y']['z']: raise Exception("d['x']['y']['z']={}".format(d['x']['y']['z']))
    
    def test_standardize(self):
        from rdkit.Chem import MolToSmiles
        
        
        examples_dict = defaultdict(dict)
        
        examples_dict[1]['input'] = 'CCC[O-].[Na+]'
        examples_dict[1]['expected_output'] = 'CCCO'
        
        #################################
        #Based upon Fig. 5 of the publication cited in the RDKit documentation:
        #https://link.springer.com/article/10.1007/s10822-010-9346-4
        examples_dict[2]['input'] = 'CC(O)=CC=C' 
        examples_dict[2]['expected_output'] = 'CC=CC(C)=O'
        ################################
        
        examples_dict[3]['input'] = '[CH3][CH2][OH]'
        examples_dict[3]['expected_output'] = 'CCO'
        
        examples_dict[4]['input'] = 'CCCO[Na]'
        examples_dict[4]['expected_output'] = 'CCCO'
        
        examples_dict[5]['input'] = '[CH3]CO[Fe]'
        examples_dict[5]['expected_output'] = 'CCO'
        
        for eg in examples_dict.keys():
            
            output_mol = standardize(examples_dict[eg]['input'])
            
            #----------------------------
            if not examples_dict[eg]['expected_output'] is None:
                output = MolToSmiles(output_mol)
            else:
                output = output_mol
            #----------------------------
            
            assert output == examples_dict[eg]['expected_output'],f"input = {examples_dict[eg]['input']},expected_output={examples_dict[eg]['expected_output']},output={output}"
            
            print(f'Example = {eg}, standardize(...) output (converted to SMILES) = {output}')
    
    def check_consistency_of_fps_df_and_X_and_ids_df(self,fps_df,fp_col,X_and_ids_df,id_col):
        
        X_df = X_and_ids_df.drop(id_col,axis=1)
        
        fps_df_as_bit_vector_lol = [ChemDataParser.convert_bit_vector_fingerprint_to_bits_list(fp) for fp in fps_df[fp_col].tolist()]
        assert fps_df_as_bit_vector_lol == X_df.values.tolist(),f"fps_df_as_bit_vector_lol = {fps_df_as_bit_vector_lol} vs. X_df.values.tolist()={X_df.values.tolist}"
    
    def test_insert_list_values_into_list_of_lists(self):
        res = insert_list_values_into_list_of_lists(lol=[[0,0,1],[1,0,1],[1,1,1]],my_list=[2,6,4])
        
        assert res == [[2,0,0,1],[6,1,0,1],[4,1,1,1]],f"res={res}"
    
    def test_add_elements_to_start_of_list_of_lists(self):
        res = add_elements_to_start_of_list_of_lists(lol=[[0,0,1],[1,0,1],[1,1,1]],my_list=[2,6,4])
        
        assert res == [[2,0,0,1],[6,1,0,1],[4,1,1,1]],f"res={res}"
    
    def get_input_for_testing_getInputRequiredForModellingAndAD(self):
        
        id_col = 'ID'
        smiles_col = 'SMILES'
        class_label_col = 'Actvity'
        class_1 = None #This argument is never actually used - here.
        fp_col = 'FingerPrint'
        
        train_df = pd.DataFrame({id_col:[2,6,4],smiles_col:[ChemDataParser.simple_smiles,ChemDataParser.complex_smiles,ChemDataParser.simple_smiles_salt],class_label_col:[1.78,3.5,5.8]})
        
        test_df = pd.DataFrame({id_col:[7,8,9],smiles_col:[ChemDataParser.complex_smiles,'?',ChemDataParser.simple_smiles_salt],class_label_col:[0,1,0]}) #Introduce a deliberately invalid SMILES string to check handling of this.
        
        return id_col,smiles_col,class_label_col,class_1,fp_col,train_df,test_df
    
    def run_all_check_on_output_of_getInputRequiredForModellingAndAD(self,id_col,smiles_col,class_label_col,class_1,fp_col,train_df,test_df,fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids,pre_calc_fps=False):
        
        assert fps_train.shape[0]==train_df.shape[0],f"fps_train.shape[0]={fps_train.shape[0]},train_df.shape[0]={train_df.shape[0]}"
        if not pre_calc_fps:
            assert fps_test.shape[0]==(test_df.shape[0]-1),f"fps_test.shape[0]={fps_test.shape[0]},test_df.shape[0]={test_df.shape[0]}"
        else:
            #Here, we have already removed the row for which FPs could not be computed from the input train_df and test_df:
            assert fps_test.shape[0]==test_df.shape[0],f"fps_test.shape[0]={fps_test.shape[0]},test_df.shape[0]={test_df.shape[0]}"
        
        assert train_y.tolist() == [1.78,3.5,5.8],f"train_y={train_y}"
        assert test_y.tolist() == [0,0],f"test_y={test_y}"
        assert train_ids == [2,6,4],f"train_ids={train_ids}"
        assert test_ids == [7,9],f"test_ids={test_ids}"
        
        self.check_consistency_of_fps_df_and_X_and_ids_df(fps_df=fps_train,fp_col=fp_col,X_and_ids_df=X_train_and_ids_df,id_col=id_col)
        self.check_consistency_of_fps_df_and_X_and_ids_df(fps_df=fps_test,fp_col=fp_col,X_and_ids_df=X_test_and_ids_df,id_col=id_col)
        
        #----------------------------------
        
        expected_X_train_and_ids_lol = add_elements_to_start_of_list_of_lists(lol=[ChemDataParser.expected_fp_bits_list_of_simple_mol,ChemDataParser.expected_fp_bits_list_of_complex_mol,ChemDataParser.expected_fp_bits_list_of_simple_mol],my_list=[2,6,4])
        
        assert all([1025 == len(l) for l in expected_X_train_and_ids_lol]),f"len(expected_X_train_and_ids_lol[0])={len(expected_X_train_and_ids_lol[0])}" #Before we switched from insert_list_values_into_list_of_lists(...) to add_elements_to_start_of_list_of_lists(....): pytest: all([False, True, False])
        assert [2,6,4] == [l[0] for l in expected_X_train_and_ids_lol],f"[l[0] for l in expected_X_train_and_ids_lol]={[l[0] for l in expected_X_train_and_ids_lol]}"
        
        expected_X_train_and_ids_df=pd.DataFrame(expected_X_train_and_ids_lol,columns=[id_col]+[str(n) for n in range(0,1024)])
        
        assert_frame_equal(X_train_and_ids_df,expected_X_train_and_ids_df)
        
        expected_X_test_and_ids_lol = add_elements_to_start_of_list_of_lists(lol=[ChemDataParser.expected_fp_bits_list_of_complex_mol,ChemDataParser.expected_fp_bits_list_of_simple_mol],my_list=[7,9])
        
        expected_X_test_and_ids_df=pd.DataFrame(expected_X_test_and_ids_lol,columns=[id_col]+[str(n) for n in range(0,1024)]) #expected_X_train_and_ids_df #Deliberate mistake to check this fails the first time! => Failed as expected!
        
        assert_frame_equal(X_test_and_ids_df,expected_X_test_and_ids_df)
        #----------------------------------
    
    def test_getInputRequiredForModellingAndAD_from_smiles_no_pre_calc_stand_mol(self):
        
        id_col,smiles_col,class_label_col,class_1,fp_col,train_df,test_df = self.get_input_for_testing_getInputRequiredForModellingAndAD()
        
        #==============================
        
        fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids = getInputRequiredForModellingAndAD(train_df,test_df,id_col,smiles_col,class_label_col,class_1,fp_col=fp_col,bond_radius=2,bitInfo={},nBits=1024,pre_calc_stand_mol_col=None)
        
        #==============================
        
        self.run_all_check_on_output_of_getInputRequiredForModellingAndAD(id_col,smiles_col,class_label_col,class_1,fp_col,train_df,test_df,fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids)
    
    
    def test_getInputRequiredForModellingAndAD_from_smiles_with_pre_calc_stand_mol(self):
        
        id_col,smiles_col,class_label_col,class_1,fp_col,train_df,test_df = self.get_input_for_testing_getInputRequiredForModellingAndAD()
        
        pre_calc_stand_mol_col = 'StandardizedMol'
        
        train_df.insert(0,pre_calc_stand_mol_col,[standardize(smiles) for smiles in train_df[smiles_col].tolist()],allow_duplicates=True)
        
        test_df.insert(0,pre_calc_stand_mol_col,[standardize(smiles) for smiles in test_df[smiles_col].tolist()],allow_duplicates=True)
        
        #==============================
        
        fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids = getInputRequiredForModellingAndAD(train_df,test_df,id_col,smiles_col,class_label_col,class_1,fp_col=fp_col,bond_radius=2,bitInfo={},nBits=1024,pre_calc_stand_mol_col=pre_calc_stand_mol_col)
        
        #==============================
        
        self.run_all_check_on_output_of_getInputRequiredForModellingAndAD(id_col,smiles_col,class_label_col,class_1,fp_col,train_df,test_df,fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids)
    
    def test_getInputRequiredForModellingAndAD_from_smiles_with_pre_calc_default_fps(self):
        id_col,smiles_col,class_label_col,class_1,fp_col,train_df,test_df = self.get_input_for_testing_getInputRequiredForModellingAndAD()
        
        #==============================
        
        
        pre_calc_stand_mol_col = 'StandardizedMol'
        
        train_df.insert(0,pre_calc_stand_mol_col,[standardize(smiles) for smiles in train_df[smiles_col].tolist()],allow_duplicates=True)
        
        test_df.insert(0,pre_calc_stand_mol_col,[standardize(smiles) for smiles in test_df[smiles_col].tolist()],allow_duplicates=True)
        
        #==============================
        
        train_df,train_fps_df = ChemDataParser.getFpsDfAndDataDfSkippingFailedMols(data_df=train_df,smiles_col=smiles_col,id_col=id_col,fp_col=fp_col,pre_calc_stand_mol_col=pre_calc_stand_mol_col)
        
        train_df.insert(0,fp_col,train_fps_df[fp_col].tolist(),allow_duplicates=True)
        
        test_df,test_fps_df = ChemDataParser.getFpsDfAndDataDfSkippingFailedMols(data_df=test_df,smiles_col=smiles_col,id_col=id_col,fp_col=fp_col,pre_calc_stand_mol_col=pre_calc_stand_mol_col)
        
        test_df.insert(0,fp_col,test_fps_df[fp_col].tolist(),allow_duplicates=True)
        
        
        #==============================
        
        fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids = getInputRequiredForModellingAndAD(train_df,test_df,id_col,smiles_col,class_label_col,class_1,fp_col=fp_col,pre_calc_stand_mol_col=pre_calc_stand_mol_col,pre_calc_fps=True)
        
        #==============================
        
        self.run_all_check_on_output_of_getInputRequiredForModellingAndAD(id_col,smiles_col,class_label_col,class_1,fp_col,train_df,test_df,fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids,pre_calc_fps=True)
        
        
    
    def test_this_is_a_non_organic_compound(self):
        
        examples_dict = defaultdict(dict)
        #====================
        examples_dict[1]['smiles'] = 'c1ccccc1N'
        examples_dict[1]['non-organic?'] = False
        #====================
        examples_dict[2]['smiles'] = 'CC[O-].[Na+]'
        examples_dict[2]['non-organic?'] = False
        #====================
        examples_dict[3]['smiles'] = 'CCO[Na]' #Incorrect representation of example 2!
        examples_dict[3]['non-organic?'] = False
        #====================
        examples_dict[4]['smiles'] = 'c1ccccc1[Sn](c1ccccc1)([OH])c1ccccc1'
        examples_dict[4]['non-organic?'] = True
        #====================
        
        
        
        for eg in examples_dict.keys():
            print(f'Checking if example {eg} is a non-organic compound')
            raw_rdkit_mol = ChemDataParser.Chem.MolFromSmiles(examples_dict[eg]['smiles'])
            
            status = ChemDataParser.this_is_a_non_organic_compound(raw_rdkit_mol)
            
            expected_status = examples_dict[eg]['non-organic?']
            
            assert status == expected_status,f"examples_dict[eg]['smiles']={examples_dict[eg]['smiles']} is perceived as non-organic={status}"
    
    def test_get_all_non_salt_mol_component_as_molecules(self):
        
        examples_dict = defaultdict(dict)
        #====================
        examples_dict[1]['smiles'] = 'c1ccccc1N'
        examples_dict[1]['no. non-salt components'] = 1
        #====================
        examples_dict[2]['smiles'] = 'CC[O-].[Na+]'
        examples_dict[2]['no. non-salt components'] = 1
        #====================
        examples_dict[3]['smiles'] = 'CCO[Na]' #Incorrect representation of example 2!
        examples_dict[3]['no. non-salt components'] = 1
        #====================
        examples_dict[4]['smiles'] = 'c1ccccc1[Sn](c1ccccc1)([OH])c1ccccc1'
        examples_dict[4]['no. non-salt components'] = 1
        #====================
        examples_dict[5]['smiles'] = 'c1ccccc1.Nc1ccccc1'
        examples_dict[5]['no. non-salt components'] = 2
        #====================
        examples_dict[6]['smiles'] = 'CC[O-].[Na+].Nc1ccccc1'
        examples_dict[6]['no. non-salt components'] = 2
        #======================
        
        for eg in examples_dict.keys():
            print(f'Checking if non-salt components for example {eg} are correctly recognized')
            mol = ChemDataParser.Chem.MolFromSmiles(examples_dict[eg]['smiles'])
            
            components = ChemDataParser.get_all_non_salt_mol_component_as_molecules(mol)
            
            assert len(components) == examples_dict[eg]['no. non-salt components'],f"eg={eg},len(components)={len(components)},examples_dict[eg]['no. non-salt components']={examples_dict[eg]['no. non-salt components']}"

    
    def test_is_valid_smiles(self):
        
        assert ChemDataParser.is_valid_smiles('CCO')
        assert not ChemDataParser.is_valid_smiles('?')
    
    def test_consistency_of_tanimoto_distance_calculations_used_for_AD_methods(self):
        id_col,smiles_col,class_label_col,class_1,fp_col,train_df,test_df = self.get_input_for_testing_getInputRequiredForModellingAndAD()
        
        fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids = getInputRequiredForModellingAndAD(train_df,test_df,id_col,smiles_col,class_label_col,class_1,fp_col=fp_col,bond_radius=2,bitInfo={},nBits=1024,pre_calc_stand_mol_col=None)
        
        #----------------------
        assert isinstance(fps_train,pd.DataFrame)
        assert isinstance(fps_test,pd.DataFrame)
        assert isinstance(X_train_and_ids_df,pd.DataFrame)
        assert isinstance(X_test_and_ids_df,pd.DataFrame)
        #---------------------
        
        for train_index,train_fps_row in fps_train.iterrows():
            print(f'train_index={train_index}')
            X_train_and_ids_df_row_df = get_pandas_df_row_as_df(pandas_df=X_train_and_ids_df,row_index=train_index)
            
            X_train_and_ids_df_row_df_dummy_id = X_train_and_ids_df_row_df.copy(deep=True)
            dummy_id_df = pd.DataFrame({id_col:[1000]})
            X_train_and_ids_df_row_df_dummy_id.update(dummy_id_df)
            
            duplicated_X_train_and_ids_df_row_df = pd.concat([X_train_and_ids_df_row_df,X_train_and_ids_df_row_df_dummy_id]) #A weird error was raised by SciKit-Learn when I tried to supply the dataframe with one instance (row) to KNeighboursClassifier(...) [implicitly, via calling code below]
            
            #Debug:
            print(f'train IDs in duplicated_X_train_and_ids_df_row_df={duplicated_X_train_and_ids_df_row_df[id_col].tolist()}')
            
            k_val = 1

            kNN_based_AD_Method_Class_Instance = UNC_like_AD.UNC_like_AD_approach(train_df=duplicated_X_train_and_ids_df_row_df, id_col=id_col, k=k_val, distance_metric='jaccard', scale=False, endpoint_col=None,expected_descs_no=None)#only_consider_k_neighbours_for_final_threshold_calculations=True) #Is only_consider_k_neighbours_for_final_threshold_calculations redundant for UNC?
            
            kNN_based_AD_Method_Class_Instance.buildNNmodel(k_val)
            
            for test_index,test_fps_row in fps_test.iterrows():
                print(f'train_index={train_index},test_index={test_index}')
                
                X_test_and_ids_df_row_df = get_pandas_df_row_as_df(pandas_df=X_test_and_ids_df,row_index=test_index)
                
                
                print(f'Checking consistency of Tanimoto distance calculations for train ID={X_train_and_ids_df_row_df[id_col].tolist()[0]} and test ID={X_test_and_ids_df_row_df[id_col].tolist()[0]}')
                
                nd_list,ni_list,test_ids = kNN_based_AD_Method_Class_Instance.getNewCmpdsDistancesToTrainKNN(test_compounds_X_df=X_test_and_ids_df_row_df,id_col=id_col)
                
                #-----------------------
                assert 1 == len(nd_list)
                assert 1 == len(nd_list[0])
                #-----------------------
                
                tanimoto_dist_dkNN = nd_list[0][0]
                
                tanimoto_dist_d1NN = 1 - compute_tanimoto_similarity(train_fps_row[fp_col],test_fps_row[fp_col])
                
                assert tanimoto_dist_d1NN == tanimoto_dist_dkNN,f"tanimoto_dist_d1NN = {tanimoto_dist_d1NN} vs. tanimoto_dist_dkNN = {tanimoto_dist_dkNN}"
                
    def test_ACP_aggregation_of_prediction_intervals_is_based_on_medians_for_upper_and_lower_limits(self):
        
        ###################
        #Construct pretend ICP intervals based upon consulting the relevant ACP code in funcsToApplyRegUncertaintyToNewDatasets.py and the functions imported therein:
        X_test = pd.DataFrame({'x1':[0,1,0],'x2':[1,0,0]})
        
        
        icp_intervals = np.zeros((X_test.shape[0],  2))
        icp_intervals[0,0] = 0.1
        icp_intervals[0,1] = 0.8
        
        icp_intervals_2 = np.zeros((X_test.shape[0],  2))
        icp_intervals_2[0,0] = 0.08
        icp_intervals_2[0,1] = 0.9
        
        icp_intervals_3 = np.zeros((X_test.shape[0],  2))
        icp_intervals_3[0,0] = 0.0
        icp_intervals_3[0,1] = 100.0
        
        n_source=3
        
        sigLevels = [0.32]
        
        i=0 #Only one value, as only one sig-level
        
        intervals = np.zeros((len(sigLevels), n_source, X_test.shape[0], 2))
        
        for sourceIndex in range(0,n_source):
            if 0 == sourceIndex:
                current_icp_intervals = icp_intervals
            elif 1 == sourceIndex:
                current_icp_intervals = icp_intervals_2
            elif 2 == sourceIndex:
                current_icp_intervals = icp_intervals_3
            else:
                raise Exception(f'{sourceIndex}')
            
            intervals[i, sourceIndex, :, :] = current_icp_intervals
        
        combined_intervals = aggregateIntervals(intervals[i])
        #####################
        
        expected_combined_intervals = np.array([[0.08, 0.90],[0.  , 0.  ],[0.  , 0.  ]])
        
        if not np.allclose(combined_intervals, expected_combined_intervals, rtol=10**-5, atol=10**-8, equal_nan=False): raise Exception(f'combined_intervals={combined_intervals} vs. expected_combined_intervals={expected_combined_intervals}')
    
    def test_compute_prediction_as_the_midpoint_of_prediction_intervals(self):
        
        preds = RegUncert.compute_prediction_as_the_midpoint_of_prediction_intervals(prediction_intervals=np.array([[0.09, 0.85],[0.  , 0.  ],[0.  , 0.  ]]))
        
        expected_preds = np.array([0.47, 0.  , 0.  ])
        
        if not np.allclose(preds, expected_preds, rtol=10**-5, atol=10**-8, equal_nan=False): raise Exception(f'preds={preds} vs. expected_preds={expected_preds}')
    
    def test_compute_prediction_as_the_midpoint_of_prediction_intervals_still_works_with_infinitely_large_intervals(self):

        preds = RegUncert.compute_prediction_as_the_midpoint_of_prediction_intervals(prediction_intervals=np.array([[-np.inf, np.inf],[0.  , 0.  ],[0.  , 0.  ]]))
        
        expected_preds = np.array([0.0, 0.  , 0.  ])
        
        if not np.allclose(preds, expected_preds, rtol=10**-5, atol=10**-8, equal_nan=False): raise Exception(f'preds={preds} vs. expected_preds={expected_preds}') 
    
    def test_get_pandas_df_row_as_df(self):
        
        test_X = pd.DataFrame({'x1':[1,0],'x2':[0,1],'x3':[0,0]})
        
        for row_index in test_X.index.tolist():
            test_X_instance = get_pandas_df_row_as_df(pandas_df=test_X,row_index=row_index)
            
            assert isinstance(test_X_instance,pd.DataFrame),type(test_X_instance)
            assert test_X_instance.shape[1]==3,test_X_instance.shape[1]
            assert test_X_instance.shape[0]==1,test_X_instance.shape[0]
            assert test_X_instance.columns.values.tolist()==['x1','x2','x3'],test_X_instance.columns.values.tolist()
            assert test_X_instance.index.tolist()==[0],test_X_instance.index.tolist()
    
    def get_rf_regression_model_and_test_set_for_checking(self):
        
        train_x = pd.DataFrame({'x1':[0,1,1,0,1],'x2':[1,0,0,0,1],'x3':[0,0,0,1,0]})
        train_y = pd.Series([1.2,1.03,1.6,-0.5,1.1])
        
        model = RandomForestRegressor(n_estimators=10,random_state=42,max_features=1).fit(train_x,train_y)
        
        test_X = pd.DataFrame({'x1':[1,0],'x2':[0,1],'x3':[0,0]})
        
        assert test_X.index.tolist()==list(range(test_X.shape[0])),test_X.index.tolist()
        
        return model,test_X
    
    def test_get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance(self):
        model,test_X = self.get_rf_regression_model_and_test_set_for_checking()
        
        all_ensemble_predictions = model.predict(test_X).tolist()
        
        #======================
        #Following lines copied and adapted from RF_prediction_intervals.py:def pred_ints(...):
        list_of_all_trees = [tree for tree  in model.estimators_]
        
        for row_index in test_X.index.tolist():
            test_X_instance = get_pandas_df_row_as_df(pandas_df=test_X,row_index=row_index)
            
            preds = [get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance(tree=list_of_all_trees[tree_index],ensemble=model,tree_index=tree_index,test_X=test_X_instance,expected_type_of_test_X=pd.DataFrame,want_class_probs=False,ignore_user_warning=False) for tree_index in range(0,len(list_of_all_trees))]
            
            assert all_ensemble_predictions[row_index]==np.mean(preds),f"row_index={row_index},all_ensemble_predictions[row_index]={all_ensemble_predictions[row_index]},np.mean(preds)={np.mean(preds)}"
        #=========================
    
    def check_applyNativeMLRegModelToNewCmpds(self,uncertainty_alg):
        
        model,test_x = self.get_rf_regression_model_and_test_set_for_checking()
        
        
        testPred_list,intervals = RegUncert.applyNativeMLRegModelToNewCmpds(X_test=test_x,model=model,ml_alg="RandomForestRegressor",uncertainty_alg=uncertainty_alg,percentile=68,calc_preds_once=True,ignore_user_warning=False)
        
        assert len(intervals)==test_x.shape[0],len(intervals)
        assert len(testPred_list)==test_x.shape[0],len(testPred_list)
        
        for row_index in range(test_x.shape[0]):
            interval_lower = intervals[row_index,0]
            interval_upper = intervals[row_index,1]
            
            assert interval_lower < testPred_list[row_index],f"row_index={row_index}: interval_lower ={interval_lower} vs. testPred_list[row_index]={testPred_list[row_index]}"
            assert interval_upper > testPred_list[row_index],f"row_index={row_index}: interval_upper ={interval_upper} vs. testPred_list[row_index]={testPred_list[row_index]}"
    
    def test_applyNativeMLRegModelToNewCmpds_PI(self):
        
        self.check_applyNativeMLRegModelToNewCmpds("PI")
    
    
    def test_applyNativeMLRegModelToNewCmpds_ENS(self):
        
        self.check_applyNativeMLRegModelToNewCmpds("ENS")
    
    def test_compute_y_std(self):
        
        model,test_x = self.get_rf_regression_model_and_test_set_for_checking()
        
        y_std_list = compute_y_std(X=test_x, numberTrees=10,RF_reg_model=model,use_v1=True)
        
        assert len(y_std_list)==test_x.shape[0],len(y_std_list)
        assert all([isinstance(v,float) for v in y_std_list]),[type(v) for v in y_std_list]
        
        try:
            y_std_list = compute_y_std(X=test_x, numberTrees=9,RF_reg_model=model,use_v1=True)
        except AssertionError:
            assertion_error_raised_due_to_inconsistent_number_of_trees = True
        
        assert assertion_error_raised_due_to_inconsistent_number_of_trees
    
    def test_lamda_exp_std(self):
        from modelsADuncertaintyPkg.qsar_reg_uncertainty.RegressionICP import lamda_exp_std

        res1 = lamda_exp_std(np.array([1.5,1.2]))

        np.array_equal(np.array([np.exp(1.5),np.exp(1.2)]),res1)

        res2 = lamda_exp_std(pd.Series([1.5,1.2]))

        np.array_equal(np.array([np.exp(1.5),np.exp(1.2)]),res2)
    
    def test_compute_y_std_variations_give_consistent_results(self):

        model,test_x = self.get_rf_regression_model_and_test_set_for_checking()
        
        y_std_list_v1 = compute_y_std(X=test_x, numberTrees=10,RF_reg_model=model,use_v1=True)

        y_std_list_v2 = compute_y_std(X=test_x, numberTrees=10,RF_reg_model=model,use_v1=False)

        y_std_list_default = compute_y_std(X=test_x, numberTrees=10,RF_reg_model=model)

        assert approx(y_std_list_v1) == y_std_list_default
        assert approx(y_std_list_v2) == y_std_list_default
    
    def call_compute_y_std_v1(self):
        
        compute_y_std(X=self.test_x, numberTrees=10,RF_reg_model=self.model,use_v1=True)
    
    def call_compute_y_std_v2(self):
        
        compute_y_std(X=self.test_x, numberTrees=10,RF_reg_model=self.model,use_v1=False)


    def check_compute_y_std_v2_is_faster(self,how_many=50,repeats=2):
        model,test_x = self.get_rf_regression_model_and_test_set_for_checking()

        self.model = model
        self.test_x= test_x

        imports_and_definitions_statement = ''
        
        v1_times = times_of_repeated_many_function_calls(self.call_compute_y_std_v1,imports_and_definitions_statement,how_many=how_many,repeats=repeats,collect_garbage=True)

        v2_times = times_of_repeated_many_function_calls(self.call_compute_y_std_v2,imports_and_definitions_statement,how_many=how_many,repeats=repeats,collect_garbage=True)

        del self.model
        del self.test_x

        max_v2_time = max(v2_times)

        print(f'max_v2_time={max_v2_time}')

        min_v1_time = min(v1_times)

        print(f'min_v1_time={min_v1_time}')

        assert max_v2_time < min_v1_time
    
    def test_compute_y_std_v2_is_faster(self):
        self.check_compute_y_std_v2_is_faster()

    def test_pred_ints_variations_give_consistent_results(self):

        model,test_x = self.get_rf_regression_model_and_test_set_for_checking()

        y_pred = model.predict(test_x)

        pred_ints_v1 = pred_ints(model, test_x, y_pred, 67,ignore_user_warning=False,use_v1=True)

        pred_ints_v2 = pred_ints(model, test_x, y_pred, 67,ignore_user_warning=False,use_v1=False)

        pred_ints_default = pred_ints(model, test_x, y_pred, 67,ignore_user_warning=False)

        assert approx(pred_ints_v1) == pred_ints_default
        assert approx(pred_ints_v2) == pred_ints_default

    def call_pred_ints_v1(self):
        
        pred_ints(self.model, self.test_x, self.y_pred, 67,ignore_user_warning=False,use_v1=True)
    
    def call_pred_ints_v2(self):
        
        pred_ints(self.model, self.test_x, self.y_pred, 67,ignore_user_warning=False,use_v1=False)

    def check_pred_ints_v2_is_faster(self,how_many=50,repeats=2):
        model,test_x = self.get_rf_regression_model_and_test_set_for_checking()

        self.model = model
        self.test_x= test_x
        self.y_pred = model.predict(test_x)

        imports_and_definitions_statement = ''
        
        v1_times = times_of_repeated_many_function_calls(self.call_pred_ints_v1,imports_and_definitions_statement,how_many=how_many,repeats=repeats,collect_garbage=True)

        v2_times = times_of_repeated_many_function_calls(self.call_pred_ints_v2,imports_and_definitions_statement,how_many=how_many,repeats=repeats,collect_garbage=True)

        del self.model
        del self.test_x
        del self.y_pred

        max_v2_time = max(v2_times)

        print(f'max_v2_time={max_v2_time}')

        min_v1_time = min(v1_times)

        print(f'min_v1_time={min_v1_time}')

        assert max_v2_time < min_v1_time
    
    def test_pred_ints_v2_is_faster(self):
        self.check_pred_ints_v2_is_faster()
    
    def test_compute_intervals(self):
        y_pred = [4.5,8.5,9.5,1.5]
        half_interval_pred_sizes = [0.4,0.2,0.3,0.1]
        
        res_1 = compute_pred_intervals(y_pred,half_interval_pred_sizes)
        res_2 = compute_pred_intervals(np.array(y_pred),np.array(half_interval_pred_sizes))
        res_3 = compute_pred_intervals(y_pred,np.array(half_interval_pred_sizes))
        res_4 = compute_pred_intervals(np.array(y_pred),half_interval_pred_sizes)

        assert np.array_equal(res_1,np.array([[4.1,4.9],[8.3,8.7],[9.2,9.8],[1.4,1.6]]))
        assert np.array_equal(res_2,np.array([[4.1,4.9],[8.3,8.7],[9.2,9.8],[1.4,1.6]]))
        assert np.array_equal(res_3,np.array([[4.1,4.9],[8.3,8.7],[9.2,9.8],[1.4,1.6]]))
        assert np.array_equal(res_4,np.array([[4.1,4.9],[8.3,8.7],[9.2,9.8],[1.4,1.6]]))


    
    def test_get_no_instances(self):
        
        examples = []
        examples.append(np.array([1.1,1.8,2.1]))
        examples.append(pd.Series([1.2,1.8,2.1]))
        examples.append([1.3,1.8,2.1])
        examples.append(np.zeros((3,  2)))#c.f. intervals object construction in RegressionICP.py
        
        for eg in examples:
            assert 3 == get_no_instances(eg),f'type(eg)={type(eg)},eg={eg},get_no_instances(eg)={get_no_instances(eg)}'

    def get_inputs_required_for_checking_conformal_regression_workflow_functions(self):
        from modelsADuncertaintyPkg.utils.load_example_datasets_for_code_checks import generateExampleDatasetForChecking_Housing, Housing_x_names, Housing_y_name
        from modelsADuncertaintyPkg.utils.ML_utils import singleRandomSplit

        overall_trial_size = 50
        test_size = 10
        calib_size = 10

        cal_fract = calib_size/(overall_trial_size-test_size)

        seed = 42

        subset_df = generateExampleDatasetForChecking_Housing().sample(n=overall_trial_size,random_state=seed).reset_index(drop=True) #sample(...,ignore_index=True) only available as of pandas version 1.3.0!

        subset_x = subset_df[Housing_x_names]
        #------------------
        assert isinstance(subset_x,pd.DataFrame),f'type(subset_x)={type(subset_x)}'
        assert list(range(subset_x.shape[0]))==subset_x.index.tolist(),f'subset_x.index.tolist()={subset_x.index.tolist()}'
        #------------------

        subset_y = subset_df[Housing_y_name]

        train_inc_calib_x,train_inc_calib_y,test_x,test_y = singleRandomSplit(data_x=subset_x,data_y=subset_y,test_fraction=(test_size/overall_trial_size),random_state=seed,stratified=True)

        return train_inc_calib_x,train_inc_calib_y,test_x,test_y,cal_fract,seed,calib_size

    def checks_which_we_expect_to_almost_always_apply_to_conformal_regression_confScores(self,confScores):
        assert isinstance(confScores,pd.Series) #repeated from checks below
        try:
            assert not any([0==non_conf_score for non_conf_score in confScores.tolist()]),f'confScores.tolist()={confScores.tolist()}'
        except AssertionError:
            print(f'WARNING: some of these non-conformity scores are zero: {confScores.tolist()} ')
            zero_count = len([v for v in confScores.tolist() if 0==v])
            assert not len(confScores.tolist()) == zero_count

    def checks_which_we_expect_to_almost_always_apply_to_conformal_regression_intervals(self,intervals,testPred_list):
        assert isinstance(intervals,np.ndarray) #repeated from checks below
        assert isinstance(testPred_list,list) #repeated from checks below
        
        for i in range(intervals.shape[0]):
            lower = intervals[i,0]
            upper = intervals[i,1]
            width = abs(upper-lower)
            assert not 0 == width
        
            #if not lower == approx(upper):
            assert lower < upper,f'lower={lower},upper={upper}'
            
            assert testPred_list[i] > lower and testPred_list[i] < upper,f'testPred_list[i]={testPred_list[i]},lower={lower},upper={upper}'

    
    def checks_to_apply_to_conformal_regression_confScores_and_model(self,confScores,model,calib_size):
        assert isinstance(confScores,pd.Series),f'type(confScores)={type(confScores)}' #when y=pd.Series(...),y_std=list,pred=list, then steps in lamda_exp_std(y_std, w=1) followed by computeNonConformityScores(pred, y, lamda_calib) were observed to generate confScores as a pd.Series object!
        assert calib_size == len(confScores),f'calib_size={calib_size}, len(confScores)={len(confScores)}'
        assert isinstance(model,RandomForestRegressor)

        self.checks_which_we_expect_to_almost_always_apply_to_conformal_regression_confScores(confScores)

        

    def checks_to_apply_to_conformal_regression_predictions_and_intervals(self,testPred_list,intervals):
        assert isinstance(testPred_list,list),f'type(testPred_list)={type(testPred_list)}'
        assert isinstance(intervals,np.ndarray),f'type(intervals)={type(intervals)}'
        assert len(testPred_list) == intervals.shape[0]
        assert 2 == intervals.shape[1]
        
        assert len(testPred_list)==len(intervals),f'len(testPred_list)={len(testPred_list)} vs. len(intervals)={len(intervals)}'
        
        assert not any(pd.isna(np.array(testPred_list))),f'testPred_list={testPred_list}'
            

        self.checks_which_we_expect_to_almost_always_apply_to_conformal_regression_intervals(intervals,testPred_list)

    def test_getConformalRegressionModelsPlusCalibDetails_ICP(self):
        
        train_inc_calib_x,train_inc_calib_y,test_x,test_y,cal_fract,seed,calib_size = self.get_inputs_required_for_checking_conformal_regression_workflow_functions()
        
        confScores,model = RegUncert.getConformalRegressionModelsPlusCalibDetails_ICP(train_inc_calib_x,train_inc_calib_y,global_random_seed=seed,calib_fraction=cal_fract,ml_alg="RandomForestRegressor",nrTrees=100,non_conformity_scaling="exp(stdev of tree predictions)",stratified=False)

        self.checks_to_apply_to_conformal_regression_confScores_and_model(confScores,model,calib_size)

    def test_applyConformalRegressionToNewCmpds_ICP(self):

        train_inc_calib_x,train_inc_calib_y,test_x,test_y,cal_fract,seed,calib_size = self.get_inputs_required_for_checking_conformal_regression_workflow_functions()
        
        _no_trees = 40

        confScores,model = RegUncert.getConformalRegressionModelsPlusCalibDetails_ICP(train_inc_calib_x,train_inc_calib_y,global_random_seed=seed,calib_fraction=cal_fract,ml_alg="RandomForestRegressor",nrTrees=_no_trees,non_conformity_scaling="exp(stdev of tree predictions)",stratified=False)

        testPred_list,intervals = RegUncert.applyConformalRegressionToNewCmpds_ICP(confScores,model,X_test=test_x,sig_level=0.32,ml_alg="RandomForestRegressor",nrTrees=_no_trees,non_conformity_scaling="exp(stdev of tree predictions)")

        self.checks_to_apply_to_conformal_regression_predictions_and_intervals(testPred_list,intervals)
    

    def test_getConformalRegressionModelsPlusCalibDetails_ACP(self):
        train_inc_calib_x,train_inc_calib_y,test_x,test_y,cal_fract,seed,calib_size = self.get_inputs_required_for_checking_conformal_regression_workflow_functions()
        
        _no_trees = 40
        _no_calib_splits = 20

        ACP_dict_of_confScores,ACP_dict_of_model = RegUncert.getConformalRegressionModelsPlusCalibDetails_ACP(train_inc_calib_x,train_inc_calib_y,global_random_seed=seed,number_of_calib_splits=_no_calib_splits,ml_alg="RandomForestRegressor",nrTrees=_no_trees,non_conformity_scaling="exp(stdev of tree predictions)",stratified=False,calib_fraction=cal_fract)

        assert isinstance(ACP_dict_of_confScores,dict)
        assert isinstance(ACP_dict_of_model,dict)
        assert _no_calib_splits == len(ACP_dict_of_model.keys())
        assert _no_calib_splits == len(ACP_dict_of_confScores.keys())

        for split_index in range(_no_calib_splits):
            confScores = ACP_dict_of_confScores[split_index]
            model = ACP_dict_of_model[split_index]

            self.checks_to_apply_to_conformal_regression_confScores_and_model(confScores,model,calib_size)
    
    def test_applyConformalRegressionToNewCmpds_ACP(self):
        train_inc_calib_x,train_inc_calib_y,test_x,test_y,cal_fract,seed,calib_size = self.get_inputs_required_for_checking_conformal_regression_workflow_functions()
        
        _no_trees = 40
        _no_calib_splits = 20

        ACP_dict_of_confScores,ACP_dict_of_model = RegUncert.getConformalRegressionModelsPlusCalibDetails_ACP(train_inc_calib_x,train_inc_calib_y,global_random_seed=seed,number_of_calib_splits=_no_calib_splits,ml_alg="RandomForestRegressor",nrTrees=_no_trees,non_conformity_scaling="exp(stdev of tree predictions)",stratified=False,calib_fraction=cal_fract)

        testPred_list,intervals = RegUncert.applyConformalRegressionToNewCmpds_ACP(ACP_dict_of_confScores,ACP_dict_of_model,X_test=test_x,sig_level=0.32,ml_alg="RandomForestRegressor",nrTrees=_no_trees,non_conformity_scaling="exp(stdev of tree predictions)")

        self.checks_to_apply_to_conformal_regression_predictions_and_intervals(testPred_list,intervals)

    def test_getConformalRegressionModelsPlusCalibDetails_SCP(self):
        train_inc_calib_x,train_inc_calib_y,test_x,test_y,cal_fract,seed,calib_size = self.get_inputs_required_for_checking_conformal_regression_workflow_functions()

        _no_trees = 40
        _no_scp_train_splits = 5

        confScores,SCP_dict_of_model = RegUncert.getConformalRegressionModelsPlusCalibDetails_SCP(train_inc_calib_x,train_inc_calib_y,global_random_seed=seed,number_of_train_splits=_no_scp_train_splits,calib_fraction=cal_fract,ml_alg="RandomForestRegressor",nrTrees=_no_trees,non_conformity_scaling="exp(stdev of tree predictions)",stratified=False)

        assert isinstance(SCP_dict_of_model,dict)
        assert _no_scp_train_splits == len(SCP_dict_of_model.keys())

        for split_index in range(_no_scp_train_splits):
            
            model = SCP_dict_of_model[split_index]

            self.checks_to_apply_to_conformal_regression_confScores_and_model(confScores,model,calib_size)
    
    def test_applyConformalRegressionToNewCmpds_SCP(self):
        train_inc_calib_x,train_inc_calib_y,test_x,test_y,cal_fract,seed,calib_size = self.get_inputs_required_for_checking_conformal_regression_workflow_functions()

        _no_trees = 40
        _no_scp_train_splits = 5

        confScores,SCP_dict_of_model = RegUncert.getConformalRegressionModelsPlusCalibDetails_SCP(train_inc_calib_x,train_inc_calib_y,global_random_seed=seed,number_of_train_splits=_no_scp_train_splits,calib_fraction=cal_fract,ml_alg="RandomForestRegressor",nrTrees=_no_trees,non_conformity_scaling="exp(stdev of tree predictions)",stratified=False)

        testPred_list,intervals = RegUncert.applyConformalRegressionToNewCmpds_SCP(confScores,SCP_dict_of_model,X_test=test_x,sig_level=0.32,ml_alg="RandomForestRegressor",nrTrees=_no_trees,non_conformity_scaling="exp(stdev of tree predictions)")

        self.checks_to_apply_to_conformal_regression_predictions_and_intervals(testPred_list,intervals)
    

    def test_parse_acp_nan_predictions(self):

        intervals_1 = np.array([[1.3,1.7],[4.9,5.1],[-1.9,-1.7]])
        input_1 = np.array([1.5,5.0,-1.8])
        output_1 = RegUncert.parse_acp_nan_predictions(input_1,intervals_1)

        assert isinstance(output_1,np.ndarray),f'type(output_1)={type(output_1)}'
        assert np.array_equal(output_1,input_1),output_1

        intervals_2 = np.array([[1.3,1.7],[-np.inf,np.inf],[-1.9,-1.7]])
        input_2 = np.array([1.5,np.nan,-1.8])
        output_2 = RegUncert.parse_acp_nan_predictions(input_2,intervals_2)

        assert isinstance(output_2,np.ndarray),f'type(output_2)={type(output_2)}'
        assert np.array_equal(np.array([1.5,0.0,-1.8]),output_2),output_2
    
    def test_getTestPredictionsAndPredictionIntervalsForAllSignificanceLevels_not_allowing_for_acp_predictions_could_be_inconsistent_at_different_significance_levels(self):
        train_inc_calib_x,train_inc_calib_y,test_x,test_y,cal_fract,seed,calib_size = self.get_inputs_required_for_checking_conformal_regression_workflow_functions()

        all_sig_levels_considered_to_compute_ECE = [0.32] #In reality, for ECE calculation, we would consider the whole range from 0 to 1.0, but speed up the calculations for this test!
        sig_level_of_interest = 0.32
        allow_for_acp_predictions_could_be_inconsistent=False

        ml_alg = "RandomForestRegressor"
        non_conformity_scaling = "exp(stdev of tree predictions)"
        global_random_seed = 42
        number_of_scp_splits = 5
        number_of_acp_splits = 5
        icp_calib_fraction=0.25
        nrTrees = 40
        native_uncertainty_alg_variant = "PI"


        for uncertainty_method in ["Native","ICP","ACP","SCP"]:

            sig_level_2_test_predictions,sig_level_2_prediction_intervals = RegUncert.getTestPredictionsAndPredictionIntervalsForAllSignificanceLevels(all_sig_levels_considered_to_compute_ECE,uncertainty_method,train_inc_calib_x,train_inc_calib_y,test_x,non_conformity_scaling,ml_alg,global_random_seed,number_of_scp_splits,number_of_acp_splits,icp_calib_fraction,nrTrees,native_uncertainty_alg_variant,allow_for_acp_predictions_could_be_inconsistent=allow_for_acp_predictions_could_be_inconsistent)

            test_predictions = sig_level_2_test_predictions[sig_level_of_interest]

            for sig_level in all_sig_levels_considered_to_compute_ECE:
                intervals = sig_level_2_prediction_intervals[sig_level]

                self.checks_to_apply_to_conformal_regression_predictions_and_intervals(testPred_list=test_predictions,intervals=intervals)

    def test_getTestPredictionsAndPredictionIntervalsForAllSignificanceLevels_allowing_for_acp_predictions_could_be_inconsistent_at_different_significance_levels(self):
        train_inc_calib_x,train_inc_calib_y,test_x,test_y,cal_fract,seed,calib_size = self.get_inputs_required_for_checking_conformal_regression_workflow_functions()

        all_sig_levels_considered_to_compute_ECE = [0.32] #In reality, for ECE calculation, we would consider the whole range from 0 to 1.0, but speed up the calculations for this test!
        sig_level_of_interest = 0.32
        allow_for_acp_predictions_could_be_inconsistent=True

        ml_alg = "RandomForestRegressor"
        non_conformity_scaling = "exp(stdev of tree predictions)"
        global_random_seed = 42
        number_of_scp_splits = 5
        number_of_acp_splits = 5
        icp_calib_fraction=0.25
        nrTrees = 40
        native_uncertainty_alg_variant = "PI"


        for uncertainty_method in ["ACP"]:

            sig_level_2_test_predictions,sig_level_2_prediction_intervals = RegUncert.getTestPredictionsAndPredictionIntervalsForAllSignificanceLevels(all_sig_levels_considered_to_compute_ECE,uncertainty_method,train_inc_calib_x,train_inc_calib_y,test_x,non_conformity_scaling,ml_alg,global_random_seed,number_of_scp_splits,number_of_acp_splits,icp_calib_fraction,nrTrees,native_uncertainty_alg_variant,allow_for_acp_predictions_could_be_inconsistent=allow_for_acp_predictions_could_be_inconsistent)

            test_predictions = sig_level_2_test_predictions[sig_level_of_interest]

            for sig_level in all_sig_levels_considered_to_compute_ECE:
                intervals = sig_level_2_prediction_intervals[sig_level]

                self.checks_to_apply_to_conformal_regression_predictions_and_intervals(testPred_list=test_predictions,intervals=intervals)

    def test_getTestPredictionsAndPredictionIntervalsForAllSignificanceLevels_does_not_crash_when_we_monitor_time(self):
        train_inc_calib_x,train_inc_calib_y,test_x,test_y,cal_fract,seed,calib_size = self.get_inputs_required_for_checking_conformal_regression_workflow_functions()

        all_sig_levels_considered_to_compute_ECE = [0.32] #In reality, for ECE calculation, we would consider the whole range from 0 to 1.0, but speed up the calculations for this test!
        sig_level_of_interest = 0.32
        allow_for_acp_predictions_could_be_inconsistent=True

        ml_alg = "RandomForestRegressor"
        non_conformity_scaling = "exp(stdev of tree predictions)"
        global_random_seed = 42
        number_of_scp_splits = 5
        number_of_acp_splits = 5
        icp_calib_fraction=0.25
        nrTrees = 40
        native_uncertainty_alg_variant = "PI"


        for uncertainty_method in ["Native","ICP","ACP","SCP"]:

            sig_level_2_test_predictions,sig_level_2_prediction_intervals = RegUncert.getTestPredictionsAndPredictionIntervalsForAllSignificanceLevels(all_sig_levels_considered_to_compute_ECE,uncertainty_method,train_inc_calib_x,train_inc_calib_y,test_x,non_conformity_scaling,ml_alg,global_random_seed,number_of_scp_splits,number_of_acp_splits,icp_calib_fraction,nrTrees,native_uncertainty_alg_variant,allow_for_acp_predictions_could_be_inconsistent=allow_for_acp_predictions_could_be_inconsistent,switch_on_monitor_time=True)

            test_predictions = sig_level_2_test_predictions[sig_level_of_interest]

            for sig_level in all_sig_levels_considered_to_compute_ECE:
                intervals = sig_level_2_prediction_intervals[sig_level]

                self.checks_to_apply_to_conformal_regression_predictions_and_intervals(testPred_list=test_predictions,intervals=intervals)
    
    
    def test_regression_statistics_functions_work_if_have_inf_predictions(self):
        test_y=[1.02,1.8,1.9]
        test_predictions=[1.5,np.inf,-1.0]

        sig_level_of_interest = 0.32

        sig_level_2_prediction_intervals = {sig_level_of_interest:np.zeros((3,  2))}
        
        sig_level_2_prediction_intervals[sig_level_of_interest][0,0] = 1.3
        sig_level_2_prediction_intervals[sig_level_of_interest][0,1] = 1.7
        
        sig_level_2_prediction_intervals[sig_level_of_interest][1,0] = -np.inf
        sig_level_2_prediction_intervals[sig_level_of_interest][1,1] = np.inf

        sig_level_2_prediction_intervals[sig_level_of_interest][2,0] = -1.2
        sig_level_2_prediction_intervals[sig_level_of_interest][2,1] = -0.8

        rmse,MAD,R2,Pearson,Pearson_Pval_one_tail,Spearman,Spearman_Pval_one_tail,validity, efficiency,ECE_new,ENCE,errorRate_s,no_compounds,scc = computeAllRegMetrics(test_y,test_predictions,sig_level_of_interest,sig_level_2_prediction_intervals,smallest_no_cmpds_to_compute_stats=2)

        assert np.inf == rmse,f'rmse={rmse}'
        assert np.inf==MAD,f'MAD={MAD}'
        assert -np.inf==R2,f'R2={R2}'

    def test_scsplit_stratified_continuous_split_df(self):
        check_stratified_continuous_split.check_scsplit()
    
    def test_scsplit_stratified_continuous_split_X_and_y(self):
        check_stratified_continuous_split.check_scsplit_with_separate_x_and_y()
    
    def test_scsplit_with_categorical_dataset(self):
        check_stratified_continuous_split.check_scsplit_with_categorical_dataset()
    
    def test_scsplit_with_separate_x_and_categorical_y(self):
        check_stratified_continuous_split.check_scsplit_with_separate_x_and_categorical_y()
    
    def check_dk_NN_calculations_give_consistent_results_to_2016_r_code_minor_adaptation(self,all_data_df,id_col,endpoint_col,r_code_training_thresholds_csv,expected_test_id_ad_status_dict,k_val,distance_metric_for_comparison,scaling):
        
        test_ids = list(expected_test_id_ad_status_dict.keys())

        expected_training_thresholds_df = pd.read_csv(r_code_training_thresholds_csv)
        
        expected_TrainIdsMatchedToThresholds = dict(zip(expected_training_thresholds_df[id_col].tolist(),expected_training_thresholds_df['Threshold']))
        
        train_df = all_data_df[~all_data_df[id_col].isin(test_ids)].drop(endpoint_col,axis=1,inplace=False)
        
        test_df = all_data_df[all_data_df[id_col].isin(test_ids)].drop(endpoint_col,axis=1,inplace=False)
        
        
        #####################
        #Before 17/11/23: Copied and adapted from applyADmethods.py:
        X_train = train_df
        X_test = test_df
        
        #####################
        #17/11/23: This might cause issues when wanting to check the code with continuous descriptors!
        #X_train = X_train.astype(int)
        #X_test = X_test.astype(int)
        #####################
        
        dk_NN_thresholds_instance = dk_NN_AD.dk_NN_thresholds(train_df=X_train, id_col=id_col, k=k_val,distance_metric=distance_metric_for_comparison, scale=scaling,debug=False)
        
        dk_NN_thresholds_instance.debug=True

        dk_NN_thresholds_instance.getInitialTrainingSetThresholds()
        
        dk_NN_thresholds_instance.updateZeroValuedTrainingSetThresholds()
        
        test_id_ad_status_dict = dk_NN_thresholds_instance.getADstatusOfTestSetCompounds(test_compounds_X_df=X_test, id_col=id_col)
        #########################
        
        assert list(expected_TrainIdsMatchedToThresholds.keys())==list(dk_NN_thresholds_instance.TrainIdsMatchedToThresholds.keys())

        for train_id in expected_TrainIdsMatchedToThresholds.keys():
            assert approx(expected_TrainIdsMatchedToThresholds[train_id]) == dk_NN_thresholds_instance.TrainIdsMatchedToThresholds[train_id],f'train_id={train_id}. expected threshold={expected_TrainIdsMatchedToThresholds[train_id]} vs. threshold={dk_NN_thresholds_instance.TrainIdsMatchedToThresholds[train_id]}'
        
        assert expected_test_id_ad_status_dict==test_id_ad_status_dict#pytest should provide this information:,f'test_id_ad_status_dict={test_id_ad_status_dict} vs. expected_test_id_ad_status_dict={expected_test_id_ad_status_dict}'

    def test_dk_NN_calculations_give_consistent_results_to_2016_r_code_minor_adaptation_BCWD(self):
        from modelsADuncertaintyPkg.utils.load_example_datasets_for_code_checks import generateExampleDatasetForChecking_BCWD
        
        all_data_df = generateExampleDatasetForChecking_BCWD()
        id_col = 'ID'
        endpoint_col = 'Class'
        
        k_val = 50

        distance_metric_for_comparison='euclidean'
        
        test_ids = [0,3]
        
        r_code_training_thresholds_csv = os.path.sep.join([this_dir,f'expected_BCWD_dk_NN_training_set_thresholds_k={k_val}_test_ids={"_and_".join([str(v) for v in test_ids])}.csv'])
        
        expected_test_id_ad_status_dict = defaultdict(dict)
        expected_test_id_ad_status_dict[0]['InsideAD'] = True
        expected_test_id_ad_status_dict[3]['InsideAD'] = False

        self.check_dk_NN_calculations_give_consistent_results_to_2016_r_code_minor_adaptation(all_data_df,id_col,endpoint_col,r_code_training_thresholds_csv,expected_test_id_ad_status_dict,k_val,distance_metric_for_comparison,scaling=True)

    def test_dk_NN_calculations_give_consistent_results_to_2016_r_code_minor_adaptation_ToyData(self):
        toy_data_csv = os.path.sep.join([this_dir,'Toy.csv']) #All descriptor values are 0 to 1, so scaling is redundant!
        
        all_data_df = pd.read_csv(toy_data_csv)
        id_col = 'ID'
        endpoint_col = 'Class'
        
        k_val = 2

        distance_metric_for_comparison='euclidean'
        
        test_ids = [0,3]
        
        r_code_training_thresholds_csv = os.path.sep.join([this_dir,f'expected_ToyData_dk_NN_training_set_thresholds_k={k_val}_test_ids={"_and_".join([str(v) for v in test_ids])}.csv'])
        
        expected_test_id_ad_status_dict = defaultdict(dict)
        expected_test_id_ad_status_dict[0]['InsideAD'] = True
        expected_test_id_ad_status_dict[3]['InsideAD'] = False

        self.check_dk_NN_calculations_give_consistent_results_to_2016_r_code_minor_adaptation(all_data_df,id_col,endpoint_col,r_code_training_thresholds_csv,expected_test_id_ad_status_dict,k_val,distance_metric_for_comparison,scaling=False)
        
    def test_report_name_of_function_where_this_is_called(self):
        
        reported_name = report_name_of_function_where_this_is_called()

        assert reported_name == 'test_report_name_of_function_where_this_is_called',f'reported_name={reported_name}'


    def test_numpy_array_contains_zeros(self):

        a1 = np.array([1,1.5,2.8])
        a2 = np.array([1,0,2.8])

        assert not numpy_array_contains_zeros(a1)
        assert numpy_array_contains_zeros(a2)
    
    def test_ENCE_binning_for_small_test_sets(self):

        examples_dict = defaultdict(dict)
        ##############################
        examples_dict['40_cmpds']['y_true'] = [1.5,1.8,4.2,3.8,4.9,3.7,5.6,8.9,8.7,2.3,4.2,0.5,2.9,11.9,9.1,6.2,3.81,3.79,3.4,4.8,3.738416237,2.567501089,0.668148014,-0.639995051,-1.921008667,-3.268343681,-4.776955907,-6.599146981,-7.856838111,-9.452347361,-10.52231359,-11.58469748,-12.72269451,-14.57062656,-15.84610038,-17.27026746,-19.00891467,-20.98244982,-22.44643775,-23.60451254]
        examples_dict['40_cmpds']['y_pred'] = [1.531094177,2.04099802,4.557883766,4.139333809,5.465407798,3.814994406,5.749089749,9.877191696,10.30589635,2.332727598,4.310342403,0.422800585,3.272572201,12.92028741,9.115967329,7.303709706,4.117752416,4.326521168,3.833264737,4.800017326,4.373933722,2.874686226,0.587295354,-0.790414425,-2.217289321,-3.890144906,-5.233617029,-7.182248061,-8.523878383,-11.38445113,-11.53476957,-11.6883688,-14.91278059,-16.28678725,-18.33956271,-18.97449796,-19.64410457,-21.26880675,-24.39275463,-26.11893217]
        examples_dict['40_cmpds']['estimated_variance'] = [2.372231966,4.336364821,20.9479272,17.22973884,29.96617983,14.91268639,33.43824155,98.45370326,106.4092907,5.579494291,18.79370788,0.218173743,10.8183659,167.6857064,83.72248865,53.72832643,16.9574983,18.97382249,14.92890401,23.42079471,19.40788086,8.466564145,0.383471426,0.605134077,4.827188637,14.77009263,27.14788135,51.48774731,72.04206859,128.632606,132.997857,136.3610982,221.2246532,263.7992233,335.5065783,359.1391982,384.4076445,451.6889723,593.0941181,682.1929084]
        examples_dict['40_cmpds']['no_cmpds'] = 40
        examples_dict['40_cmpds']['min_no_samples_per_bin'] = 20
        examples_dict['40_cmpds']['expected_no_bins'] = 2
        #############################
        ##############################
        examples_dict['3_cmpds']['y_true'] = [1.5,1.8,4.2]
        examples_dict['3_cmpds']['y_pred'] = [1.531094177,2.04099802,4.557883766]
        examples_dict['3_cmpds']['estimated_variance'] = [2.372231966,4.336364821,20.9479272]
        examples_dict['3_cmpds']['no_cmpds'] = 3
        examples_dict['3_cmpds']['min_no_samples_per_bin'] = 20
        examples_dict['3_cmpds']['expected_no_bins'] = 1
        #############################
        ##############################
        examples_dict['22_cmpds']['y_true'] = [1.5,1.8,4.2,3.8,4.9,3.7,5.6,8.9,8.7,2.3,4.2,0.5,2.9,11.9,9.1,6.2,3.81,3.79,3.4,4.8,3.738416237,2.567501089]
        examples_dict['22_cmpds']['y_pred'] = [1.531094177,2.04099802,4.557883766,4.139333809,5.465407798,3.814994406,5.749089749,9.877191696,10.30589635,2.332727598,4.310342403,0.422800585,3.272572201,12.92028741,9.115967329,7.303709706,4.117752416,4.326521168,3.833264737,4.800017326,4.373933722,2.874686226]
        examples_dict['22_cmpds']['estimated_variance'] = [2.372231966,4.336364821,20.9479272,17.22973884,29.96617983,14.91268639,33.43824155,98.45370326,106.4092907,5.579494291,18.79370788,0.218173743,10.8183659,167.6857064,83.72248865,53.72832643,16.9574983,18.97382249,14.92890401,23.42079471,19.40788086,8.466564145]
        examples_dict['22_cmpds']['no_cmpds'] = 22
        examples_dict['22_cmpds']['min_no_samples_per_bin'] = 20
        examples_dict['22_cmpds']['expected_no_bins'] = 1
        #############################
        ##############################
        examples_dict['41_cmpds']['y_true'] = [55,1.5,1.8,4.2,3.8,4.9,3.7,5.6,8.9,8.7,2.3,4.2,0.5,2.9,11.9,9.1,6.2,3.81,3.79,3.4,4.8,3.738416237,2.567501089,0.668148014,-0.639995051,-1.921008667,-3.268343681,-4.776955907,-6.599146981,-7.856838111,-9.452347361,-10.52231359,-11.58469748,-12.72269451,-14.57062656,-15.84610038,-17.27026746,-19.00891467,-20.98244982,-22.44643775,-23.60451254]
        examples_dict['41_cmpds']['y_pred'] = [52,1.531094177,2.04099802,4.557883766,4.139333809,5.465407798,3.814994406,5.749089749,9.877191696,10.30589635,2.332727598,4.310342403,0.422800585,3.272572201,12.92028741,9.115967329,7.303709706,4.117752416,4.326521168,3.833264737,4.800017326,4.373933722,2.874686226,0.587295354,-0.790414425,-2.217289321,-3.890144906,-5.233617029,-7.182248061,-8.523878383,-11.38445113,-11.53476957,-11.6883688,-14.91278059,-16.28678725,-18.33956271,-18.97449796,-19.64410457,-21.26880675,-24.39275463,-26.11893217]
        examples_dict['41_cmpds']['estimated_variance'] = [800,2.372231966,4.336364821,20.9479272,17.22973884,29.96617983,14.91268639,33.43824155,98.45370326,106.4092907,5.579494291,18.79370788,0.218173743,10.8183659,167.6857064,83.72248865,53.72832643,16.9574983,18.97382249,14.92890401,23.42079471,19.40788086,8.466564145,0.383471426,0.605134077,4.827188637,14.77009263,27.14788135,51.48774731,72.04206859,128.632606,132.997857,136.3610982,221.2246532,263.7992233,335.5065783,359.1391982,384.4076445,451.6889723,593.0941181,682.1929084]
        examples_dict['41_cmpds']['no_cmpds'] = 41
        examples_dict['41_cmpds']['min_no_samples_per_bin'] = 20
        examples_dict['41_cmpds']['expected_no_bins'] = 2
        #############################

        for eg in examples_dict.keys():
            print(f'Running {eg} in {report_name_of_function_where_this_is_called()}')

            assert examples_dict[eg]['no_cmpds'] == len(examples_dict[eg]['y_true'])
            assert examples_dict[eg]['no_cmpds'] == len(examples_dict[eg]['y_pred'])
            assert examples_dict[eg]['no_cmpds'] == len(examples_dict[eg]['estimated_variance'])

            y_true = examples_dict[eg]['y_true']
            y_pred = examples_dict[eg]['y_pred']
            var_estimated = examples_dict[eg]['estimated_variance']

            data = organize_variables_as_required_for_ENCE(y_true, y_pred, var_estimated)

            bins = bin_data_for_ENCE(data,min_no_samples_per_bin=examples_dict[eg]['min_no_samples_per_bin'])

            assert isinstance(bins,list)
            assert examples_dict[eg]['expected_no_bins'] == len(bins)
            for BIN in bins:
                assert isinstance(BIN,np.ndarray)
                assert 3 == BIN.shape[1]
                print(f'no. samples in this bin = {BIN.shape[0]}')
                if examples_dict[eg]['no_cmpds'] >= examples_dict[eg]['min_no_samples_per_bin']:
                    assert examples_dict[eg]['min_no_samples_per_bin'] <= BIN.shape[0]
                else:
                    assert examples_dict[eg]['no_cmpds'] == BIN.shape[0]


    def test_get_precursor_info_for_ENCE(self):

        examples_dict = defaultdict(dict)
        ##############################
        examples_dict['40_cmpds']['y_true'] = [1.5,1.8,4.2,3.8,4.9,3.7,5.6,8.9,8.7,2.3,4.2,0.5,2.9,11.9,9.1,6.2,3.81,3.79,3.4,4.8,3.738416237,2.567501089,0.668148014,-0.639995051,-1.921008667,-3.268343681,-4.776955907,-6.599146981,-7.856838111,-9.452347361,-10.52231359,-11.58469748,-12.72269451,-14.57062656,-15.84610038,-17.27026746,-19.00891467,-20.98244982,-22.44643775,-23.60451254]
        examples_dict['40_cmpds']['y_pred'] = [1.531094177,2.04099802,4.557883766,4.139333809,5.465407798,3.814994406,5.749089749,9.877191696,10.30589635,2.332727598,4.310342403,0.422800585,3.272572201,12.92028741,9.115967329,7.303709706,4.117752416,4.326521168,3.833264737,4.800017326,4.373933722,2.874686226,0.587295354,-0.790414425,-2.217289321,-3.890144906,-5.233617029,-7.182248061,-8.523878383,-11.38445113,-11.53476957,-11.6883688,-14.91278059,-16.28678725,-18.33956271,-18.97449796,-19.64410457,-21.26880675,-24.39275463,-26.11893217]
        examples_dict['40_cmpds']['estimated_variance'] = [2.372231966,4.336364821,20.9479272,17.22973884,29.96617983,14.91268639,33.43824155,98.45370326,106.4092907,5.579494291,18.79370788,0.218173743,10.8183659,167.6857064,83.72248865,53.72832643,16.9574983,18.97382249,14.92890401,23.42079471,19.40788086,8.466564145,0.383471426,0.605134077,4.827188637,14.77009263,27.14788135,51.48774731,72.04206859,128.632606,132.997857,136.3610982,221.2246532,263.7992233,335.5065783,359.1391982,384.4076445,451.6889723,593.0941181,682.1929084]
        #From manual sorting, hence binning, and calculations in Excel:
        examples_dict['40_cmpds']['expected_array_of_RMSE_values_for_each_y_pred_bin'] = np.array([0.335720621,1.399076282])
        examples_dict['40_cmpds']['expected_array_of_root_mean_squared_estimated_variance_in_prediction'] = np.array([3.500699385,14.80874507])
        examples_dict['40_cmpds']['no_cmpds'] = 40
        examples_dict['40_cmpds']['min_no_samples_per_bin'] = 20
        #############################

        for eg in examples_dict.keys():
            
            print(f'Running {eg} in {report_name_of_function_where_this_is_called()}')

            assert examples_dict[eg]['no_cmpds'] == len(examples_dict[eg]['y_true'])
            assert examples_dict[eg]['no_cmpds'] == len(examples_dict[eg]['y_pred'])
            assert examples_dict[eg]['no_cmpds'] == len(examples_dict[eg]['estimated_variance'])

            y_true = examples_dict[eg]['y_true']
            y_pred = examples_dict[eg]['y_pred']
            var_estimated = examples_dict[eg]['estimated_variance']

            estimated_variance = var_estimated

            array_of_RMSE_values_for_each_y_pred_bin,array_of_root_mean_squared_estimated_variance_in_prediction = get_precursor_info_for_ENCE(y_true,
                   y_pred,
                   estimated_variance,min_no_samples_per_bin=examples_dict[eg]['min_no_samples_per_bin'])
            

            np.testing.assert_allclose(examples_dict[eg]['expected_array_of_RMSE_values_for_each_y_pred_bin'],array_of_RMSE_values_for_each_y_pred_bin)

            np.testing.assert_allclose(examples_dict[eg]['expected_array_of_root_mean_squared_estimated_variance_in_prediction'],array_of_root_mean_squared_estimated_variance_in_prediction)
        
    def test_compute_ENCE(self):

        examples_dict = defaultdict(dict)
        ##############################
        examples_dict['40_cmpds']['y_true'] = [1.5,1.8,4.2,3.8,4.9,3.7,5.6,8.9,8.7,2.3,4.2,0.5,2.9,11.9,9.1,6.2,3.81,3.79,3.4,4.8,3.738416237,2.567501089,0.668148014,-0.639995051,-1.921008667,-3.268343681,-4.776955907,-6.599146981,-7.856838111,-9.452347361,-10.52231359,-11.58469748,-12.72269451,-14.57062656,-15.84610038,-17.27026746,-19.00891467,-20.98244982,-22.44643775,-23.60451254]
        examples_dict['40_cmpds']['y_pred'] = [1.531094177,2.04099802,4.557883766,4.139333809,5.465407798,3.814994406,5.749089749,9.877191696,10.30589635,2.332727598,4.310342403,0.422800585,3.272572201,12.92028741,9.115967329,7.303709706,4.117752416,4.326521168,3.833264737,4.800017326,4.373933722,2.874686226,0.587295354,-0.790414425,-2.217289321,-3.890144906,-5.233617029,-7.182248061,-8.523878383,-11.38445113,-11.53476957,-11.6883688,-14.91278059,-16.28678725,-18.33956271,-18.97449796,-19.64410457,-21.26880675,-24.39275463,-26.11893217]
        examples_dict['40_cmpds']['estimated_variance'] = [2.372231966,4.336364821,20.9479272,17.22973884,29.96617983,14.91268639,33.43824155,98.45370326,106.4092907,5.579494291,18.79370788,0.218173743,10.8183659,167.6857064,83.72248865,53.72832643,16.9574983,18.97382249,14.92890401,23.42079471,19.40788086,8.466564145,0.383471426,0.605134077,4.827188637,14.77009263,27.14788135,51.48774731,72.04206859,128.632606,132.997857,136.3610982,221.2246532,263.7992233,335.5065783,359.1391982,384.4076445,451.6889723,593.0941181,682.1929084]
        #From manual sorting, hence binning, and calculations in Excel:
        examples_dict['40_cmpds']['expected_ENCE'] = 0.904811315
        examples_dict['40_cmpds']['no_cmpds'] = 40
        examples_dict['40_cmpds']['min_no_samples_per_bin'] = 20
        #############################

        for eg in examples_dict.keys():
            
            print(f'Running {eg} in {report_name_of_function_where_this_is_called()}')

            assert examples_dict[eg]['no_cmpds'] == len(examples_dict[eg]['y_true'])
            assert examples_dict[eg]['no_cmpds'] == len(examples_dict[eg]['y_pred'])
            assert examples_dict[eg]['no_cmpds'] == len(examples_dict[eg]['estimated_variance'])

            y_true = examples_dict[eg]['y_true']
            y_pred = examples_dict[eg]['y_pred']
            var_estimated = examples_dict[eg]['estimated_variance']

            ENCE = compute_ENCE(y_true,y_pred,estimated_variance=var_estimated,min_no_samples_per_bin=examples_dict[eg]['min_no_samples_per_bin'])

            assert ENCE == approx(examples_dict[eg]['expected_ENCE'])
    
    def test_getPercentageSigLevel(self):
        from modelsADuncertaintyPkg.qsar_eval.reg_Uncertainty_metrics import getPercentageSigLevel
        
        assert 32 == getPercentageSigLevel(0.32)
        assert 32 == getPercentageSigLevel(0.319111)

    def construct_toy_sig_level_2_prediction_intervals_for_tests(self,y_pred,y_experi,no_test_instances,sig_level_2_half_pred_interval_sizes):
        sig_level_2_prediction_intervals = {}

        for sig_level in sig_level_2_half_pred_interval_sizes.keys():
            half_pred_interval_sizes = sig_level_2_half_pred_interval_sizes[sig_level]
            
            assert len(y_pred) == len(half_pred_interval_sizes)
            assert len(y_pred) == len(y_experi)
            assert no_test_instances == len(y_pred)
            
            #==================
            #Construct intervals based upon def pred_ints_v1(...):
            intervals = np.zeros((no_test_instances, 2))
            for k in range(0,no_test_instances):

                intervals[k, 0] = y_pred[k] - (half_pred_interval_sizes[k])
                intervals[k, 1] = y_pred[k] + (half_pred_interval_sizes[k])
            #===================

            sig_level_2_prediction_intervals[sig_level] = intervals

        return sig_level_2_prediction_intervals

    def test_getAllerrorRate_s(self):
        from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import getAllerrorRate_s

        y_pred = [1.5,2.8,3.4,1.0]
        y_experi = [0.0,3.0,3.3,5.0]
        no_test_instances = len(y_pred)

        sig_level_2_half_pred_interval_sizes = {}
        sig_level_2_half_pred_interval_sizes[0.0] = [0.0,0.0,0.0,0.0]
        sig_level_2_half_pred_interval_sizes[0.1] = [0.0,0.0,0.0,0.0]
        sig_level_2_half_pred_interval_sizes[0.2] = [0.0,0.0,0.0,0.0]
        sig_level_2_half_pred_interval_sizes[0.32] = [0.0,0.0,0.0,0.0]
        sig_level_2_half_pred_interval_sizes[0.4] = [0.0,0.0,0.0,0.0]
        sig_level_2_half_pred_interval_sizes[0.5] = [0.0,0.0,0.0,0.0]
        sig_level_2_half_pred_interval_sizes[0.6] = [0.0,0.0,0.0,0.0]
        sig_level_2_half_pred_interval_sizes[0.7] = [0.0,0.0,0.0,0.0]
        sig_level_2_half_pred_interval_sizes[0.8] = [0.5,0.6,0.2,3.0]
        sig_level_2_half_pred_interval_sizes[0.9] = [10.0]*no_test_instances
        

        sig_level_2_prediction_intervals = self.construct_toy_sig_level_2_prediction_intervals_for_tests(y_pred,y_experi,no_test_instances,sig_level_2_half_pred_interval_sizes)

        errorRate_s = getAllerrorRate_s(sig_level_2_prediction_intervals,y_test=y_experi)

        assert np.array_equal(errorRate_s,np.array([1.0]*8+[0.5,0.0]))

    def test_compute_ECE(self):
        from modelsADuncertaintyPkg.qsar_eval.reg_Uncertainty_metrics import compute_ECE
        
        error_rate_s = np.array([1.0]*8+[0.5,0.0])

        ECE = compute_ECE(error_rate_s,sig_levels=[0.0,0.1,0.2,0.32,0.4,0.5,0.6,0.7,0.8,0.9])


        assert ECE == approx(sum([0.9,0.8,0.68,0.6,0.5,0.4,0.3]+[0.3,0.9,1.0])/10)

    def test_consistency_of_different_validity_and_efficieny_calculations(self):
        from modelsADuncertaintyPkg.qsar_eval.reg_Uncertainty_metrics import compute_validity_efficiency
        from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import getIntervalsWidthOfInterest
        from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import index2SigLevel,sigLevels
        from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import getAllerrorRate_s
        from modelsADuncertaintyPkg.utils.basic_utils import getKeyFromValue
        from modelsADuncertaintyPkg.qsar_eval.perf_measure import Efficiency,ErrorRate

        y_pred = [1.5,2.8,3.4,1.0,50.0]
        y_experi = [0.0,3.0,3.3,5.0,2.0]
        no_test_instances = len(y_pred)

        sig_level_of_interest = 0.32

        sig_level_2_half_pred_interval_sizes = {}
        sig_level_2_half_pred_interval_sizes[sig_level_of_interest] = [0.5,0.6,0.2,3.0,0.1]
        

        sig_level_2_prediction_intervals = self.construct_toy_sig_level_2_prediction_intervals_for_tests(y_pred,y_experi,no_test_instances,sig_level_2_half_pred_interval_sizes)

        widths_for_intervals_of_interest = getIntervalsWidthOfInterest(sig_level_2_prediction_intervals,sig_level_of_interest)

        errorRate_s = getAllerrorRate_s(sig_level_2_prediction_intervals,y_experi,index2SigLevel,sigLevels)

        validity, efficiency = compute_validity_efficiency(error_rate=errorRate_s[getKeyFromValue(index2SigLevel,sig_level_of_interest)], intervals_width=widths_for_intervals_of_interest)

        intervals = sig_level_2_prediction_intervals[sig_level_of_interest]

        alt_efficiency = Efficiency(intervals)

        alt_validity = (1-ErrorRate(intervals,y_experi))

        assert approx(validity) == alt_validity
        assert approx(efficiency) == alt_efficiency 

        assert approx(validity) == 0.4
        assert approx(efficiency) == 1.76

    def test_estimate_residuals_for_test_cmpds(self):
        from modelsADuncertaintyPkg.qsar_reg_uncertainty.RegressionICP import estimate_residuals_for_test_cmpds

        confScore = np.array([1.0,2.0,0.5,0.6,0.9,1.2,1.3,0.2,0.91,0.98]) 
        assert 10 == len(confScore) #This makes figuring out what the expected answers are easier.
        #after sorting: confScore = np.array([0.2 , 0.5 , 0.6 , 0.9 , 0.91, 0.98, 1.  , 1.2 , 1.3 , 2.  ])

        #####################################
        #Svensson et al. (2018) [https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00054] cites Norinder et al. (2014) for a practical description of how non-conformity scores are selected to define prediction intervals
        #Norinder et al. (2014) [https://pubs.acs.org/doi/full/10.1021/ci5001168] indicates (see equation (3) and surrounding text) that a test set non-conformity value should be chosen such that the fraction in the calibration set (or, in contrast to the text, slightly less than this fraction if the + 1 in the denominator of equation (3) is considered) with non-conformity values >= the test set selected value EXCEEDS epsilon (the significance level)
        #Papdopoulos et al. (2011) [https://doi.org/10.1613/jair.3198] explicitly describes conformal regression and the text on p.823-824 (pdf pages 9-10 of 26) indicates that we should select the largest calibration set non-conformity value [alpha_test], out of all possible alpha_i values, such that p = ([number of calibration set compound non-conformity scores () >= alpha_test] + 1) divided by (1+number of calibration set compounds) EXCEEDS the significance level 
        #For example, if significance level = 0.2, we would start with the largest alpha_i = 2.0 and find that only one alpha_i was >= this value, so p = (1+1)/(10+1) = 0.18, which is not > 0.2. So, we would need to select the next largest alpha_i as alpha_test = 1.3, for which p = (2+1)/(10+1) = 0.27, which is > 0.2. Hence, we should select alpha_test = 1.3.
        #Gauraha et al. (2021), "Synergy Conformal Prediction for Regression" notation indicates the same approach as per Papdopoulos et al. (2011) is applied.
        ####################################

        examples_dict = defaultdict(dict)
        ##################################
        examples_dict[1]['lambda_test'] = np.array([1.0,1.0]) #This makes the expected_q the same as the expected non-conformity score selecte for the test set compounds based upon the calibration set non-conformity scores and epsilon
        examples_dict[1]['epsilon'] = 0.2
        examples_dict[1]['expected_q'] = np.array([1.3]*2)
        ###################################
        ##################################
        examples_dict[2]['lambda_test'] = np.array([1.0,1.0]) 
        examples_dict[2]['epsilon'] = 0.1
        examples_dict[2]['expected_q'] = np.array([2.0]*2)
        ###################################
        #################################
        examples_dict[3]['lambda_test'] = np.array([1.0,1.0]) 
        examples_dict[3]['epsilon'] = 0.05
        examples_dict[3]['expected_q'] = np.array([2.0]*2)
        ###################################
        #################################
        #Designed to check that lamda scaling works as expected
        examples_dict[4]['lambda_test'] = np.array([2.0,1.0]) 
        examples_dict[4]['epsilon'] = 0.05
        examples_dict[4]['expected_q'] = np.array([4.0,2.0])
        ###################################
        #################################
        examples_dict[5]['lambda_test'] = np.array([1.0,1.0]) 
        examples_dict[5]['epsilon'] = 0.0 #Test limits!
        examples_dict[5]['expected_q'] = np.array([2.0]*2)
        ###################################
        #################################
        examples_dict[6]['lambda_test'] = np.array([1.0,1.0]) 
        examples_dict[6]['epsilon'] = 1.0 #Test limits!
        examples_dict[6]['expected_q'] = np.array([0.2]*2)
        ###################################

        for eg in examples_dict.keys():
            print(f'Checking example {eg} for {report_name_of_function_where_this_is_called()}')
            lamda_test = examples_dict[eg]['lambda_test']
            epsilon = examples_dict[eg]['epsilon']
            expected_q = examples_dict[eg]['expected_q']

            q = estimate_residuals_for_test_cmpds(confScore,lamda_test,epsilon)

            np.testing.assert_array_almost_equal(expected_q,q)

    
    def test_compute_residuals(self):
        from modelsADuncertaintyPkg.qsar_eval.reg_perf_pred_stats import compute_residuals

        y_true = [1.0,2.0,3.0,1.5,0.8]
        y_pred = [0.5,2.0,3.1,1.8,0.2]
        expected_residuals = [0.5,0.0,0.1,0.3,0.6]

        assert approx(expected_residuals) == compute_residuals(y_true,y_pred).tolist()
    
    def test_compute_Spearman_rank_correlation_between_intervals_width_and_residuals(self):
        from modelsADuncertaintyPkg.qsar_eval.reg_Uncertainty_metrics import compute_Spearman_rank_correlation_between_intervals_width_and_residuals
        from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import getIntervalsWidthOfInterest

        y_true = [1.0,2.0,3.0,1.5,0.8]
        y_pred = [0.5,2.0,3.1,1.8,0.2]

        y_experi = y_true
        no_test_instances = len(y_true)

        sig_level_of_interest = 0.32

        sig_level_2_half_pred_interval_sizes = {}
        sig_level_2_half_pred_interval_sizes[sig_level_of_interest] = [0.5,0.6,0.2,3.0,0.1]

        expected_residuals = [0.5,0.0,0.1,0.3,0.6]
        expected_spearman_coeff = SpearmanCoeff(e=expected_residuals,p=sig_level_2_half_pred_interval_sizes[sig_level_of_interest])
        

        sig_level_2_prediction_intervals = self.construct_toy_sig_level_2_prediction_intervals_for_tests(y_pred,y_experi,no_test_instances,sig_level_2_half_pred_interval_sizes)

        widths_for_intervals_of_interest = getIntervalsWidthOfInterest(sig_level_2_prediction_intervals,sig_level_of_interest)

        
        spearman_coeff = compute_Spearman_rank_correlation_between_intervals_width_and_residuals(y_true,y_pred,widths_for_intervals_of_interest)

        assert approx(expected_spearman_coeff) == spearman_coeff
    
    def test_get_experi_class_1_probs(self):

        res = get_experi_class_1_probs(class_1_probs_in_order=[0.01,0.02,0.48,0.059],experi_class_labels=[0.0,0.0,0.0,1.0],delta_prob=0.05)

        assert [(1/3),(1/3),0.0,(1/3)] == res,res

    def test_how_ad_p_value_calculation_handles_missing_metrics_for_one_fold_only(self):
        from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import workflowForAFamilyOfRawResults
        from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict
        from modelsADuncertaintyPkg.utils.ML_utils import compute_pred_intervals

        ##########################
        #R2: according to our code [coeffOfDetermination(,...check_consistency=True)], this be treated as None (missing) if there is only 1 value in a subset. [Zero values: ValueError: Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required.]
        #See comments inside def workflowForAFamilyOfRawResult(...) for an explantion of the following structure.
        example_dict_for_all_test_sets_of_raw_subset_results = neverEndingDefaultDict()
        test_set_plus_methods_name = 'TestSet1_M2'
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results = neverEndingDefaultDict()
        #------------
        fold_and_or_model_seed_combination = 'F0_S1'
        dict_of_raw_subset_results = neverEndingDefaultDict()
        #=
        subset='Inside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [1,2,3,4,5]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [1.5,1.8,1.9,2.1,2.2]
        y_pred = [3.8,1.6,1.7,2.2,2.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5,0.6,0.8,0.9,1.0]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        subset = 'Outside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [10,20,30,40,50]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [2.5,2.8,2.9,3.1,3.2]
        y_pred = [3.8,1.6,1.7,2.2,2.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5,0.6,0.8,0.9,1.0]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results
        #--------------
        #------------
        fold_and_or_model_seed_combination = 'F1_S1'
        dict_of_raw_subset_results = neverEndingDefaultDict()
        #=
        subset='Inside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [1,2,3,4,5]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [0.5,1.8,1.9,2.1,2.2]
        y_pred = [3.8,1.6,1.7,2.2,2.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5,0.6,0.8,0.9,1.0]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        subset = 'Outside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [10,20,30,40,50]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [2.8,2.8,2.9,3.1,3.2]
        y_pred = [3.8,1.6,1.7,2.2,2.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5,0.6,0.8,0.9,1.0]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results
        #--------------
        #------------
        fold_and_or_model_seed_combination = 'F2_S1'
        dict_of_raw_subset_results = neverEndingDefaultDict()
        #=
        subset='Inside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [1,2,3,4,5]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [1.5,1.8,1.9,2.1,2.2]
        y_pred = [3.8,1.6,1.7,2.2,2.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5,0.6,0.8,0.9,1.0]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        subset = 'Outside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [50]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [1.1]
        y_pred = [1.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results
        #--------------
        example_dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name] = dict_for_all_folds_and_or_model_seeds_of_raw_subset_results
        ####################################
        p_vals_table = 'one_tail_tmp_PVals.csv'
        

        metric = 'R2'
        ####################################

        workflowForAFamilyOfRawResults(out_dir=os.getcwd(),dict_for_all_test_sets_of_raw_subset_results=example_dict_for_all_test_sets_of_raw_subset_results,subset_1_name='Inside',subset_2_name='Outside',
                                   type_of_modelling='regression',no_rand_splits=100,strat_rand_split_y_name='strat_y',rand_seed=42,
                                   metrics_of_interest=[metric],x_lab='Groups',y_lab='Shift-Metric',legend_lab='Split Basis',sig_level_perc=5,
                                   one_sided_sig_test=True,metric_to_expected_sub_1_minus_sub_2_sign={'R2':1},p_vals_table=p_vals_table,
                                   adjusted_p_vals_table_name=None,results_are_obtained_over_multiple_folds_and_or_seeds=True,create_plots=False,
                                   debug=True, scenarios_with_errors=[])
        
        p_vals_df = pd.read_csv(p_vals_table)

        #####################
        #The only thing we are checking here is that a missing metric for one subset of one fold does not mean that a p-value is not calculated - this was observed manually before writing the next lines of code!
        #11/03/25: When the p-value aggregation function was fixed, following consultation of the statistics literature, the previously observed p-values were increased by a factor of 2, then changed again following fixing a typo in the input.
        expected_p_vals_df = pd.DataFrame({'Scenario':[test_set_plus_methods_name],'Metric':[metric],'Shift-Metric P-value':[0.31]})
        #11/03/25: Also, dummy values were inserted into the additional p-values column which is now expected as part of the AD p-values output, prior to manually checking internal consistency between the generated p-values and the other results and replacing the dummy values for the next run of this test.
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'all_p_vals_str',['0.28;0.03;nan'],allow_duplicates=False)
        #11/03/25: In addition, the following extra column values were populated based upon applying sklean.metrics.r2_score for the 'F0_S1' and 'F1_S1' fold-seed combinations above:
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'all_shift_metrics_str',['2.6666666666666643;35.882456140350804;None'],allow_duplicates=False)
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'average_shift_metric_val',[19.274561403508734],allow_duplicates=False)
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'average_shift_metric_val_has_wrong_sign',[False],allow_duplicates=False)

        assert_frame_equal(p_vals_df,expected_p_vals_df)
        #####################

        os.remove(p_vals_table)
        
    
    def test_how_ad_p_value_calculation_handles_missing_metrics_for_all_folds(self):
        from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import workflowForAFamilyOfRawResults
        from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict
        from modelsADuncertaintyPkg.utils.ML_utils import compute_pred_intervals

        ##########################
        #R2: according to our code [coeffOfDetermination(,...check_consistency=True)], this be treated as None (missing) if there is only 1 value in a subset. [Zero values: ValueError: Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required.]
        #See comments inside def workflowForAFamilyOfRawResult(...) for an explantion of the following structure.
        example_dict_for_all_test_sets_of_raw_subset_results = neverEndingDefaultDict()
        test_set_plus_methods_name = 'TestSet1_M2'
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results = neverEndingDefaultDict()
        #------------
        fold_and_or_model_seed_combination = 'F0_S1'
        dict_of_raw_subset_results = neverEndingDefaultDict()
        #=
        subset='Inside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [1,2,3,4,5]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [1.5,1.8,1.9,2.1,2.2]
        y_pred = [3.8,1.6,1.7,2.2,2.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5,0.6,0.8,0.9,1.0]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        subset = 'Outside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [50]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [1.1]
        y_pred = [1.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results
        #--------------
        #------------
        fold_and_or_model_seed_combination = 'F1_S1'
        dict_of_raw_subset_results = neverEndingDefaultDict()
        #=
        subset='Inside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [1,2,3,4,5]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [0.5,1.8,1.9,2.1,2.2]
        y_pred = [3.8,1.6,1.7,2.2,2.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5,0.6,0.8,0.9,1.0]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        subset = 'Outside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [50]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [1.1]
        y_pred = [1.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results
        #--------------
        #------------
        fold_and_or_model_seed_combination = 'F2_S1'
        dict_of_raw_subset_results = neverEndingDefaultDict()
        #=
        subset='Inside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [1,2,3,4,5]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [1.5,1.8,1.9,2.1,2.2]
        y_pred = [3.8,1.6,1.7,2.2,2.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5,0.6,0.8,0.9,1.0]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        subset = 'Outside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [50]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [1.1]
        y_pred = [1.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results
        #--------------
        example_dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name] = dict_for_all_folds_and_or_model_seeds_of_raw_subset_results
        ####################################
        p_vals_table = 'one_tail_tmp_PVals.csv'
        

        metric = 'R2'
        ####################################

        workflowForAFamilyOfRawResults(out_dir=os.getcwd(),dict_for_all_test_sets_of_raw_subset_results=example_dict_for_all_test_sets_of_raw_subset_results,subset_1_name='Inside',subset_2_name='Outside',
                                   type_of_modelling='regression',no_rand_splits=100,strat_rand_split_y_name='strat_y',rand_seed=42,
                                   metrics_of_interest=[metric],x_lab='Groups',y_lab='Shift-Metric',legend_lab='Split Basis',sig_level_perc=5,
                                   one_sided_sig_test=True,metric_to_expected_sub_1_minus_sub_2_sign={'R2':1},p_vals_table=p_vals_table,
                                   adjusted_p_vals_table_name=None,results_are_obtained_over_multiple_folds_and_or_seeds=True,create_plots=False,
                                   debug=False, scenarios_with_errors=[])
        
        p_vals_df = pd.read_csv(p_vals_table)

        #####################
        #11/03/25: copied and adapted from updated checking lines for test_how_ad_p_value_calculation_handles_missing_metrics_for_one_fold_only:
        
        expected_p_vals_df = pd.DataFrame({'Scenario':[test_set_plus_methods_name],'Metric':[metric],'Shift-Metric P-value':[np.nan]})
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'all_p_vals_str',['nan;nan;nan'],allow_duplicates=False)
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'all_shift_metrics_str',['None;None;None'],allow_duplicates=False)
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'average_shift_metric_val',[np.nan],allow_duplicates=False)
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'average_shift_metric_val_has_wrong_sign',[np.nan],allow_duplicates=False)

        assert_frame_equal(p_vals_df,expected_p_vals_df)
        #####################

        os.remove(p_vals_table)
        
    
    def test_plotPredPlusEstimatedErrorBarsVsExperi(self):
        from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import plotPredPlusEstimatedErrorBarsVsExperi

        y_true = [1.0,2.0,3.0,1.5,0.8]
        y_pred = [0.5,2.0,3.1,1.8,0.2]

        y_experi = y_true
        no_test_instances = len(y_true)

        sig_level_of_interest = 0.32

        sig_level_2_half_pred_interval_sizes = {}
        sig_level_2_half_pred_interval_sizes[sig_level_of_interest] = [0.5,0.6,0.2,3.0,0.1]

        sig_level_2_prediction_intervals = self.construct_toy_sig_level_2_prediction_intervals_for_tests(y_pred,y_experi,no_test_instances,sig_level_2_half_pred_interval_sizes)

        y_test = y_true
        testPred = y_pred

        plot_file_name = 'test_reg_preds_plus_PIs_vs_experi_plot.tiff'
        plot_title = 'Test example'
        
        plotPredPlusEstimatedErrorBarsVsExperi(y_test, testPred,sig_level_2_prediction_intervals,sig_level_of_interest,plot_file_name,plot_title,assume_PIs_are_symmetric=True,experi_err_sizes=None,offset=0.2)
    
    def get_input_where_one_metric_cannot_be_computed_for_checking_ad_p_values_calculation_with_or_without_set_p_to_nan_if_metric_cannot_be_computed_eq_True(self):
        ##########################
        #The following example has been contrived such that R2 = None (constant experimental values) for one subset, but RMSE can still be computed:
        metric_to_expected_sub_1_minus_sub_2_sign={'R2':1,'RMSE':-1}
        metrics_of_interest = list(metric_to_expected_sub_1_minus_sub_2_sign.keys())

        ##########################
        example_dict_for_all_test_sets_of_raw_subset_results = neverEndingDefaultDict()
        test_set_plus_methods_name = 'TestSet1_M2'
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results = neverEndingDefaultDict()
        #------------
        fold_and_or_model_seed_combination = 'F0_S1'
        dict_of_raw_subset_results = neverEndingDefaultDict()
        #=
        subset='Inside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [1,2,3,4,5]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [1.5,1.8,1.9,2.1,2.2]
        y_pred = [3.8,1.6,1.7,2.2,2.3]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5,0.6,0.8,0.9,1.0]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        subset = 'Outside'
        dict_of_raw_subset_results[subset]['subset_test_ids'] = [50,51,52,53,54]
        dict_of_raw_subset_results[subset]['subset_test_y'] = [1.1,1.1,1.1,1.1,1.1]
        y_pred = [1.3,1.8,1.9,1.0,32.0]
        dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
        half_pred_interval_sizes = [0.5,0.6,0.5,0.5,0.5]
        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {0.32:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
        #=
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results
        #--------------

        #--------------
        example_dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name] = dict_for_all_folds_and_or_model_seeds_of_raw_subset_results
        ####################################
        p_vals_table = 'one_tail_tmp_PVals.csv'
        

        return test_set_plus_methods_name,p_vals_table,metrics_of_interest,metric_to_expected_sub_1_minus_sub_2_sign,example_dict_for_all_test_sets_of_raw_subset_results
    
    def test_assess_stat_sig_shift_metrics_workflowForAFamilyOfRawResults_set_p_to_nan_if_metric_cannot_be_computed_eq_True(self):
        from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import workflowForAFamilyOfRawResults

        from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict
        from modelsADuncertaintyPkg.utils.ML_utils import compute_pred_intervals

        
        test_set_plus_methods_name,p_vals_table,metrics_of_interest,metric_to_expected_sub_1_minus_sub_2_sign,example_dict_for_all_test_sets_of_raw_subset_results = self.get_input_where_one_metric_cannot_be_computed_for_checking_ad_p_values_calculation_with_or_without_set_p_to_nan_if_metric_cannot_be_computed_eq_True()
        
        ####################################

        workflowForAFamilyOfRawResults(out_dir=os.getcwd(),dict_for_all_test_sets_of_raw_subset_results=example_dict_for_all_test_sets_of_raw_subset_results,subset_1_name='Inside',subset_2_name='Outside',
                                   type_of_modelling='regression',no_rand_splits=100,strat_rand_split_y_name='strat_y',rand_seed=42,
                                   metrics_of_interest=metrics_of_interest,x_lab='Groups',y_lab='Shift-Metric',legend_lab='Split Basis',sig_level_perc=5,
                                   one_sided_sig_test=True,metric_to_expected_sub_1_minus_sub_2_sign=metric_to_expected_sub_1_minus_sub_2_sign,p_vals_table=p_vals_table,
                                   adjusted_p_vals_table_name=None,results_are_obtained_over_multiple_folds_and_or_seeds=True,create_plots=False,
                                   debug=True, scenarios_with_errors=[],set_p_to_nan_if_metric_cannot_be_computed_for_orig_split=True)
        
        p_vals_df = pd.read_csv(p_vals_table)

        ######################
        #11/03/25: it appears that a column with string values that can be seen as nan or float values when written to a csv will otherwise be misinterpreted:
        p_vals_df = p_vals_df.astype({'all_p_vals_str':'str','all_shift_metrics_str':'str'})
        #####################

        #####################
        #11/03/25: previous raw p_value_for_rmse (0.77) assumed to be the same - one fold-seed - but rounded up to smallest out of 1.0 and twice the previous value: 
        overall_p_value_for_rmse = 1.0
        raw_p_value_for_rmse = 0.77
        p_value_for_R2 = np.nan
        expected_p_vals_df = pd.DataFrame({'Scenario':[test_set_plus_methods_name]*2,'Metric':['R2','RMSE'],'Shift-Metric P-value':[p_value_for_R2,overall_p_value_for_rmse]})
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'all_p_vals_str',[f'{p_value_for_R2}',f'{raw_p_value_for_rmse}'],allow_duplicates=False)
        #11/03/25: performed rmse calculation to enable following updates:
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'all_shift_metrics_str',['None','-12.789168705965603'],allow_duplicates=False)
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'average_shift_metric_val',[np.nan,-12.789168705965603],allow_duplicates=False)
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'average_shift_metric_val_has_wrong_sign',[np.nan,False],allow_duplicates=False)

        assert_frame_equal(p_vals_df,expected_p_vals_df)
        #####################

        os.remove(p_vals_table)
        
    
    def get_input_where_both_shift_metrics_cannot_be_computed_for_checking_ad_p_values_calculation_with_or_without_set_p_to_nan_if_metric_cannot_be_computed_eq_True(self):

        test_set_plus_methods_name,p_vals_table,metrics_of_interest,metric_to_expected_sub_1_minus_sub_2_sign,example_dict_for_all_test_sets_of_raw_subset_results = self.get_input_where_one_metric_cannot_be_computed_for_checking_ad_p_values_calculation_with_or_without_set_p_to_nan_if_metric_cannot_be_computed_eq_True()

        for fold_and_or_model_seed_combination in example_dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name].keys():
            for raw_results_type in example_dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name][fold_and_or_model_seed_combination]['Outside']:
                if not 'subset_sig_level_2_prediction_intervals' == raw_results_type:
                    example_dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name][fold_and_or_model_seed_combination]['Outside'][raw_results_type] = []
                else:
                    example_dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name][fold_and_or_model_seed_combination]['Outside'][raw_results_type] = {0.32:[]}
        
        return test_set_plus_methods_name,p_vals_table,metrics_of_interest,metric_to_expected_sub_1_minus_sub_2_sign,example_dict_for_all_test_sets_of_raw_subset_results
    

    def test_assess_stat_sig_shift_metrics_workflowForAFamilyOfRawResults_set_p_to_nan_if_metric_cannot_be_computed_eq_True_when_all_shift_metrics_of_interest_cannot_be_computed(self):
        from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import workflowForAFamilyOfRawResults

        from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict
        from modelsADuncertaintyPkg.utils.ML_utils import compute_pred_intervals

        
        test_set_plus_methods_name,p_vals_table,metrics_of_interest,metric_to_expected_sub_1_minus_sub_2_sign,example_dict_for_all_test_sets_of_raw_subset_results = self.get_input_where_both_shift_metrics_cannot_be_computed_for_checking_ad_p_values_calculation_with_or_without_set_p_to_nan_if_metric_cannot_be_computed_eq_True()
        
        ####################################

        workflowForAFamilyOfRawResults(out_dir=os.getcwd(),dict_for_all_test_sets_of_raw_subset_results=example_dict_for_all_test_sets_of_raw_subset_results,subset_1_name='Inside',subset_2_name='Outside',
                                   type_of_modelling='regression',no_rand_splits=100,strat_rand_split_y_name='strat_y',rand_seed=42,
                                   metrics_of_interest=metrics_of_interest,x_lab='Groups',y_lab='Shift-Metric',legend_lab='Split Basis',sig_level_perc=5,
                                   one_sided_sig_test=True,metric_to_expected_sub_1_minus_sub_2_sign=metric_to_expected_sub_1_minus_sub_2_sign,p_vals_table=p_vals_table,
                                   adjusted_p_vals_table_name=None,results_are_obtained_over_multiple_folds_and_or_seeds=True,create_plots=False,
                                   debug=False, scenarios_with_errors=[],set_p_to_nan_if_metric_cannot_be_computed_for_orig_split=True)
        
        p_vals_df = pd.read_csv(p_vals_table)

        ######################
        #11/03/25: it appears that a column with string values that can be seen as nan or float values when written to a csv will otherwise be misinterpreted:
        p_vals_df = p_vals_df.astype({'all_p_vals_str':'str','all_shift_metrics_str':'str'})
        #####################

        #####################
        p_value_for_rmse = np.nan
        p_value_for_R2 = np.nan
        expected_p_vals_df = pd.DataFrame({'Scenario':[test_set_plus_methods_name]*2,'Metric':['R2','RMSE'],'Shift-Metric P-value':[p_value_for_R2,p_value_for_rmse]})

        #11/03/25: added extra columns manually:
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'all_p_vals_str',['nan','nan'],allow_duplicates=False)
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'all_shift_metrics_str',['None','None'],allow_duplicates=False)
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'average_shift_metric_val',[np.nan,np.nan],allow_duplicates=False)
        expected_p_vals_df.insert(expected_p_vals_df.shape[1],'average_shift_metric_val_has_wrong_sign',[np.nan,np.nan],allow_duplicates=False)
        assert_frame_equal(p_vals_df,expected_p_vals_df)
        #####################

        os.remove(p_vals_table)
        
    
    def test_workflowForAFamilyOfRawResults_does_not_crash_with_non_default_conformal_sig_level(self):
        #Artificial sizes of prediction intervals mean that we will not currently be able to check different p-values here, as that depends upon pre-computed prediction interval sizes which do depend upon the sig-levels if they were computed properly!
        from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import workflowForAFamilyOfRawResults

        for sig_level in ['default',0.10]:

            if 'default' == sig_level:
                numeric_sig_level = 0.32
            elif 0.10 == sig_level:
                numeric_sig_level = sig_level
            else:
                raise Exception(f'Unexpected sig_level={sig_level}')

            example_dict_for_all_test_sets_of_raw_subset_results = neverEndingDefaultDict()
            test_set_plus_methods_name = 'TestSet1_M2'
            dict_for_all_folds_and_or_model_seeds_of_raw_subset_results = neverEndingDefaultDict()
            #------------
            fold_and_or_model_seed_combination = 'F0_S1'
            dict_of_raw_subset_results = neverEndingDefaultDict()
            #=
            subset='Inside'
            dict_of_raw_subset_results[subset]['subset_test_ids'] = [1,2,3,4,5]
            dict_of_raw_subset_results[subset]['subset_test_y'] = [1.5,1.8,1.9,2.1,2.2]
            y_pred = [3.8,1.6,1.7,2.2,2.3]
            dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
            half_pred_interval_sizes = [0.5,0.6,0.8,0.9,1.0]
            dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {numeric_sig_level:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
            ##################
            #Debug:
            print(f"sig_level = {sig_level} Inside subset_sig_level_2_prediction_intervals = {dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals']}")
            ##################
            #=
            subset = 'Outside'
            dict_of_raw_subset_results[subset]['subset_test_ids'] = [10,20,30,40,50]
            dict_of_raw_subset_results[subset]['subset_test_y'] = [2.5,2.8,2.9,3.1,3.2]
            y_pred = [3.8,1.6,1.7,2.2,2.3]
            dict_of_raw_subset_results[subset]['subset_test_predictions'] = y_pred
            half_pred_interval_sizes = [0.5,0.6,0.8,0.9,1.0]
            dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {numeric_sig_level:compute_pred_intervals(y_pred,half_pred_interval_sizes)}
            #=
            dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results
            #--------------
            example_dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name] = dict_for_all_folds_and_or_model_seeds_of_raw_subset_results

            ####################################
            p_vals_table = f'one_tail_tmp_PVals_{sig_level}.csv'
            

            metric = 'Efficiency'
            ####################################

            if 0.10 == sig_level:

                workflowForAFamilyOfRawResults(out_dir=os.getcwd(),dict_for_all_test_sets_of_raw_subset_results=example_dict_for_all_test_sets_of_raw_subset_results,subset_1_name='Inside',subset_2_name='Outside',
                                        type_of_modelling='regression',no_rand_splits=100,strat_rand_split_y_name='strat_y',rand_seed=42,
                                        metrics_of_interest=[metric],x_lab='Groups',y_lab='Shift-Metric',legend_lab='Split Basis',sig_level_perc=5,
                                        one_sided_sig_test=True,metric_to_expected_sub_1_minus_sub_2_sign={'Efficiency':1},p_vals_table=p_vals_table,
                                        adjusted_p_vals_table_name=None,results_are_obtained_over_multiple_folds_and_or_seeds=True,create_plots=False,
                                        debug=False, scenarios_with_errors=[],conformal_sig_level=sig_level)
            elif 'default' == sig_level:
                workflowForAFamilyOfRawResults(out_dir=os.getcwd(),dict_for_all_test_sets_of_raw_subset_results=example_dict_for_all_test_sets_of_raw_subset_results,subset_1_name='Inside',subset_2_name='Outside',
                                        type_of_modelling='regression',no_rand_splits=100,strat_rand_split_y_name='strat_y',rand_seed=42,
                                        metrics_of_interest=[metric],x_lab='Groups',y_lab='Shift-Metric',legend_lab='Split Basis',sig_level_perc=5,
                                        one_sided_sig_test=True,metric_to_expected_sub_1_minus_sub_2_sign={'Efficiency':1},p_vals_table=p_vals_table,
                                        adjusted_p_vals_table_name=None,results_are_obtained_over_multiple_folds_and_or_seeds=True,create_plots=False,
                                        debug=False, scenarios_with_errors=[])
            else:
                raise Exception(f'Unexpected sig_level={sig_level}')

            
            

            os.remove(p_vals_table)
            

    def inputs_for_test_workflowForAFamilyOfRawResults_prevents_one_tail_stat_sig_p_vals_if_have_wrong_average_sign_by_default(self):
        #===========================
        example_dict_for_all_test_sets_of_raw_subset_results = neverEndingDefaultDict()
        test_set_plus_methods_name = 'TestSet1_M2'
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results = neverEndingDefaultDict()
        #############################
        #Real example:
        fold_and_or_model_seed_combination='F0_42'
        dict_of_raw_subset_results = neverEndingDefaultDict()
        dict_of_raw_subset_results['Inside']['subset_test_ids'] = [33, 37, 77, 86, 101, 107, 108, 109, 163, 169, 179]
        dict_of_raw_subset_results['Inside']['subset_test_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
        dict_of_raw_subset_results['Inside']['subset_probs_for_class_1'] = [0.8974358974358974, 0.72, 0.9395973154362415, 0.9622641509433963, 0.8181818181818182, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.27999999999999997, 0.27999999999999997, 0.9473684210526315]
        dict_of_raw_subset_results['Inside']['subset_predicted_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
        dict_of_raw_subset_results['Outside']['subset_test_ids'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273]
        dict_of_raw_subset_results['Outside']['subset_test_y'] = [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        dict_of_raw_subset_results['Outside']['subset_probs_for_class_1'] = [0.6875, 0.3478260869565218, 0.6875, 0.3076923076923077, 0.4166666666666667, 0.8181818181818182, 0.7741935483870966, 0.6875, 0.27999999999999997, 0.27999999999999997, 0.8181818181818182, 0.8181818181818182, 0.8181818181818182, 0.8424507658643327, 0.3478260869565218, 0.3478260869565218, 0.36363636363636365, 0.36363636363636365, 0.810810810810811, 0.6875, 0.7741935483870966, 0.7328244274809159, 0.6470588235294117, 0.5116279069767442, 0.3076923076923077, 0.810810810810811, 0.6875, 0.6875, 0.6875, 0.7328244274809159, 0.7328244274809159, 0.8181818181818182, 0.8181818181818182, 0.6875, 0.6875, 0.8974358974358974, 0.72, 0.6470588235294117, 0.6875, 0.72, 0.6875, 0.8181818181818182, 0.6875, 0.8424507658643327, 0.8181818181818182, 0.6875, 0.6875, 0.8181818181818182, 0.6875, 0.6875, 0.6875, 0.8181818181818182, 0.6875, 0.8181818181818182, 0.8181818181818182, 0.6470588235294117, 0.6470588235294117, 0.6875, 0.6875, 0.6875, 0.6875, 0.8181818181818182, 0.7741935483870966, 0.72, 0.810810810810811, 0.810810810810811, 0.8181818181818182, 0.6875, 0.6470588235294117, 0.8181818181818182, 0.5945945945945945, 0.36363636363636365, 0.6875, 0.8181818181818182, 0.8181818181818182, 0.8181818181818182, 0.6875, 0.6875, 0.6875, 0.36363636363636365, 0.8181818181818182, 0.8181818181818182, 0.6470588235294117, 0.6875, 0.3478260869565218, 0.3478260869565218, 0.3478260869565218, 0.6875, 0.6875, 0.8361774744027305, 0.8181818181818182, 0.3478260869565218, 0.6470588235294117, 0.36363636363636365, 0.6875, 0.6875, 0.6875, 0.5945945945945945, 0.7741935483870966, 0.810810810810811, 0.6875, 0.3478260869565218, 0.4166666666666667, 0.8361774744027305, 0.6470588235294117, 0.3076923076923077, 0.3076923076923077, 0.36363636363636365, 0.8181818181818182, 0.36363636363636365, 0.6875, 0.810810810810811, 0.7328244274809159, 0.36363636363636365, 0.6875, 0.7328244274809159, 0.6875, 0.6470588235294117, 0.8181818181818182, 0.8181818181818182, 0.6470588235294117, 0.6470588235294117, 0.72, 0.8181818181818182, 0.8181818181818182, 0.6875, 0.3478260869565218, 0.6470588235294117, 0.8181818181818182, 0.36363636363636365, 0.6875, 0.36363636363636365, 0.5116279069767442, 0.6875, 0.6875, 0.6875, 0.6875, 0.6470588235294117, 0.8181818181818182, 0.6875, 0.36363636363636365, 0.8181818181818182, 0.8361774744027305, 0.8361774744027305, 0.36363636363636365, 0.6875, 0.6875, 0.6875, 0.8181818181818182, 0.8181818181818182, 0.8181818181818182, 0.72, 0.8181818181818182, 0.810810810810811, 0.36363636363636365, 0.6470588235294117, 0.27999999999999997, 0.36363636363636365, 0.32653061224489793, 0.36363636363636365, 0.3478260869565218, 0.3076923076923077, 0.6470588235294117, 0.6875, 0.8181818181818182, 0.8361774744027305, 0.8181818181818182, 0.7741935483870966, 0.8181818181818182, 0.8181818181818182, 0.810810810810811, 0.810810810810811, 0.8361774744027305, 0.8361774744027305, 0.8181818181818182, 0.6875, 0.810810810810811, 0.6875, 0.6470588235294117, 0.6875, 0.6470588235294117, 0.810810810810811, 0.3076923076923077, 0.3076923076923077, 0.3478260869565218, 0.3076923076923077, 0.32653061224489793, 0.3076923076923077, 0.36363636363636365, 0.6470588235294117, 0.6875, 0.4166666666666667, 0.3478260869565218, 0.6875, 0.8181818181818182, 0.6470588235294117, 0.8181818181818182, 0.36363636363636365, 0.72, 0.7741935483870966, 0.8181818181818182, 0.6470588235294117, 0.36363636363636365, 0.8181818181818182, 0.8181818181818182, 0.8181818181818182, 0.8181818181818182, 0.8181818181818182, 0.32653061224489793, 0.6875, 0.27999999999999997, 0.8181818181818182, 0.8181818181818182, 0.8181818181818182, 0.8361774744027305, 0.6470588235294117, 0.3076923076923077, 0.27999999999999997, 0.3076923076923077, 0.27999999999999997, 0.3478260869565218, 0.3478260869565218, 0.72, 0.72, 0.7741935483870966, 0.9266802443991855, 0.8181818181818182, 0.6875, 0.6470588235294117, 0.8181818181818182, 0.8424507658643327, 0.8361774744027305, 0.8974358974358974, 0.6875, 0.945945945945946, 0.8424507658643327, 0.8181818181818182, 0.8181818181818182, 0.8361774744027305, 0.8361774744027305, 0.8181818181818182, 0.8361774744027305, 0.8361774744027305, 0.8181818181818182, 0.5945945945945945, 0.5116279069767442, 0.6875, 0.3076923076923077, 0.3478260869565218, 0.8181818181818182, 0.8181818181818182, 0.3478260869565218, 0.3076923076923077, 0.27999999999999997, 0.27999999999999997, 0.27999999999999997, 0.27999999999999997, 0.27999999999999997, 0.27999999999999997, 0.27999999999999997, 0.27999999999999997, 0.8181818181818182, 0.8181818181818182]
        dict_of_raw_subset_results['Outside']['subset_predicted_y'] = [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results

        fold_and_or_model_seed_combination='F0_100'
        dict_of_raw_subset_results = neverEndingDefaultDict()
        dict_of_raw_subset_results['Inside']['subset_test_ids'] = [33, 37, 77, 86, 101, 107, 108, 109, 163, 169, 179]
        dict_of_raw_subset_results['Inside']['subset_test_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
        dict_of_raw_subset_results['Inside']['subset_probs_for_class_1'] = [0.8450704225352113, 0.6895161290322581, 0.8745247148288973, 0.9444444444444444, 0.6895161290322581, 0.9743589743589745, 0.9743589743589745, 0.9743589743589745, 0.6310679611650485, 0.6310679611650485, 0.9444444444444444]
        dict_of_raw_subset_results['Inside']['subset_predicted_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        dict_of_raw_subset_results['Outside']['subset_test_ids'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273]
        dict_of_raw_subset_results['Outside']['subset_test_y'] = [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        dict_of_raw_subset_results['Outside']['subset_probs_for_class_1'] = [0.8421052631578948, 0.6459627329192548, 0.8421052631578948, 0.6459627329192548, 0.7003891050583658, 0.7411764705882353, 0.6895161290322581, 0.6705882352941176, 0.6459627329192548, 0.6459627329192548, 0.7804878048780488, 0.8450704225352113, 0.7500000000000001, 0.8421052631578948, 0.6459627329192548, 0.6459627329192548, 0.6310679611650485, 0.6459627329192548, 0.7411764705882353, 0.7411764705882353, 0.7411764705882353, 0.6705882352941176, 0.6895161290322581, 0.6895161290322581, 0.6310679611650485, 0.7500000000000001, 0.7411764705882353, 0.7411764705882353, 0.7411764705882353, 0.7500000000000001, 0.7500000000000001, 0.7411764705882353, 0.7411764705882353, 0.7105263157894737, 0.7411764705882353, 0.8817204301075269, 0.6705882352941176, 0.6459627329192548, 0.7003891050583658, 0.7411764705882353, 0.7500000000000001, 0.7411764705882353, 0.7500000000000001, 0.8817204301075269, 0.7411764705882353, 0.6459627329192548, 0.6459627329192548, 0.7105263157894737, 0.7105263157894737, 0.6895161290322581, 0.7411764705882353, 0.6705882352941176, 0.7105263157894737, 0.7500000000000001, 0.7500000000000001, 0.6459627329192548, 0.6459627329192548, 0.7411764705882353, 0.6895161290322581, 0.7105263157894737, 0.7411764705882353, 0.7500000000000001, 0.6895161290322581, 0.6895161290322581, 0.7411764705882353, 0.6666666666666666, 0.7500000000000001, 0.7938931297709924, 0.6705882352941176, 0.7500000000000001, 0.6459627329192548, 0.6459627329192548, 0.7411764705882353, 0.8421052631578948, 0.7500000000000001, 0.7500000000000001, 0.6666666666666666, 0.7411764705882353, 0.7411764705882353, 0.7411764705882353, 0.6459627329192548, 0.6459627329192548, 0.6895161290322581, 0.8421052631578948, 0.6666666666666666, 0.6705882352941176, 0.6459627329192548, 0.8421052631578948, 0.6459627329192548, 0.7652173913043478, 0.8421052631578948, 0.6310679611650485, 0.6895161290322581, 0.45454545454545453, 0.7500000000000001, 0.6459627329192548, 0.6459627329192548, 0.6705882352941176, 0.7500000000000001, 0.8057553956834531, 0.7105263157894737, 0.6459627329192548, 0.6459627329192548, 0.8450704225352113, 0.6459627329192548, 0.6310679611650485, 0.45454545454545453, 0.6459627329192548, 0.7411764705882353, 0.6459627329192548, 0.6459627329192548, 0.6705882352941176, 0.7003891050583658, 0.6459627329192548, 0.6459627329192548, 0.7105263157894737, 0.6895161290322581, 0.7411764705882353, 0.7804878048780488, 0.8421052631578948, 0.6895161290322581, 0.6895161290322581, 0.7411764705882353, 0.7500000000000001, 0.8450704225352113, 0.7652173913043478, 0.7003891050583658, 0.7500000000000001, 0.7500000000000001, 0.7652173913043478, 0.7500000000000001, 0.7500000000000001, 0.7105263157894737, 0.7938931297709924, 0.7500000000000001, 0.7500000000000001, 0.7652173913043478, 0.7500000000000001, 0.7652173913043478, 0.8817204301075269, 0.7411764705882353, 0.7804878048780488, 0.7500000000000001, 0.7411764705882353, 0.7500000000000001, 0.7411764705882353, 0.7500000000000001, 0.7003891050583658, 0.8057553956834531, 0.7500000000000001, 0.7652173913043478, 0.8450704225352113, 0.8450704225352113, 0.7500000000000001, 0.6459627329192548, 0.6895161290322581, 0.6310679611650485, 0.6459627329192548, 0.6459627329192548, 0.6459627329192548, 0.6459627329192548, 0.6310679611650485, 0.6459627329192548, 0.6459627329192548, 0.7652173913043478, 0.7652173913043478, 0.7652173913043478, 0.8057553956834531, 0.7652173913043478, 0.8450704225352113, 0.8450704225352113, 0.8641975308641975, 0.7652173913043478, 0.8817204301075269, 0.7500000000000001, 0.7003891050583658, 0.8450704225352113, 0.7652173913043478, 0.7500000000000001, 0.7411764705882353, 0.7411764705882353, 0.8421052631578948, 0.6310679611650485, 0.6310679611650485, 0.6459627329192548, 0.6310679611650485, 0.6310679611650485, 0.6459627329192548, 0.7500000000000001, 0.6895161290322581, 0.6895161290322581, 0.6459627329192548, 0.6459627329192548, 0.7500000000000001, 0.8421052631578948, 0.7938931297709924, 0.8745247148288973, 0.6459627329192548, 0.8057553956834531, 0.7652173913043478, 0.8057553956834531, 0.6666666666666666, 0.6459627329192548, 0.8421052631578948, 0.7411764705882353, 0.7500000000000001, 0.8450704225352113, 0.7411764705882353, 0.6310679611650485, 0.6459627329192548, 0.45454545454545453, 0.8421052631578948, 0.8421052631578948, 0.6310679611650485, 0.8421052631578948, 0.6459627329192548, 0.6459627329192548, 0.5777777777777778, 0.6310679611650485, 0.6310679611650485, 0.6459627329192548, 0.6459627329192548, 0.6895161290322581, 0.7500000000000001, 0.7411764705882353, 0.8421052631578948, 0.8421052631578948, 0.6666666666666666, 0.6895161290322581, 0.8450704225352113, 0.8745247148288973, 0.8641975308641975, 0.8817204301075269, 0.6459627329192548, 0.8817204301075269, 0.7938931297709924, 0.8421052631578948, 0.7938931297709924, 0.8421052631578948, 0.8421052631578948, 0.7938931297709924, 0.8421052631578948, 0.8421052631578948, 0.7938931297709924, 0.7500000000000001, 0.6705882352941176, 0.6459627329192548, 0.6459627329192548, 0.6459627329192548, 0.8421052631578948, 0.7938931297709924, 0.6705882352941176, 0.6459627329192548, 0.6459627329192548, 0.6459627329192548, 0.6459627329192548, 0.6459627329192548, 0.6459627329192548, 0.6459627329192548, 0.6459627329192548, 0.6459627329192548, 0.8421052631578948, 0.8450704225352113]
        dict_of_raw_subset_results['Outside']['subset_predicted_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results

        fold_and_or_model_seed_combination='F0_1'
        dict_of_raw_subset_results = neverEndingDefaultDict()
        dict_of_raw_subset_results['Inside']['subset_test_ids'] = [33, 37, 77, 86, 101, 107, 108, 109, 163, 169, 179]
        dict_of_raw_subset_results['Inside']['subset_test_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
        dict_of_raw_subset_results['Inside']['subset_probs_for_class_1'] = [0.9166666666666667, 0.8684210526315789, 0.9166666666666667, 0.9827586206896551, 0.7388059701492538, 0.9827586206896551, 0.9583333333333334, 0.978723404255319, 0.2962962962962963, 0.2962962962962963, 0.9827586206896551]
        dict_of_raw_subset_results['Inside']['subset_predicted_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
        dict_of_raw_subset_results['Outside']['subset_test_ids'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273]
        dict_of_raw_subset_results['Outside']['subset_test_y'] = [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        dict_of_raw_subset_results['Outside']['subset_probs_for_class_1'] = [0.6111111111111112, 0.45454545454545453, 0.6111111111111112, 0.45454545454545453, 0.6111111111111112, 0.9583333333333334, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.6111111111111112, 0.6111111111111112, 0.5641025641025642, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.45454545454545453, 0.6111111111111112, 0.7079856972586412, 0.7079856972586412, 0.7079856972586412, 0.7079856972586412, 0.7079856972586412, 0.7079856972586412, 0.7079856972586412, 0.6111111111111112, 0.6111111111111112, 0.9166666666666667, 0.9120603015075376, 0.6111111111111112, 0.6111111111111112, 0.6960297766749379, 0.6111111111111112, 0.6960297766749379, 0.6960297766749379, 0.9166666666666667, 0.9499999999999998, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9423076923076924, 0.6960297766749379, 0.6960297766749379, 0.882943143812709, 0.7388059701492538, 0.7388059701492538, 0.6960297766749379, 0.882943143812709, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6960297766749379, 0.9120603015075376, 0.6111111111111112, 0.5188679245283019, 0.6111111111111112, 0.882943143812709, 0.6960297766749379, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6960297766749379, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6960297766749379, 0.7079856972586412, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.7079856972586412, 0.6111111111111112, 0.8684210526315789, 0.9166666666666667, 0.6111111111111112, 0.6111111111111112, 0.45454545454545453, 0.9166666666666667, 0.6111111111111112, 0.6111111111111112, 0.5188679245283019, 0.6111111111111112, 0.7388059701492538, 0.6111111111111112, 0.45454545454545453, 0.6111111111111112, 0.9166666666666667, 0.6111111111111112, 0.45454545454545453, 0.45454545454545453, 0.6111111111111112, 0.7388059701492538, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.7079856972586412, 0.6111111111111112, 0.6111111111111112, 0.6960297766749379, 0.7079856972586412, 0.7388059701492538, 0.6111111111111112, 0.5188679245283019, 0.6111111111111112, 0.7079856972586412, 0.6111111111111112, 0.7388059701492538, 0.6111111111111112, 0.6111111111111112, 0.6960297766749379, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.7388059701492538, 0.7079856972586412, 0.6111111111111112, 0.9120603015075376, 0.6960297766749379, 0.7079856972586412, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.7388059701492538, 0.6960297766749379, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.6111111111111112, 0.6111111111111112, 0.45454545454545453, 0.2962962962962963, 0.45454545454545453, 0.3870967741935483, 0.45454545454545453, 0.2962962962962963, 0.2962962962962963, 0.6111111111111112, 0.6111111111111112, 0.7079856972586412, 0.7388059701492538, 0.6960297766749379, 0.7079856972586412, 0.7079856972586412, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.8684210526315789, 0.6111111111111112, 0.7079856972586412, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.45454545454545453, 0.45454545454545453, 0.6111111111111112, 0.45454545454545453, 0.6111111111111112, 0.45454545454545453, 0.6111111111111112, 0.9166666666666667, 0.7079856972586412, 0.6111111111111112, 0.45454545454545453, 0.7079856972586412, 0.9120603015075376, 0.7388059701492538, 0.9120603015075376, 0.6111111111111112, 0.7388059701492538, 0.6111111111111112, 0.9166666666666667, 0.6111111111111112, 0.6111111111111112, 0.9166666666666667, 0.7079856972586412, 0.9166666666666667, 0.9166666666666667, 0.6960297766749379, 0.45454545454545453, 0.6111111111111112, 0.45454545454545453, 0.9166666666666667, 0.9166666666666667, 0.6111111111111112, 0.9120603015075376, 0.6111111111111112, 0.6111111111111112, 0.45454545454545453, 0.6111111111111112, 0.5188679245283019, 0.5188679245283019, 0.5188679245283019, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.9166666666666667, 0.7388059701492538, 0.6111111111111112, 0.6111111111111112, 0.9166666666666667, 0.9210526315789472, 0.9166666666666667, 0.9230769230769229, 0.6111111111111112, 0.9230769230769229, 0.9166666666666667, 0.9120603015075376, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.9166666666666667, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.45454545454545453, 0.7079856972586412, 0.882943143812709, 0.5188679245283019, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.6111111111111112, 0.9166666666666667, 0.894578313253012]
        dict_of_raw_subset_results['Outside']['subset_predicted_y'] = [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results

        fold_and_or_model_seed_combination='F0_1000'
        dict_of_raw_subset_results = neverEndingDefaultDict()
        dict_of_raw_subset_results['Inside']['subset_test_ids'] = [33, 37, 77, 86, 101, 107, 108, 109, 163, 169, 179]
        dict_of_raw_subset_results['Inside']['subset_test_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
        dict_of_raw_subset_results['Inside']['subset_probs_for_class_1'] = [0.8533333333333335, 0.7311827956989247, 0.9142857142857144, 0.9571428571428572, 0.7311827956989247, 0.9571428571428572, 0.9565217391304348, 0.955223880597015, 0.3724137931034483, 0.3724137931034483, 0.9571428571428572]
        dict_of_raw_subset_results['Inside']['subset_predicted_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
        dict_of_raw_subset_results['Outside']['subset_test_ids'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273]
        dict_of_raw_subset_results['Outside']['subset_test_y'] = [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        dict_of_raw_subset_results['Outside']['subset_probs_for_class_1'] = [0.7311827956989247, 0.6, 0.7311827956989247, 0.5945945945945945, 0.6923076923076922, 0.8196721311475411, 0.6923076923076922, 0.6923076923076922, 0.4716981132075471, 0.4716981132075471, 0.6923076923076922, 0.6923076923076922, 0.6923076923076922, 0.6923076923076922, 0.6, 0.6, 0.4615384615384615, 0.6923076923076922, 0.6923076923076922, 0.7222222222222222, 0.6923076923076922, 0.6923076923076922, 0.6923076923076922, 0.7222222222222222, 0.5945945945945945, 0.7222222222222222, 0.7222222222222222, 0.7311827956989247, 0.7311827956989247, 0.7311827956989247, 0.7311827956989247, 0.7311827956989247, 0.7311827956989247, 0.6923076923076922, 0.7311827956989247, 0.7368421052631579, 0.8196721311475411, 0.6923076923076922, 0.6923076923076922, 0.7222222222222222, 0.6923076923076922, 0.7311827956989247, 0.7311827956989247, 0.7975460122699387, 0.7311827956989247, 0.7311827956989247, 0.7311827956989247, 0.7311827956989247, 0.7311827956989247, 0.7058823529411765, 0.7311827956989247, 0.7311827956989247, 0.7311827956989247, 0.8421052631578949, 0.8421052631578949, 0.6923076923076922, 0.6923076923076922, 0.7311827956989247, 0.7311827956989247, 0.7311827956989247, 0.7311827956989247, 0.7311827956989247, 0.6923076923076922, 0.6923076923076922, 0.6923076923076922, 0.6923076923076922, 0.6, 0.6923076923076922, 0.8791208791208791, 0.7311827956989247, 0.7311827956989247, 0.6923076923076922, 0.7311827956989247, 0.7608695652173914, 0.7222222222222222, 0.7311827956989247, 0.6923076923076922, 0.6923076923076922, 0.7311827956989247, 0.6923076923076922, 0.6923076923076922, 0.6923076923076922, 0.7311827956989247, 0.7311827956989247, 0.6923076923076922, 0.6923076923076922, 0.6, 0.7311827956989247, 0.6923076923076922, 0.7311827956989247, 0.7608695652173914, 0.4716981132075471, 0.6, 0.5945945945945945, 0.6923076923076922, 0.6923076923076922, 0.6, 0.6923076923076922, 0.6923076923076922, 0.7311827956989247, 0.7311827956989247, 0.6, 0.6, 0.8196721311475411, 0.6923076923076922, 0.4615384615384615, 0.4615384615384615, 0.5945945945945945, 0.7222222222222222, 0.6, 0.6923076923076922, 0.6923076923076922, 0.7058823529411765, 0.6, 0.6923076923076922, 0.7058823529411765, 0.6, 0.6923076923076922, 0.7058823529411765, 0.7311827956989247, 0.6923076923076922, 0.6923076923076922, 0.7311827956989247, 0.7311827956989247, 0.8421052631578949, 0.6923076923076922, 0.6923076923076922, 0.6923076923076922, 0.8196721311475411, 0.6923076923076922, 0.7311827956989247, 0.6923076923076922, 0.6923076923076922, 0.7311827956989247, 0.7222222222222222, 0.6923076923076922, 0.7311827956989247, 0.7311827956989247, 0.7975460122699387, 0.7692307692307693, 0.6923076923076922, 0.7692307692307693, 0.7311827956989247, 0.7311827956989247, 0.6923076923076922, 0.7311827956989247, 0.6923076923076922, 0.6, 0.7311827956989247, 0.7311827956989247, 0.7608695652173914, 0.8533333333333335, 0.8196721311475411, 0.7311827956989247, 0.6, 0.5670103092783504, 0.4615384615384615, 0.4716981132075471, 0.4615384615384615, 0.4615384615384615, 0.4615384615384615, 0.4615384615384615, 0.6923076923076922, 0.6923076923076922, 0.7311827956989247, 0.7608695652173914, 0.7311827956989247, 0.7311827956989247, 0.7222222222222222, 0.8196721311475411, 0.9208633093525178, 0.9208633093525178, 0.8196721311475411, 0.7311827956989247, 0.7311827956989247, 0.6923076923076922, 0.7222222222222222, 0.6923076923076922, 0.6521739130434783, 0.6923076923076922, 0.7311827956989247, 0.7311827956989247, 0.4716981132075471, 0.5670103092783504, 0.5670103092783504, 0.4716981132075471, 0.5945945945945945, 0.5945945945945945, 0.6, 0.7058823529411765, 0.6923076923076922, 0.6, 0.5670103092783504, 0.6923076923076922, 0.8196721311475411, 0.7311827956989247, 0.8533333333333335, 0.6923076923076922, 0.7975460122699387, 0.784313725490196, 0.7975460122699387, 0.7311827956989247, 0.6, 0.8791208791208791, 0.7975460122699387, 0.9208633093525178, 0.8196721311475411, 0.7975460122699387, 0.4615384615384615, 0.7311827956989247, 0.4716981132075471, 0.7222222222222222, 0.7222222222222222, 0.6, 0.7311827956989247, 0.6, 0.4615384615384615, 0.4716981132075471, 0.4716981132075471, 0.4615384615384615, 0.6, 0.4716981132075471, 0.6923076923076922, 0.6923076923076922, 0.6923076923076922, 0.7975460122699387, 0.7311827956989247, 0.6923076923076922, 0.6923076923076922, 0.8421052631578949, 0.8791208791208791, 0.9020771513353116, 0.8791208791208791, 0.6923076923076922, 0.8533333333333335, 0.7692307692307693, 0.8421052631578949, 0.7311827956989247, 0.7222222222222222, 0.7222222222222222, 0.7311827956989247, 0.7222222222222222, 0.7222222222222222, 0.7311827956989247, 0.6923076923076922, 0.6923076923076922, 0.6923076923076922, 0.6, 0.6, 0.7311827956989247, 0.7608695652173914, 0.6, 0.5945945945945945, 0.4615384615384615, 0.4615384615384615, 0.4615384615384615, 0.4615384615384615, 0.4615384615384615, 0.4615384615384615, 0.4615384615384615, 0.4615384615384615, 0.7058823529411765, 0.8421052631578949]
        dict_of_raw_subset_results['Outside']['subset_predicted_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results

        fold_and_or_model_seed_combination='F0_657'
        dict_of_raw_subset_results = neverEndingDefaultDict()
        dict_of_raw_subset_results['Inside']['subset_test_ids'] = [33, 37, 77, 86, 101, 107, 108, 109, 163, 169, 179]
        dict_of_raw_subset_results['Inside']['subset_test_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
        dict_of_raw_subset_results['Inside']['subset_probs_for_class_1'] = [0.7337278106508877, 0.7142857142857143, 0.7398843930635838, 0.9684608915054668, 0.7142857142857143, 0.9684608915054668, 0.9684608915054668, 0.9684608915054668, 0.5833333333333334, 0.5384615384615384, 0.9648382559774966]
        dict_of_raw_subset_results['Inside']['subset_predicted_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        dict_of_raw_subset_results['Outside']['subset_test_ids'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273]
        dict_of_raw_subset_results['Outside']['subset_test_y'] = [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        dict_of_raw_subset_results['Outside']['subset_probs_for_class_1'] = [0.7142857142857143, 0.6896551724137931, 0.7142857142857143, 0.6896551724137931, 0.6896551724137931, 0.946236559139785, 0.7142857142857143, 0.6896551724137931, 0.6493506493506492, 0.6493506493506492, 0.7337278106508877, 0.7804878048780488, 0.7441860465116278, 0.7804878048780488, 0.6896551724137931, 0.6896551724137931, 0.6493506493506492, 0.6896551724137931, 0.6896551724137931, 0.7398843930635838, 0.7441860465116278, 0.7337278106508877, 0.7142857142857143, 0.6896551724137931, 0.6493506493506492, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.6896551724137931, 0.6896551724137931, 0.7804878048780488, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7804878048780488, 0.7804878048780488, 0.7337278106508877, 0.7337278106508877, 0.7261613691931541, 0.7337278106508877, 0.7261613691931541, 0.7261613691931541, 0.7337278106508877, 0.7337278106508877, 0.7337278106508877, 0.7398843930635838, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.6896551724137931, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.6896551724137931, 0.7142857142857143, 0.7301927194860813, 0.7142857142857143, 0.7398843930635838, 0.7142857142857143, 0.6896551724137931, 0.7142857142857143, 0.7261613691931541, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.6896551724137931, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.6896551724137931, 0.6896551724137931, 0.7261613691931541, 0.6896551724137931, 0.7142857142857143, 0.6896551724137931, 0.7261613691931541, 0.7142857142857143, 0.8301886792452832, 0.7804878048780488, 0.6896551724137931, 0.6896551724137931, 0.6493506493506492, 0.6896551724137931, 0.7142857142857143, 0.6896551724137931, 0.6896551724137931, 0.7142857142857143, 0.7142857142857143, 0.6896551724137931, 0.6493506493506492, 0.6896551724137931, 0.8301886792452832, 0.6896551724137931, 0.6493506493506492, 0.641025641025641, 0.6896551724137931, 0.9214659685863874, 0.6896551724137931, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.6896551724137931, 0.6896551724137931, 0.7142857142857143, 0.6896551724137931, 0.6896551724137931, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7804878048780488, 0.7804878048780488, 0.7261613691931541, 0.7142857142857143, 0.6896551724137931, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.6896551724137931, 0.6896551724137931, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.8979591836734693, 0.7398843930635838, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7301927194860813, 0.7142857142857143, 0.7261613691931541, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.6896551724137931, 0.8301886792452832, 0.7142857142857143, 0.7142857142857143, 0.9565217391304348, 0.8979591836734693, 0.7441860465116278, 0.6896551724137931, 0.6896551724137931, 0.641025641025641, 0.6896551724137931, 0.6493506493506492, 0.6493506493506492, 0.641025641025641, 0.5384615384615384, 0.7142857142857143, 0.6896551724137931, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7261613691931541, 0.7142857142857143, 0.7398843930635838, 0.7441860465116278, 0.7441860465116278, 0.8979591836734693, 0.7804878048780488, 0.7441860465116278, 0.7142857142857143, 0.7337278106508877, 0.7804878048780488, 0.7142857142857143, 0.7142857142857143, 0.6896551724137931, 0.7142857142857143, 0.6493506493506492, 0.6493506493506492, 0.6896551724137931, 0.6493506493506492, 0.6896551724137931, 0.6493506493506492, 0.6896551724137931, 0.7142857142857143, 0.6896551724137931, 0.6896551724137931, 0.6896551724137931, 0.6896551724137931, 0.7142857142857143, 0.6896551724137931, 0.7142857142857143, 0.6896551724137931, 0.7261613691931541, 0.7142857142857143, 0.7142857142857143, 0.6896551724137931, 0.6896551724137931, 0.7261613691931541, 0.7142857142857143, 0.7804878048780488, 0.7142857142857143, 0.6896551724137931, 0.6896551724137931, 0.7142857142857143, 0.6896551724137931, 0.8301886792452832, 0.8301886792452832, 0.6896551724137931, 0.8979591836734693, 0.6896551724137931, 0.6493506493506492, 0.6493506493506492, 0.6493506493506492, 0.6493506493506492, 0.6896551724137931, 0.6896551724137931, 0.6896551724137931, 0.7142857142857143, 0.7142857142857143, 0.7398843930635838, 0.8301886792452832, 0.7142857142857143, 0.7142857142857143, 0.9565217391304348, 0.946236559139785, 0.9565217391304348, 0.9214659685863874, 0.7142857142857143, 0.9565217391304348, 0.9214659685863874, 0.7142857142857143, 0.7261613691931541, 0.7441860465116278, 0.7441860465116278, 0.7261613691931541, 0.7441860465116278, 0.7441860465116278, 0.7261613691931541, 0.6896551724137931, 0.6896551724137931, 0.7142857142857143, 0.6896551724137931, 0.6896551724137931, 0.7142857142857143, 0.7337278106508877, 0.6896551724137931, 0.6896551724137931, 0.641025641025641, 0.6493506493506492, 0.6493506493506492, 0.6493506493506492, 0.6493506493506492, 0.6493506493506492, 0.6493506493506492, 0.6493506493506492, 0.7142857142857143, 0.7441860465116278]
        dict_of_raw_subset_results['Outside']['subset_predicted_y'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results
        #############################
        example_dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name] = dict_for_all_folds_and_or_model_seeds_of_raw_subset_results

        metric_we_are_interested_in = 'R2 (cal)'
        metric_to_expected_sub_1_minus_sub_2_sign={'R2 (cal)':1}

        return example_dict_for_all_test_sets_of_raw_subset_results,metric_we_are_interested_in,metric_to_expected_sub_1_minus_sub_2_sign,test_set_plus_methods_name

    def test_getAdjustedPValues_works_with_missing_vals(self):
        from modelsADuncertaintyPkg.qsar_eval.adjust_p_vals import getAdjustedPValues

        ps = [0.04, 0.001, np.nan, 0.01]
        ps_no_nan = [v for v in ps if not pd.isna(v)]

        adjusted_ps = getAdjustedPValues(ps)

        assert pd.isna(adjusted_ps[2])
        assert all([isinstance(adjusted_ps[i],float) for i in range(0,4) if not 2 == i])
        assert all([(adjusted_ps[i] > ps[i]) for i in range(0,4) if not 2 == i])

        adjusted_ps_no_nan = getAdjustedPValues(ps_no_nan)

        filtered_nan_from_adjusted_ps = [v for v in adjusted_ps if not pd.isna(v)]

        assert adjusted_ps_no_nan == filtered_nan_from_adjusted_ps

        assert adjusted_ps_no_nan == getAdjustedPValues(ps_no_nan,skip_missing=False)

        np.testing.assert_equal(np.array([adjusted_ps_no_nan[0],adjusted_ps_no_nan[1],np.nan,adjusted_ps_no_nan[2]]),np.array(adjusted_ps))
    
    def test_get_multiple_random_splits_DOES_GIVE_DIFFERENT_SPLITS(self):
        from modelsADuncertaintyPkg.utils.ML_utils import get_multiple_random_splits

        #===================
        #Toy example c.f. use of get_multiple_random_splits(...) inside funcsToApplyRegUncertaintyToNewDatasets.py: 
        XX = pd.DataFrame({'x1':[1,2,3,4,5],'x2':[1,0,0,1,0]})
        yy = pd.Series([1.1,1.2,1.3,1.4,1.5])
        #===================

        no_splits = 5

        dict_of_splits = get_multiple_random_splits(no_splits=no_splits,data_x=XX,data_y=yy,test_fraction=0.2,random_state=42,stratified=True,reset_indices=True)

        old_X_train,old_y_train,old_y_train_sorted,old_X_calib,old_y_calib,old_y_calib_sorted = [None]*6

        checks_count = 0
        for split in dict_of_splits.keys():
        
            X_train = dict_of_splits[split]['train_x']
            y_train = dict_of_splits[split]['train_y']
            y_train_sorted = y_train.sort_values(ignore_index=True)
            X_calib = dict_of_splits[split]['test_or_calib_x']
            y_calib = dict_of_splits[split]['test_or_calib_y']
            y_calib_sorted = y_calib.sort_values(ignore_index=True)

            if not old_X_train is None:
                assert not old_X_train.equals(X_train)
                assert not old_y_train.equals(y_train)
                assert not old_y_train_sorted.equals(y_train_sorted)
                assert not old_X_calib.equals(X_calib)
                assert not old_y_calib.equals(y_calib)
                assert not old_y_calib_sorted.equals(y_calib_sorted)
                checks_count += 1
            
            old_X_train = X_train
            old_y_train = y_train
            old_y_train_sorted = y_train_sorted
            old_X_calib = X_calib
            old_y_calib = y_calib
            old_y_calib_sorted = y_calib_sorted
        
        assert checks_count == (no_splits-1),f'checks_count ={checks_count} vs. no_splits={no_splits}'

                
    def test_create_subset_sig_level_2_prediction_interval_val(self):
        from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import create_subset_sig_level_2_prediction_interval_val
        from modelsADuncertaintyPkg.utils.ML_utils import compute_pred_intervals

        subset = 'Inside'

        dict_of_raw_subset_results  = neverEndingDefaultDict()

        sig_level = 0.32

        y_pred = [1.5,2.5,3.0]
        half_pred_interval_sizes = [0.1,0.50,0.05]

        intervals = compute_pred_intervals(y_pred,half_pred_interval_sizes)


        dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'][sig_level] = intervals

        for index in range(3):
            print(f'Checking for index={index}')

            expected_res = compute_pred_intervals([y_pred[index]],[half_pred_interval_sizes[index]])

            res = create_subset_sig_level_2_prediction_interval_val(dict_of_raw_subset_results,subset,index)[sig_level]

            #c.f. how the output of the previous function is used in assess_stat_sig_shift_metrics.py: np.array([dict_of_merged_ids_matched_to_raw_res[id_]['subset_sig_level_2_prediction_interval_val'][sig_level] for id_ in subset_ids])
            res_converted_back_to_2d_arrray = np.array([res for id_ in [0]])

            assert np.array_equal(expected_res,res_converted_back_to_2d_arrray)

    def test_convert_sdf_to_DataFrame_with_SMILES(self):
        from modelsADuncertaintyPkg.CheminformaticsUtils.chem_data_parsing_utils import convert_sdf_to_DataFrame_with_SMILES

        smiles_col = 'Calc_SMILES'

        mol_smiles_list = ['CCO','CCN']

        tmp_sdf_file_name = 'test_convert_sdf_to_DataFrame_with_SMILES.sdf'

        expected_df = pd.DataFrame({'ID':['mol-1','mol-2'],smiles_col:mol_smiles_list})

        assert not os.path.exists(tmp_sdf_file_name)

        writer = ChemDataParser.Chem.SDWriter(tmp_sdf_file_name)

        for i in range(len(mol_smiles_list)):
            mol = ChemDataParser.Chem.MolFromSmiles(mol_smiles_list[i])
            mol.SetProp("_Name",f'mol-{i+1}')
            writer.write(mol)
        writer.close()

        assert os.path.exists(tmp_sdf_file_name)

        #copied from generate_Tox21_SMILES_files.py:
        df = ChemDataParser.convert_sdf_to_DataFrame_with_SMILES(sdf_name=tmp_sdf_file_name,mol_id='ID',smiles_col_name=smiles_col,drop_rdkit_mol=True)

        assert_frame_equal(df,expected_df)

        os.remove(tmp_sdf_file_name)

        assert not os.path.exists(tmp_sdf_file_name)


    def test_getADSubsetTestIDsInOrder(self):
        from modelsADuncertaintyPkg.qsar_AD.applyADmethods import getADSubsetTestIDsInOrder

        #In all modelling scripts and inside AD percentage script, test_ids is a list:
        test_ids = [9,8,1,2]
        
        ad_status = [True,False,False,True]
        p_vals = [0.32,0.01,0.04,0.58]

        #adapted from UNC_like_adapted_AD_approach.py:
        test_id_ad_status_dict = defaultdict(dict)
            
        for i in range(len(test_ids)):
            new_id = test_ids[i]
            inside_AD = ad_status[i]
            p_value = p_vals[i]
                
            
            test_id_ad_status_dict[new_id]['InsideAD'] = inside_AD
            test_id_ad_status_dict[new_id]['P-value'] = p_value
            
        
        test_id_status_dict = test_id_ad_status_dict

        #copied from classification exemplar endpoints script and occurs verbatim in SYN classification and regression scripts
        #regression exemplar endpoints script usage is identical, with some spaces added: subset_test_ids = getADSubsetTestIDsInOrder(AD_subset, test_id_status_dict, test_ids)
        #AD subset names are used consistently in all modelling scripts

        for AD_subset in ['All', 'Inside', 'Outside']:

            subset_test_ids = getADSubsetTestIDsInOrder(AD_subset,test_id_status_dict,test_ids)

            if 'All' == AD_subset:
                assert test_ids == subset_test_ids
            
            elif 'Inside' == AD_subset:
                assert [9,2] == subset_test_ids
            
            elif 'Outside' == AD_subset:
                assert [8,1] == subset_test_ids
            
            else:
                raise Exception(f'Unexpected AD_subset={AD_subset}')


    def test_computeRecallAndPrecision(self):
        from modelsADuncertaintyPkg.qsar_eval.all_key_class_stats_and_plots import computeRecallAndPrecision


        preds = [1,0,0,0,1]
        experi = [1,1,1,1,0]

        precision,recall = computeRecallAndPrecision(class_val=1,experi_class_labels=experi,y_pred=preds)
        assert precision == 0.5
        assert recall == 0.25

        precision,recall = computeRecallAndPrecision(class_val=0,experi_class_labels=experi,y_pred=preds)
        assert precision == 0.0
        assert recall == 0.0

        preds = [0,0,0,0,0]

        precision,recall = computeRecallAndPrecision(class_val=1,experi_class_labels=experi,y_pred=preds)
        assert precision is None
        assert recall == 0.0

        precision,recall = computeRecallAndPrecision(class_val=0,experi_class_labels=experi,y_pred=preds)
        assert precision == 0.2
        assert recall == 1.0

        experi = [1,1,1,1,1]

        precision,recall = computeRecallAndPrecision(class_val=0,experi_class_labels=experi,y_pred=preds)
        assert precision  == 0.0
        assert recall is None

    def test_classesAreOneOrZeroAndOneClassMissing(self):

        from modelsADuncertaintyPkg.qsar_eval.all_key_class_stats_and_plots import classesAreOneOrZeroAndOneClassMissing

        #looking at where this is used and all occurences of computeAllClassMetrics(...) in modelling scripts, it appears that the inputs are always lists:

        assert not classesAreOneOrZeroAndOneClassMissing([1,0,1])
        assert classesAreOneOrZeroAndOneClassMissing([1,1,1])
        assert classesAreOneOrZeroAndOneClassMissing([0,0,0])

        try:
            classesAreOneOrZeroAndOneClassMissing([1,2,1])
            raise Exception('This should have failed and been caught before now!')
        except Exception:
            pass
    
    def test_computeAllClassMetrics(self):

        from modelsADuncertaintyPkg.qsar_eval import all_key_class_stats_and_plots as ClassEval

        lit_precedent_delta_for_calib_plot = 0.05
        calc_dir = os.getcwd()
        context_label = 'unit.test'
        AD_subset = 'unit.test'
        method = 'unit.test'

        subset_test_y = [1,1,1,1,0]
        subset_predicted_y = [1,0,0,0,1]
        subset_probs_for_class_1 = [0.8,0.49,0.20,0.10,0.60]

        #copied from exemplar targets modelling script (used in exactly the same way for SYN classification script):

        precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal,no_cmpds_1,no_cmpds_0,no_cmpds =        ClassEval.computeAllClassMetrics(test_y=subset_test_y,predicted_y=subset_predicted_y,probs_for_class_1=subset_probs_for_class_1,method=method,subset_name=f'{context_label}_{AD_subset}',output_dir=calc_dir,delta_for_calib_plot=lit_precedent_delta_for_calib_plot)

        #these expected results were computed by hand
        assert precision_1 == 0.50
        assert precision_0 == 0.0
        assert recall_1 == 0.25
        assert recall_0 == 0.0
        assert ba == 0.125
        assert mcc == approx(-0.6123724356957946)
        assert kappa == approx(-0.4285714285714287)
        assert brier == 0.42202
        assert strat_brier == 0.3987625
        assert no_cmpds == 5
        assert no_cmpds_1 == 4
        assert no_cmpds_0 == 1
        #because no compounds with different class labels were within 0.05 of one another:
        experi_probs = [1,1,1,1,0]
        assert rmseCal == np.sqrt(brier)
        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
        from sklearn.metrics import mean_absolute_error
        assert MADCal == mean_absolute_error(y_true=experi_probs,y_pred=subset_probs_for_class_1)
        from sklearn.metrics import r2_score
        assert coeffOfDeterminationCal == r2_score(y_true=experi_probs,y_pred=subset_probs_for_class_1)
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        #Was keyword alternative introduced later?
        from scipy.stats import pearsonr
        assert PearsonCoeffCal == pearsonr(experi_probs,subset_probs_for_class_1)[0]#,alternative="greater")[0]
        assert PearsonCoeffCal < 0
        assert PearsonCoeffPvalCal == 1-(pearsonr(experi_probs,subset_probs_for_class_1)[1]/2)#,alternative="greater")[1]
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
        from scipy.stats import spearmanr
        assert SpearmanCoeffCal == spearmanr(experi_probs,subset_probs_for_class_1)[0]#,alternative="greater")[0]
        assert SpearmanCoeffCal < 0
        assert SpearmanCoeffPvalCal == 1 - (spearmanr(experi_probs,subset_probs_for_class_1)[1]/2)#,alternative="greater")[1]
        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        from sklearn.metrics import roc_auc_score
        assert auc == roc_auc_score(y_true=subset_test_y,y_score=subset_probs_for_class_1)
    
    def check_produceGraphicalSummaries(self,orig_shift_metric_val,rand_shift_metric_vals,plot_file_prefix,sig_level_perc,one_tail):
        from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import produceGraphicalSummaries
        from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import getAllInputsForGraphicalSummaries

        x_lab = 'Groups'
        y_lab = 'Shift-Metric'
        legend_lab = 'Split Basis'
        

        metric_name = y_lab

        #===================
        #c.f. how this is used inside def getAllInputsForGraphicalSummaries(...): and created inside def getForAllMetricsOriginalVsRandShiftMetricVals(...):
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals = defaultdict(dict)
        for metric in [metric_name]:
            dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals[metric]['original'] = orig_shift_metric_val
        
            dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals[metric]['random'] = rand_shift_metric_vals
        #====================

        input_for_graphical_summaries = getAllInputsForGraphicalSummaries(dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals)

        produceGraphicalSummaries(input_for_graphical_summaries,plot_file_prefix,x_lab=x_lab,y_lab=y_lab,legend_lab=legend_lab,sig_level_perc=sig_level_perc,one_sided_sig_test=one_tail,debug=True)


    def test_produceGraphicalSummaries_two_tail_not_stat_sig(self,debug=True):
        from modelsADuncertaintyPkg.utils.basic_utils import report_name_of_function_where_this_is_called

        one_tail = False

        orig_shift_metric_val = 0.40

        sig_level_perc = 5

        generator = np.random.default_rng(seed=42)
        rand_shift_metric_vals = [float(v)/100 for v in generator.integers(low=-(orig_shift_metric_val*100),high=(orig_shift_metric_val*100),size=(100-sig_level_perc))]
        rand_shift_metric_vals += [-0.52,0.60,-0.60,0.55,0.70]

        if debug:
            rand_shift_metric_vals.sort()
            print(f'rand_shift_metric_vals={rand_shift_metric_vals}')

        plot_file_prefix = f'{report_name_of_function_where_this_is_called()}'
        
        self.check_produceGraphicalSummaries(orig_shift_metric_val,rand_shift_metric_vals,plot_file_prefix,sig_level_perc,one_tail)
        
    def test_produceGraphicalSummaries_two_tail_stat_sig(self,debug=True):
        from modelsADuncertaintyPkg.utils.basic_utils import report_name_of_function_where_this_is_called

        one_tail = False

        orig_shift_metric_val = 0.40

        sig_level_perc = 5

        generator = np.random.default_rng(seed=42)
        rand_shift_metric_vals = [float(v)/100 for v in generator.integers(low=-(orig_shift_metric_val*100)+3,high=(orig_shift_metric_val*100)-3,size=(100-sig_level_perc))]
        rand_shift_metric_vals += [0.38,0.39,0.41,-0.38,-0.40]

        if debug:
            rand_shift_metric_vals.sort()
            print(f'rand_shift_metric_vals={rand_shift_metric_vals}')

        plot_file_prefix = f'{report_name_of_function_where_this_is_called()}'
        
        self.check_produceGraphicalSummaries(orig_shift_metric_val,rand_shift_metric_vals,plot_file_prefix,sig_level_perc,one_tail)


    def test_produceGraphicalSummaries_one_tail_based_on_two_tail_not_stat_sig(self,debug=True):
        from modelsADuncertaintyPkg.utils.basic_utils import report_name_of_function_where_this_is_called

        one_tail = True

        orig_shift_metric_val = 0.40

        sig_level_perc = 5

        generator = np.random.default_rng(seed=42)
        rand_shift_metric_vals = [float(v)/100 for v in generator.integers(low=-(orig_shift_metric_val*100),high=(orig_shift_metric_val*100),size=(100-sig_level_perc))]
        rand_shift_metric_vals += [-0.52,0.60,-0.60,0.55,0.70]

        if debug:
            rand_shift_metric_vals.sort()
            print(f'rand_shift_metric_vals={rand_shift_metric_vals}')

        plot_file_prefix = f'{report_name_of_function_where_this_is_called()}'
        
        self.check_produceGraphicalSummaries(orig_shift_metric_val,rand_shift_metric_vals,plot_file_prefix,sig_level_perc,one_tail)
        
    def test_produceGraphicalSummaries_one_tail_based_on_two_tail_stat_sig(self,debug=True):
        from modelsADuncertaintyPkg.utils.basic_utils import report_name_of_function_where_this_is_called

        one_tail = True

        orig_shift_metric_val = 0.40

        sig_level_perc = 5

        generator = np.random.default_rng(seed=42)
        rand_shift_metric_vals = [float(v)/100 for v in generator.integers(low=-(orig_shift_metric_val*100)+3,high=(orig_shift_metric_val*100)-3,size=(100-sig_level_perc))]
        rand_shift_metric_vals += [0.38,0.39,0.41,-0.38,-0.40]

        if debug:
            rand_shift_metric_vals.sort()
            print(f'rand_shift_metric_vals={rand_shift_metric_vals}')

        plot_file_prefix = f'{report_name_of_function_where_this_is_called()}'
        
        self.check_produceGraphicalSummaries(orig_shift_metric_val,rand_shift_metric_vals,plot_file_prefix,sig_level_perc,one_tail)


    def test_makeIDsNumeric(self):
        from modelsADuncertaintyPkg.utils.ML_utils import makeIDsNumeric

        input_df = pd.DataFrame({'ID':['a','c','b'],'x1':[1.5,6.1,2.3]})

        expected_output_df = pd.DataFrame({'ID':[0,1,2],'x1':[1.5,6.1,2.3]})

        

        ##############
        #c.f. use in all modelling and inside-AD percentage scripts:
        output_df = makeIDsNumeric(df=input_df,id_col='ID')
        ##############

        try:
            assert_frame_equal(output_df,expected_output_df)
        except AssertionError:
            #Strangely, putting this statement above was needed to make this test pass on my laptop (with ostensibly the same environment - but maybe a different architecture - as GitHub test runner)
            expected_output_df = expected_output_df.astype({'ID':'int32'})
            assert_frame_equal(output_df,expected_output_df)

    
    def test_makeIDsNumeric_without_continuous_indices(self):
        from modelsADuncertaintyPkg.utils.ML_utils import makeIDsNumeric

        input_df = pd.DataFrame({'ID':['a','c','b'],'x1':[1.5,6.1,2.3]})

        input_df = input_df[input_df['ID'].isin(['a','b'])]

        expected_output_df = pd.DataFrame({'ID':[0,1],'x1':[1.5,2.3]})

        

        expected_output_df.index = [0,2]

        ##############
        #c.f. use in all modelling and inside-AD percentage scripts:
        output_df = makeIDsNumeric(df=input_df,id_col='ID')
        ##############

        try:
            assert_frame_equal(output_df,expected_output_df)
        except AssertionError:
            #Strangely, putting this statement above was needed to make this test pass on my laptop (with ostensibly the same environment - but maybe a different architecture - as GitHub test runner)
            expected_output_df = expected_output_df.astype({'ID':'int32'})
            assert_frame_equal(output_df,expected_output_df)

    def test_makeYLabelsNumeric_endpoint_col(self):

        from modelsADuncertaintyPkg.utils.ML_utils import makeYLabelsNumeric

        ######
        #c.f. uses in scripts:
        #X_train[endpoint_col] = makeYLabelsNumeric(X_train[endpoint_col], class_1=class_1_label,class_0=class_0_label)
        #train_inc_calib_y = makeYLabelsNumeric(train_inc_calib_y, class_1=class_1, class_0=class_0,return_as_pandas_series=True)
        #test_y = makeYLabelsNumeric(test_y, class_1=class_1, class_0=class_0,return_as_pandas_series=True)
        ######

        endpoint_col = 'Y'

        X_train= pd.DataFrame({'ID':['a','b','c'],'x1':[1,0,1],endpoint_col:['act','inact','act']})

        X_train[endpoint_col] = makeYLabelsNumeric(X_train[endpoint_col], class_1='act',class_0='inact')

        expected_X_train = pd.DataFrame({'ID':['a','b','c'],'x1':[1,0,1],endpoint_col:[1,0,1]})

        assert_frame_equal(X_train,expected_X_train)

    def test_makeYLabelsNumeric_endpoint_series(self):

        from modelsADuncertaintyPkg.utils.ML_utils import makeYLabelsNumeric

        ######
        #c.f. uses in scripts:
        #X_train[endpoint_col] = makeYLabelsNumeric(X_train[endpoint_col], class_1=class_1_label,class_0=class_0_label)
        #train_inc_calib_y = makeYLabelsNumeric(train_inc_calib_y, class_1=class_1, class_0=class_0,return_as_pandas_series=True)
        #test_y = makeYLabelsNumeric(test_y, class_1=class_1, class_0=class_0,return_as_pandas_series=True)
        ######

        endpoint_col = 'Y'

        train_inc_calib_y = pd.Series(['act','inact','act'])

        train_inc_calib_y = makeYLabelsNumeric(train_inc_calib_y, class_1='act', class_0='inact',return_as_pandas_series=True)

        expected_train_inc_calib_y = pd.Series([1,0,1])

        assert_series_equal(train_inc_calib_y,expected_train_inc_calib_y)
    
    def test_not_resetting_y_indices_does_not_affect_native_model_fit_RF(self):

        from modelsADuncertaintyPkg.qsar_reg_uncertainty import RegressionICP as icp

        train_inc_calib_x,train_inc_calib_y,test_x,test_y,cal_fract,seed,calib_size = self.get_inputs_required_for_checking_conformal_regression_workflow_functions()

        expected_orig_indices = list(range(train_inc_calib_y.shape[0]))

        assert train_inc_calib_y.index.tolist() == expected_orig_indices

        model_1 = icp.fit_RF(train_inc_calib_x,train_inc_calib_y)

        predictions_1 = model_1.predict(test_x)

        generator = np.random.default_rng(seed=42)

        orig_indices = copy.deepcopy(expected_orig_indices)

        generator.shuffle(orig_indices)

        assert not orig_indices == expected_orig_indices

        train_inc_calib_y.index = orig_indices

        model_2 = icp.fit_RF(train_inc_calib_x,train_inc_calib_y)

        predictions_2 = model_2.predict(test_x)

        assert_series_equal(pd.Series(predictions_1),pd.Series(predictions_2))
    
    def test_this_data_frame_contains_missing_values(self):
        from modelsADuncertaintyPkg.utils.basic_utils import this_data_frame_contains_missing_values

        assert not this_data_frame_contains_missing_values(pd.DataFrame({'ID':[1,2,3],'x1':[1.5,2.5,3.5]}))
        assert this_data_frame_contains_missing_values(pd.DataFrame({'ID':[1,2,3],'x1':[None,2.5,3.5]}))
        assert this_data_frame_contains_missing_values(pd.DataFrame({'ID':[1,2,3],'x1':[-1,np.nan,3.5]}))
        assert not this_data_frame_contains_missing_values(pd.DataFrame({'ID':[1,2,3],'x1':[np.inf,2.5,3.5]}))
        assert not this_data_frame_contains_missing_values(pd.DataFrame({'ID':[1,2,3],'x1':[-np.inf,2.5,3.5]}))

    def test_addFPBitsColsToDf_Row_WithSMILES_pre_calc_fp_col_is_None_and_not_pre_calc_stand_mol_col_is_None(self):
        simple_smiles = ChemDataParser.simple_smiles
        simple_pre_calc_stand_mol = standardize(simple_smiles)

        smiles_col = 'Irrelevant'
        pre_calc_stand_mol_col = 'Arbitary'
        mol_parse_status_col='Mol.Parsed.OK'

        row = pd.Series({pre_calc_stand_mol_col:simple_pre_calc_stand_mol})

        updated_row = ChemDataParser.addFPBitsColsToDf_Row_WithSMILES(row,smiles_col,bitInfo={},report_problem_smiles=True,mol_parse_status_col=mol_parse_status_col,pre_calc_stand_mol_col=pre_calc_stand_mol_col,pre_calc_fp_col=None,type_of_fp='Morgan')

        expected_updated_row = pd.Series([simple_pre_calc_stand_mol]+ChemDataParser.expected_fp_bits_list_of_simple_mol+[True],index=[pre_calc_stand_mol_col]+[str(i) for i in range(0,1024)]+[mol_parse_status_col])

        assert_series_equal(updated_row,expected_updated_row)

    def get_toy_inputs_for_test_computeAllRegMetrics(self):
        generator = np.random.default_rng(seed=42)

        test_y=generator.random(size=40).tolist()
        offsets = [(v/2) for v in generator.random(size=40).tolist()]
        test_predictions=[(test_y[i]-offsets[i]) for i in range(40)]
        half_pred_interval_sizes = [(v/10) for v in range(40)]

        sig_level_of_interest = 0.32

        #########
        #compute_pred_intervals(...) is used in computeIntervals(...), which is used by all conformal regression approaches at some point;
        sig_level_2_prediction_intervals = {sig_level_of_interest:compute_pred_intervals(y_pred=test_predictions,half_pred_interval_sizes=half_pred_interval_sizes)}
        #########

        return test_y,test_predictions,half_pred_interval_sizes,sig_level_of_interest,sig_level_2_prediction_intervals


    def test_computeAllRegMetrics(self):

        from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import computeAllRegMetrics,getAllerrorRate_s
        from modelsADuncertaintyPkg.utils.ML_utils import compute_pred_intervals
        from modelsADuncertaintyPkg.qsar_eval.reg_Uncertainty_metrics import compute_ECE,compute_ENCE

        test_y,test_predictions,half_pred_interval_sizes,sig_level_of_interest,sig_level_2_prediction_intervals = self.get_toy_inputs_for_test_computeAllRegMetrics()

        ######################
        #c.f. its use in both regression modelling scripts which define the use of this function (vs. re-use of script function when moving to other public endpoints):
        rmse,MAD,R2,Pearson,Pearson_Pval_one_tail,Spearman,Spearman_Pval_one_tail,validity, efficiency,ECE_new,ENCE,errorRate_s,no_compounds,scc = computeAllRegMetrics(test_y,test_predictions,sig_level_of_interest,sig_level_2_prediction_intervals)
        ######################

        #######################
        #We estimated these in Excel:
        assert rmse == approx(0.27056973)
        assert MAD == approx(0.239747152)
        assert R2 == approx(0.102525429)
        assert Pearson == approx(0.91521957)
        ######################

        ##############
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
        from scipy.stats import spearmanr
        assert Spearman == spearmanr(test_y,test_predictions)[0]
        assert Spearman > 0
        assert Spearman_Pval_one_tail == (spearmanr(test_y,test_predictions)[1]/2)
        #############

        ################
        #we already have unit-tests for these functions - test_getAllerrorRate_s,test_compute_ECE and test_compute_ENCE:
        
        assert_series_equal(pd.Series(errorRate_s),pd.Series(getAllerrorRate_s(sig_level_2_prediction_intervals=sig_level_2_prediction_intervals,y_test=test_y)))
        assert ECE_new == compute_ECE(error_rate_s=errorRate_s,sig_levels=np.array([sig_level_of_interest]))

        
        expected_ENCE = compute_ENCE(y_true=test_y,y_pred=test_predictions,estimated_variance=[v**2 for v in half_pred_interval_sizes])
        assert approx(ENCE) == expected_ENCE
        ################

        assert validity == (36/40)
        abs_pi_sizes = [(2*v) for v in half_pred_interval_sizes]
        assert efficiency == np.mean(abs_pi_sizes)

        ################################0
        #computed in Excel:        
        assert approx(scc) == -0.183677298
        ################################


        assert no_compounds == 40
    
    def test_getIntervalsWidthOfInterest(self):
        from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import getIntervalsWidthOfInterest

        test_y,test_predictions,half_pred_interval_sizes,sig_level_of_interest,sig_level_2_prediction_intervals = self.get_toy_inputs_for_test_computeAllRegMetrics()

        widths = getIntervalsWidthOfInterest(sig_level_2_prediction_intervals,sig_level_of_interest)

        abs_pi_sizes = [(2*v) for v in half_pred_interval_sizes]

        assert_series_equal(pd.Series(np.array(abs_pi_sizes)),pd.Series(widths))


    def test_estimated_variance_from_CP(self):
        from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import estimated_variance_from_CP

        test_y,test_predictions,half_pred_interval_sizes,sig_level_of_interest,sig_level_2_prediction_intervals = self.get_toy_inputs_for_test_computeAllRegMetrics()

        abs_pi_sizes = [(2*v) for v in half_pred_interval_sizes]

        estimated_variance = estimated_variance_from_CP(intervals_width=np.array(abs_pi_sizes))

        assert isinstance(estimated_variance,list)

        assert_series_equal(pd.Series([v**2 for v in half_pred_interval_sizes]),pd.Series(estimated_variance))


    def get_inputs_for_test_workflowForOneSetOfCrossValidationAndOrMultipleSeedsResults_where_AD_method_works(self,regression=True,sig_level_of_interest=0.32,size_of_subsets=40):
        from modelsADuncertaintyPkg.utils.ML_utils import compute_pred_intervals



        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results = {}

        dict_for_all_folds_and_or_model_seeds_of_dicts_needed_for_this_test = defaultdict(dict)

        for fold_and_or_model_seed_combination in range(1,100,20):

            dict_of_raw_subset_results = defaultdict(dict)

            generator = np.random.default_rng(seed=fold_and_or_model_seed_combination)

            id_offset = 0

            #===============================
            #Added for the purposes of the test:

            dict_for_all_folds_and_or_model_seeds_of_dicts_needed_for_this_test[fold_and_or_model_seed_combination]['id2y'] = {}
            dict_for_all_folds_and_or_model_seeds_of_dicts_needed_for_this_test[fold_and_or_model_seed_combination]['id2_preds_y'] = {}
            #================================

            for subset in ['Inside','Outside']:
                id_offset += size_of_subsets

                dict_of_raw_subset_results[subset]['subset_test_ids'] = [(v+id_offset) for v in range(size_of_subsets)]
                
                if regression:
                    if 'Inside' == subset:
                        offset_scale_factor = 0.1
                        half_pred_interval_sizes = [(v*offset_scale_factor) for v in generator.random(size=size_of_subsets).tolist()]
                        
                    elif 'Outside' == subset:
                        offset_scale_factor = 0.5
                        half_pred_interval_sizes = [(v*offset_scale_factor*0.5) for v in generator.random(size=size_of_subsets).tolist()]
                    else:
                        raise Exception(f'Unexpected original split subset name: {subset}')
                    

                    experi_vals=generator.random(size=size_of_subsets).tolist()


                    offsets = [(v*offset_scale_factor) for v in generator.random(size=size_of_subsets).tolist()]
                    predictions=[(experi_vals[i]-offsets[i]) for i in range(size_of_subsets)]

                    reg_pred_intervals = compute_pred_intervals(y_pred=predictions,half_pred_interval_sizes=half_pred_interval_sizes)

                    


                    dict_of_raw_subset_results[subset]['subset_test_predictions'] = predictions

                    dict_of_raw_subset_results[subset]['subset_test_y'] = experi_vals

                    #class_1_probs = None

                    dict_of_raw_subset_results[subset]['subset_sig_level_2_prediction_intervals'] = {sig_level_of_interest:reg_pred_intervals}
                    
                    #delta_calib_plot_experi_probs = None

                    #===============================
                    #Added for the purposes of the test:

                    dict_for_all_folds_and_or_model_seeds_of_dicts_needed_for_this_test[fold_and_or_model_seed_combination]['id2y'].update(dict(zip(dict_of_raw_subset_results[subset]['subset_test_ids'],dict_of_raw_subset_results[subset]['subset_test_y'])))
                    dict_for_all_folds_and_or_model_seeds_of_dicts_needed_for_this_test[fold_and_or_model_seed_combination]['id2_preds_y'].update(dict(zip(dict_of_raw_subset_results[subset]['subset_test_ids'],dict_of_raw_subset_results[subset]['subset_test_predictions'])))
                    #================================

                    
                else:
                    raise Exception(f'Not currently supported!')
                    #dict_of_raw_subset_results[subset]['subset_predicted_y'] = predictions

                    #dict_of_raw_subset_results[subset]['subset_test_y'] = experi_vals

                    #dict_of_raw_subset_results[subset]['subset_probs_for_class_1'] = class_1_probs
                    
                    #reg_pred_intervals = None



            dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results
        
        return dict_for_all_folds_and_or_model_seeds_of_raw_subset_results,dict_for_all_folds_and_or_model_seeds_of_dicts_needed_for_this_test


    def get_subset_ys_and_predicted_ys_or_class_1_probs(self,list_of_ids,id2y,id2_preds_y):

        subset_y = [id2y[id_] for id_ in list_of_ids]
        

        subset_y_preds_or_class_1_probs = [id2_preds_y[id_] for id_ in list_of_ids]
        

        return subset_y,subset_y_preds_or_class_1_probs
    
    def get_this_split_shift_metric(self,split_name,dict_of_splits_matched_to_subset_ids,dict_of_raw_subset_results,metric_function,dicts_needed_for_this_test):
        ids_inside = dict_of_splits_matched_to_subset_ids[split_name]['Inside']
        ids_outside = dict_of_splits_matched_to_subset_ids[split_name]['Outside']

        y_inside,y_preds_or_class_1_probs_inside = self.get_subset_ys_and_predicted_ys_or_class_1_probs(list_of_ids=ids_inside,
                                                                                                                      id2y=dicts_needed_for_this_test['id2y'],
                                                                                                                      id2_preds_y=dicts_needed_for_this_test['id2_preds_y'])

            
        y_outside,y_preds_or_class_1_probs_outside = self.get_subset_ys_and_predicted_ys_or_class_1_probs(list_of_ids=ids_outside,
                                                                                                                      id2y=dicts_needed_for_this_test['id2y'],
                                                                                                                      id2_preds_y=dicts_needed_for_this_test['id2_preds_y'])


        metric_inside = metric_function(y_inside,y_preds_or_class_1_probs_inside)
        metric_outside = metric_function(y_outside,y_preds_or_class_1_probs_outside)

        shift_metric = (metric_inside-metric_outside)

        return shift_metric

        
            

    def check_workflowForOneSetOfCrossValidationAndOrMultipleSeedsResults_where_AD_method_works(self,regression,one_tail):
        from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import workflowForOneSetOfCrossValidationAndOrMultipleSeedsResults
        ##################################
        #The following function and the functions it calls were carefully proof-read. (I cannot see how to directly test the outputs of random sampling)
        from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import getDictOfSplitsMatchedToSubsetIDs
        #################################
        from modelsADuncertaintyPkg.qsar_eval.aggregate_p_vals import aggregate_p_vals

        if regression:
            delta_calib_prob = None
            
            type_of_modelling = 'regression'
            metric = 'R2'
        else:
            delta_calib_prob = 0.05
            type_of_modelling = 'binary_class'
            metric = 'AUC'
    
        subset_1_name = 'Inside'
        subset_2_name = 'Outside'
        no_rand_splits = 10
        strat_rand_split_y_name = 'Y_ready_for_stratified_splitting'
        rand_seed = 42



        dict_for_all_folds_and_or_model_seeds_of_raw_subset_results,dict_for_all_folds_and_or_model_seeds_of_dicts_needed_for_this_test = self.get_inputs_for_test_workflowForOneSetOfCrossValidationAndOrMultipleSeedsResults_where_AD_method_works(regression=regression,sig_level_of_interest=0.32)
  


        #For manual sanity checking:
        print(f'dict_for_all_folds_and_or_model_seeds_of_raw_subset_results={dict_for_all_folds_and_or_model_seeds_of_raw_subset_results}')
        
        #####################
        #better values are larger; the order of the experimental (1st) and predicted values (2nd) should be the same
        if 'R2' == metric:
            from modelsADuncertaintyPkg.qsar_eval.reg_perf_pred_stats import coeffOfDetermination
            metric_function = coeffOfDetermination
        elif 'AUC' == metric:
            from modelsADuncertaintyPkg.qsar_eval.all_key_class_stats_and_plots import computeAUC_TwoCategories
            metric_function = computeAUC_TwoCategories
        else:
            raise Exception(f'This checking function cannot handle this metric right now = {metric}')
        #####################

        metrics_of_interest = [metric]

        all_expected_p_vals_per_fold_and_or_model_seed_combination = []
        
        for fold_and_or_model_seed in dict_for_all_folds_and_or_model_seeds_of_raw_subset_results.keys():
            dict_of_raw_subset_results = dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed]
            ######
            #copied usage in function being tested from pkg code:
            dict_of_splits_matched_to_subset_ids = getDictOfSplitsMatchedToSubsetIDs(dict_of_raw_subset_results=dict_of_raw_subset_results,subset_1_name=subset_1_name,subset_2_name=subset_2_name,type_of_modelling=type_of_modelling,no_rand_splits=no_rand_splits,strat_rand_split_y_name=strat_rand_split_y_name,rand_seed=rand_seed,delta_calib_prob=delta_calib_prob,metrics_of_interest=metrics_of_interest)
            ######

            orig_shift_metric = self.get_this_split_shift_metric('Original',dict_of_splits_matched_to_subset_ids,dict_of_raw_subset_results,metric_function,dict_for_all_folds_and_or_model_seeds_of_dicts_needed_for_this_test[fold_and_or_model_seed])

            print(f'orig_shit_metric={orig_shift_metric}')

            count = 0 

            for rand_split in range(no_rand_splits):
                rand_split_shift_metric = self.get_this_split_shift_metric(f'Random-{rand_split}',dict_of_splits_matched_to_subset_ids,dict_of_raw_subset_results,metric_function,dict_for_all_folds_and_or_model_seeds_of_dicts_needed_for_this_test[fold_and_or_model_seed])

                print(f'rand_split_shift_metric={rand_split_shift_metric}')

                ##################
                #based upon choice of metric for unit-test (see above)
                if one_tail:
                    if rand_split_shift_metric >= orig_shift_metric:
                        count += 1
                else:
                    if abs(rand_split_shift_metric) >= abs(orig_shift_metric):
                        count += 1
                ##################
            
            this_fold_or_seed_p_value = (count/no_rand_splits)

            print(f'this_fold_of_seed_p_value={this_fold_or_seed_p_value}')

            all_expected_p_vals_per_fold_and_or_model_seed_combination.append(this_fold_or_seed_p_value)
        
        expected_p_val = aggregate_p_vals(all_expected_p_vals_per_fold_and_or_model_seed_combination)


        ######################
        #copied the use of this function from the package
        #set_p_to_nan_if_metric_cannot_be_computed_for_orig_split is always True in scripts

        dict_of_metrics_matched_to_p_vals,dict_of_metrics_matched_to_p_vals_metadata = workflowForOneSetOfCrossValidationAndOrMultipleSeedsResults(dict_for_all_folds_and_or_model_seeds_of_raw_subset_results,out_dir='n.a',test_set_plus_methods_name='n.a',subset_1_name='Inside',subset_2_name='Outside',type_of_modelling=type_of_modelling,no_rand_splits=no_rand_splits,strat_rand_split_y_name='Strat_Y',rand_seed=rand_seed,metrics_of_interest=[metric],x_lab='n.a',y_lab='n.a',legend_lab='n.a',sig_level_perc=5,one_sided_sig_test=one_tail,metric_to_expected_sub_1_minus_sub_2_sign={metric:1},create_plots=False,set_p_to_nan_if_metric_cannot_be_computed_for_orig_split=True,conformal_sig_level=0.32,delta_calib_prob=delta_calib_prob,debug=True)
        #####################
        
        cmp_msg = f'expected_p_val={expected_p_val} vs. dict_of_metrics_matched_to_p_vals[metric]={dict_of_metrics_matched_to_p_vals[metric]}'

        print(cmp_msg)

        assert expected_p_val == dict_of_metrics_matched_to_p_vals[metric]

 
    def test_workflowForOneSetOfCrossValidationAndOrMultipleSeedsResults_regression__one_tail_p_value_where_AD_method_works(self):
        

        
        #-----------------------------------
        regression = True
        one_tail = True
        #-----------------------------------

        self.check_workflowForOneSetOfCrossValidationAndOrMultipleSeedsResults_where_AD_method_works(regression,one_tail)
    
    def test_workflowForOneSetOfCrossValidationAndOrMultipleSeedsResults_regression__two_tail_p_value_where_AD_method_works(self):
        

        
        #-----------------------------------
        regression = True
        one_tail = False
        #-----------------------------------

        self.check_workflowForOneSetOfCrossValidationAndOrMultipleSeedsResults_where_AD_method_works(regression,one_tail)

        
    def test_compute_all_raw_shift_metric_p_vals_WITH_one_tail_calc_VIA_pretend_metrics_designed_to_check_different_scenarios(self):
        from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import compute_all_raw_shift_metric_p_vals

        #================================
        one_tail = True

        metric_to_expected_sub_1_minus_sub_2_sign = {'M1':1,'M2':-1,'M3':1,'M4':-1,'M5':1,'M6':1,'M7':-1}

        metrics_of_interest = ['M1','M2','M3','M4','M5','M7']

        metrics_for_which_shift_metric_cannot_be_computed_for_orig_split = ['M5']


        expected_dict_of_metrics_matched_to_pvals = {'M1':0.04,'M2':0.02,'M3':0.50,'M4':1.0,'M5':np.nan,'M7':0.25}

        ##################
        #c.f. def getForAllMetricsOriginalVsRandShiftMetricVals(...):

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals = neverEndingDefaultDict()

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M1']['original'] = 0.82
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M1']['random'] = [0.82,0.90,0.87,0.83]+[0.51]*92+[-0.82,-0.95,-1.0,-0.99]

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M2']['original'] = -0.82
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M2']['random'] = [-0.82,-0.90]+[0.90]*98

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M3']['original'] = 0.82
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M3']['random'] = [0.82]*50+[0.60]*50

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M4']['original'] = 0.82
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M4']['random'] = [0.82]*25+[0.60]*75

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M5']['original'] = 'irrelevant'
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M5']['random'] = [0.82]*25+[0.60]*75

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M6']['original'] = 'irrelevant'
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M6']['random'] = [0.82]*25+[0.60]*75

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M7']['original'] = -0.82
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M7']['random'] = [-0.82]*25+[-0.60]*75
        ##################

        #===============================


        dict_of_metrics_matched_to_pvals = compute_all_raw_shift_metric_p_vals(dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals,one_sided_sig_test=one_tail,metric_to_expected_sub_1_minus_sub_2_sign=metric_to_expected_sub_1_minus_sub_2_sign,metrics_of_interest=metrics_of_interest,metrics_for_which_shift_metric_cannot_be_computed_for_orig_split=metrics_for_which_shift_metric_cannot_be_computed_for_orig_split)

        #---------------------------------

        np.testing.assert_equal(dict_of_metrics_matched_to_pvals,expected_dict_of_metrics_matched_to_pvals)

    def test_compute_all_raw_shift_metric_p_vals_WITH_two_tail_calc_VIA_pretend_metrics_designed_to_check_different_scenarios(self):
        from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import compute_all_raw_shift_metric_p_vals

        #================================
        one_tail = False

        metric_to_expected_sub_1_minus_sub_2_sign = {'M1':1,'M2':-1,'M3':1,'M4':-1,'M5':1,'M6':1,'M7':-1}

        metrics_of_interest = ['M1','M2','M3','M4','M5','M7']

        metrics_for_which_shift_metric_cannot_be_computed_for_orig_split = ['M5']


        expected_dict_of_metrics_matched_to_pvals = {'M1':0.08,'M2':1.0,'M3':0.50,'M4':0.25,'M5':np.nan,'M7':0.25}

        ##################
        #c.f. def getForAllMetricsOriginalVsRandShiftMetricVals(...):

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals = neverEndingDefaultDict()

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M1']['original'] = 0.82
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M1']['random'] = [0.82,0.90,0.87,0.83]+[0.51]*92+[-0.82,-0.95,-1.0,-0.99]

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M2']['original'] = -0.82
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M2']['random'] = [-0.82,-0.90]+[0.90]*98

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M3']['original'] = 0.82
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M3']['random'] = [0.82]*50+[0.60]*50

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M4']['original'] = 0.82
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M4']['random'] = [0.82]*25+[0.60]*75

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M5']['original'] = 'irrelevant'
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M5']['random'] = [0.82]*25+[0.60]*75

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M6']['original'] = 'irrelevant'
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M6']['random'] = [0.82]*25+[0.60]*75

        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M7']['original'] = -0.82
        
        dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals['M7']['random'] = [-0.82]*25+[-0.60]*75
        ##################

        #===============================


        dict_of_metrics_matched_to_pvals = compute_all_raw_shift_metric_p_vals(dict_of_metrics_matched_to_orig_and_rand_shift_metric_vals,one_sided_sig_test=one_tail,metric_to_expected_sub_1_minus_sub_2_sign=metric_to_expected_sub_1_minus_sub_2_sign,metrics_of_interest=metrics_of_interest,metrics_for_which_shift_metric_cannot_be_computed_for_orig_split=metrics_for_which_shift_metric_cannot_be_computed_for_orig_split)

        #---------------------------------

        np.testing.assert_equal(dict_of_metrics_matched_to_pvals,expected_dict_of_metrics_matched_to_pvals)


    def test_applyConformalRegressionToNewCmpds_ICP_intervals_get_larger_with_smaller_sig_level(self):

        from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import getIntervalsWidthOfInterest

        train_inc_calib_x,train_inc_calib_y,test_x,test_y,cal_fract,seed,calib_size = self.get_inputs_required_for_checking_conformal_regression_workflow_functions()
        
        _no_trees = 40

        confScores,model = RegUncert.getConformalRegressionModelsPlusCalibDetails_ICP(train_inc_calib_x,train_inc_calib_y,global_random_seed=seed,calib_fraction=cal_fract,ml_alg="RandomForestRegressor",nrTrees=_no_trees,non_conformity_scaling="exp(stdev of tree predictions)",stratified=False)

        sig_level_1 = 0.32

        testPred_list_sig_level_1,intervals_sig_level_1 = RegUncert.applyConformalRegressionToNewCmpds_ICP(confScores,model,X_test=test_x,sig_level=sig_level_1,ml_alg="RandomForestRegressor",nrTrees=_no_trees,non_conformity_scaling="exp(stdev of tree predictions)")

        sig_level_2 = 0.20

        testPred_list_sig_level_2,intervals_sig_level_2 = RegUncert.applyConformalRegressionToNewCmpds_ICP(confScores,model,X_test=test_x,sig_level=sig_level_2,ml_alg="RandomForestRegressor",nrTrees=_no_trees,non_conformity_scaling="exp(stdev of tree predictions)")

        np.testing.assert_equal(testPred_list_sig_level_1,testPred_list_sig_level_2)

        widths_1 = getIntervalsWidthOfInterest({sig_level_1:intervals_sig_level_1},sig_level_1)

        widths_2 = getIntervalsWidthOfInterest({sig_level_2:intervals_sig_level_2},sig_level_2)

        assert isinstance(widths_1,np.ndarray)
        assert isinstance(widths_2,np.ndarray)

        assert len(widths_1)==len(widths_2)

        for index in range(len(widths_1)):
            assert widths_2[index] > widths_1[index]
    
    def test_computeNonConformityScores(self):
        from modelsADuncertaintyPkg.qsar_reg_uncertainty.RegressionICP import computeNonConformityScores

        for input_type in ['arrays','series']:
            if 'lists' == input_type:
                nc_scores = computeNonConformityScores(pred=[1.5,1.4,2.5], y=[1.0,1.3,3.0], lamda_calib=[0.5,0.2,0.5])
            elif 'arrays' == input_type:
                nc_scores = computeNonConformityScores(pred=np.array([1.5,1.4,2.5]), y=np.array([1.0,1.3,3.0]), lamda_calib=np.array([0.5,0.2,0.5]))
            elif 'series' == input_type:
                nc_scores = computeNonConformityScores(pred=pd.Series([1.5,1.4,2.5]), y=pd.Series([1.0,1.3,3.0]), lamda_calib=pd.Series([0.5,0.2,0.5]))
            else:
                raise Exception(f'Unrecognised input_type={input_type}')
            _nc_scores = list(nc_scores)

            assert [1.0,0.5,1.0] == approx(_nc_scores)
    
    def test_sort_numpy_array_in_descending_order(self):
        from modelsADuncertaintyPkg.utils.basic_utils import sort_numpy_array_in_descending_order

        a_ = np.array([2.1,0.68,1.5,3.4])

        a_descending = sort_numpy_array_in_descending_order(a_)

        np.testing.assert_array_equal(a_descending,np.array([3.4,2.1,1.5,0.68]))

    def get_tsne_checking_inputs(self,subset_col):

        smiles_col = 'SMILES'

        list_of_toy_smiles = []
        list_of_subset_labels = []


        aryl_smiles_start = 'Nc1ccccc1C'
        alkyl_smiles_start = 'OCC'
        for subset in ['aryl','alkyl']:
            if 'aryl' == subset:
                trivial_extension_smiles = aryl_smiles_start
                ext = 'Cc1ccccc1'
                n = 50
            elif 'alkyl' == subset:
                trivial_extension_smiles = alkyl_smiles_start
                ext = 'C'
                n = 70
            else:
                raise Exception(f'subset={subset} not recognised!')

            for i in range(n):
                trivial_extension_smiles = trivial_extension_smiles+ext
                list_of_toy_smiles.append(trivial_extension_smiles)
                list_of_subset_labels.append(subset)
        
        data_df = pd.DataFrame({subset_col:list_of_subset_labels,smiles_col:list_of_toy_smiles})
        
        data_df_with_fp_bit_cols_too = ChemDataParser.addFPBitsColsToDfWithSMILES(df_with_smiles=data_df,smiles_col=smiles_col)

        dataset_df = data_df_with_fp_bit_cols_too.drop([smiles_col],axis=1)

        #----------------------
        assert dataset_df.shape[0] == data_df_with_fp_bit_cols_too.shape[0]
        assert dataset_df.shape[1] == (data_df_with_fp_bit_cols_too.shape[1] - 1)
        #----------------------

        return dataset_df

    def test_get_tsne_plot_of_dataset_with_fp_bit_vector_array_and_subset_labels(self): 
        from modelsADuncertaintyPkg.CheminformaticsUtils.visualize_chemical_space import get_tsne_plot_of_dataset_with_fp_bit_vector_array_and_subset_labels
        
        plot_name_prefix = 'test.tsne.defaults'
        title_prefix = plot_name_prefix
        subset_col = 'Chemotype'
         
        dataset_df = self.get_tsne_checking_inputs(subset_col)
        
        get_tsne_plot_of_dataset_with_fp_bit_vector_array_and_subset_labels(plot_name_prefix,title_prefix,dataset_df,subset_col)


    def test_aggregate_p_vals_default(self):
        from modelsADuncertaintyPkg.qsar_eval.aggregate_p_vals import aggregate_p_vals

        examples_dict = defaultdict(dict)
        
        examples_dict[1]['input'] = [0.05,0.04,0.02,0.01]
        examples_dict[1]['expected_output'] = 0.06

        examples_dict[2]['input'] = [np.nan,0.05,0.04,0.02,0.01,None]
        examples_dict[2]['expected_output'] = 0.06

        examples_dict[3]['input'] = [0.60,0.78,0.79,0.83]
        examples_dict[3]['expected_output'] = 1.0


        for eg in examples_dict.keys():
            print(f'test_aggregate_p_vals_default : example {eg}')
            input = examples_dict[eg]['input']
            expected_output = examples_dict[eg]['expected_output']

            assert expected_output == aggregate_p_vals(input)








        





        










