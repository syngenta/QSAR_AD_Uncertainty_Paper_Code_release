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
#The following functions were originally written (in extraFunctions.py), with support from RMR, by Zied Hosni, as part of a Syngenta funded collaboration with the University of Sheffield
#######################
import pandas as pd
#-------------------------------
from ..utils.ML_utils import makeYLabelsNumeric
from ..qsar_AD.SingleTcDist1NNAD import TanimotoSplitting
from ..qsar_AD.SingleTcDist1NNAD import coreTanimotoSplitting
from ..qsar_AD import UNC_like_adapted_AD_approach as UNC_like_AD
from ..qsar_AD import dk_NN_thresholds as dk_NN_AD
from ..qsar_AD import rdn_update_dk_NN_thresholds as rdn_AD
from ..CheminformaticsUtils import chem_data_parsing_utils as ChemDataParser
#-------------------------------

def getInsideOutsideADSubsets(test_set_df, id_col, test_id2ADStatus):
    try:
        inside_AD_ids = [id_ for id_ in test_id2ADStatus.keys() if test_id2ADStatus[id_]['InsideAD']]

        outside_AD_ids = [id_ for id_ in test_id2ADStatus.keys() if not test_id2ADStatus[id_]['InsideAD']]
    except KeyError as err:
        print(f'getInsideOutsideADSubsets(...): test_id2ADStatus.keys()={list(test_id2ADStatus.keys())}')
        raise Exception(err)

    assert 0 == len(set(outside_AD_ids).intersection(set(inside_AD_ids)))

    if not 0 == len(inside_AD_ids):
        test_insideAD_df = test_set_df[test_set_df[id_col].isin(inside_AD_ids)]
    else:
        print('No compounds inside AD!')
        test_insideAD_df = None

    if not 0 == len(outside_AD_ids):
        test_outsideAD_df = test_set_df[test_set_df[id_col].isin(outside_AD_ids)]
    else:
        print('No compounds outside AD!')
        test_outsideAD_df = None
    return test_insideAD_df, test_outsideAD_df

def findInsideOutsideADTestIds(X_train, X_test, fps_train, fps_test, threshold, id_col,rand_seed,endpoint_col='Class', AD_method_name='Tanimoto', test_ids=None, y_train=None, y_test=None, regression=False,class_1_label='Active',class_0_label='Inactive',consistent_distance_metric='jaccard',debug=False,expected_descs_no=1024):
    ##########################################
    #rand_seed is only relevant for RDN. All other AD methods are completely deterministic.
    #X_train and X_test should actually refer to X_train_plus_IDs_dataframe and X_test_plus_ids_dataframe, where X denotes the descriptors (fingerprint bit-vectors for now)!
    #fps_train and fps_test should just refer to dataframes containing the raw fingerprint objects
    ########################################
    #=============================================
    if not isinstance(X_train,pd.DataFrame): raise Exception(f'type(X_train)={type(X_train)}')
    if not isinstance(X_test,pd.DataFrame): raise Exception(f'type(X_test)={type(X_test)}')
    if not id_col in X_train.columns.values.tolist(): raise Exception(f'X_train.columns.values.tolist()={X_train.columns.values.tolist()}')
    if not id_col in X_test.columns.values.tolist(): raise Exception(f'X_test.columns.values.tolist()={X_test.columns.values.tolist()}')
    if endpoint_col in X_train.columns.values.tolist(): raise Exception(f'X_train.columns.values.tolist()={X_train.columns.values.tolist()}')
    if endpoint_col in X_test.columns.values.tolist(): raise Exception(f'X_test.columns.values.tolist()={X_test.columns.values.tolist()}')
    if not X_train.columns.values.tolist() == X_test.columns.values.tolist(): raise Exception(f'X_train.columns.values.tolist() = {X_train.columns.values.tolist()} vs. X_test.columns.values.tolist()={X_test.columns.values.tolist()}')
    if not expected_descs_no is None:
        if not (expected_descs_no + 1) == X_train.shape[1]: raise Exception(f'X_train should have columns for descriptors (expected number = {expected_descs_no}) and an IDs column only. X_train.shape[1]={X_train.shape[1]}')
        if not (expected_descs_no + 1) == X_test.shape[1]: raise Exception(f'X_test should have columns for descriptors (expected number = {expected_descs_no}) and an IDs column only.X_test.shape[1]={X_test.shape[1]}')
    if not isinstance(fps_train,pd.DataFrame): raise Exception(f'type(fps_train)={type(fps_train)}')
    if not isinstance(fps_test,pd.DataFrame): raise Exception(f'type(fps_test)={type(fps_test)}')
    if not isinstance(y_train,pd.Series): raise Exception(f'type(y_train)={y_train}')
    if not y_test is None: raise Exception('This should never be relevant!')
    #==============================================
    
    X_train = X_train.copy(deep=True)

    #===============================================
    if test_ids is None:
        ids_are_not_already_supplied = True
    else:
        ids_are_not_already_supplied = False
        if not isinstance(test_ids,list): raise Exception(f'test_ids={test_ids}')
        if not X_test[id_col].tolist() == test_ids: raise Exception(f'X_test[id_col].tolist()={X_test[id_col].tolist()} vs. test_ids={test_ids}')
        test_ids = pd.Series(test_ids)
    #===============================================
    
    if AD_method_name == 'Tanimoto':  # Tanimoto_Splitting:
        assert 'jaccard' == consistent_distance_metric,f'consistent_distance_metric={consistent_distance_metric}'
        test_id_ad_status_dict = TanimotoSplitting(fps_train, fps_test, threshold, test_ids)
    else: #RLMR: Remove redundancy
        #==================================
        if ids_are_not_already_supplied:
            X_train = X_train.reset_index(drop=True)
            X_train[id_col] = X_train.index
            
            X_test = X_test.reset_index(drop=True)
            X_test[id_col] = X_test.index
            
        #==================================
        if AD_method_name == 'dkNN':
            
            dk_NN_thresholds_instance = dk_NN_AD.dk_NN_thresholds(train_df=X_train, id_col=id_col, k=threshold,
                                                         distance_metric=consistent_distance_metric, scale=False,expected_descs_no=expected_descs_no)
            dk_NN_thresholds_instance.getInitialTrainingSetThresholds()
            dk_NN_thresholds_instance.updateZeroValuedTrainingSetThresholds()
            test_id_ad_status_dict = dk_NN_thresholds_instance.getADstatusOfTestSetCompounds(
                test_compounds_X_df=X_test, id_col=id_col)
        elif AD_method_name == 'UNC':
            
            UNC_like_AD_Estimator = UNC_like_AD.UNC_like_AD_approach(train_df=X_train, id_col=id_col, k=threshold,
                                                         distance_metric=consistent_distance_metric, scale=False,
                                                         endpoint_col=None,expected_descs_no=expected_descs_no)
            UNC_like_AD_Estimator.getTrainingSetStats()
            test_id_ad_status_dict = UNC_like_AD_Estimator.getADstatusOfTestSetCompounds(
                test_compounds_X_df=X_test, id_col=id_col, p_value_thresh=0.05,debug=debug)
        elif AD_method_name == 'RDN':
                       
            y_train = y_train.reset_index(drop=True)
            #==============================
            if not X_train.index.tolist()==y_train.index.tolist(): raise Exception(f'Indices mismatch!')
            #==============================
            X_train[endpoint_col] = y_train
            if not regression:
                X_train[endpoint_col] = makeYLabelsNumeric(X_train[endpoint_col], class_1=class_1_label,class_0=class_0_label)
            else:
                pass
            
            X_train = X_train.astype({id_col: 'int'})
            
            
            RDN_thresholds_instance = rdn_AD.RDN_thresholds(train_df=X_train,
                                                     id_col=id_col, k=threshold,
                                                     distance_metric=consistent_distance_metric,
                                                     scale=False,
                                                     endpoint_col=endpoint_col,expected_descs_no=expected_descs_no)
            RDN_thresholds_instance.assignRdnTrainingSetSpecificThresholds(id_col=id_col,
                                                                           endpoint_col=endpoint_col,
                                                                           n_models_in_RDN_ensemble=
                                                                           10,
                                                                           cv_folds_no=2,
                                                                           SEED=rand_seed,
                                                                           use_k_fold_cv_for_ensemble_preds=
                                                                           True,
                                                                           use_sklearn_rf=True,
                                                                           round_dt_preds=False,
                                                                           write_out_intermediate_res_for_checks=False, regression=regression)

            test_id_ad_status_dict = RDN_thresholds_instance.getADstatusOfTestSetCompounds(
                test_compounds_X_df=X_test, id_col=id_col)
        else: #RLMR: replace ifs with elifs above to enable this
            raise Exception('Unrecognised AD_method_name = {}'.format(AD_method_name))
    
    return (test_id_ad_status_dict)

def getInputRequiredForModellingAndAD_OneSubset(data_df,id_col,smiles_col,class_label_col,class_1,fp_col='fp',bond_radius=2,bitInfo={},nBits=1024,pre_calc_stand_mol_col=None,pre_calc_fps=False,type_of_fp='Morgan'):
    
    data_df.reset_index(drop=True,inplace=True)
    
    if not pre_calc_fps:
        data_df,fps_df = ChemDataParser.getFpsDfAndDataDfSkippingFailedMols(data_df,smiles_col=smiles_col,id_col=id_col,fp_col=fp_col,bond_radius=bond_radius,bitInfo=bitInfo,nBits=nBits,pre_calc_stand_mol_col=pre_calc_stand_mol_col,type_of_fp=type_of_fp)
        
        data_df.insert(0,fp_col,fps_df[fp_col].tolist(),allow_duplicates=True)
    else:
        fps_df = data_df[[fp_col]]
        
    data_ids = data_df[id_col].tolist()
    
    data_df_with_fp_bit_cols_too = ChemDataParser.addFPBitsColsToDfWithSMILES(df_with_smiles=data_df,smiles_col=smiles_col,bond_radius=bond_radius,bitInfo=bitInfo,nBits=nBits,pre_calc_stand_mol_col=pre_calc_stand_mol_col,pre_calc_fp_col=fp_col,type_of_fp=type_of_fp)
    
    data_y = data_df[class_label_col]
    
    X_and_ids_df = data_df_with_fp_bit_cols_too.drop([smiles_col,class_label_col],axis=1) #fp_col should be dropped inside addFPBitsColsToDfWithSMILES(...), when fp_col is not None!
    
    #--------------------------------
    #This should still be the case if some FP calculations failed:
    assert X_and_ids_df.shape[0] == fps_df.shape[0],f"X_and_ids_df.shape[0]={X_and_ids_df.shape[0]},fps_df.shape[0]={fps_df.shape[0]}"
    assert X_and_ids_df.shape[0] == len(data_ids),f"X_and_ids_df.shape[0]={X_and_ids_df.shape[0]},len(data_ids)={len(data_ids)}"
    assert X_and_ids_df.shape[0] == len(data_y),f"X_and_ids_df.shape[0]={X_and_ids_df.shape[0]},len(data_y)={len(data_y)}"
    #The following could be relevant, e.g. Tanimoto (d1NN) AD calculations:
    assert list(X_and_ids_df.index) == list(fps_df.index),f"list(X_and_ids_df.index) = {list(X_and_ids_df.index)}, list(fps_df.index) = {list(fps_df.index)}"
    assert list(X_and_ids_df.index) == list(data_y.index),f"list(X_and_ids_df.index) = {list(X_and_ids_df.index)}, list(data_y.index) = {list(data_y.index)}"
    assert list(X_and_ids_df.index) == list(range(len(data_ids))),f"list(X_and_ids_df.index) = {list(X_and_ids_df.index)}, list(range(len(data_ids))) = {list(range(len(data_ids)))}"
    #--------------------------------
    
    return X_and_ids_df,fps_df,data_y,data_ids

def getInputRequiredForModellingAndAD(train_df,test_df,id_col,smiles_col,class_label_col,class_1,fp_col='fp',bond_radius=2,bitInfo={},nBits=1024,pre_calc_stand_mol_col=None,pre_calc_fps=False,type_of_fp='Morgan'):
    
    for train_test in ['Train','Test']:
        #==================================
        if 'Train' == train_test:
            data_df = train_df
        elif 'Test' == train_test:
            data_df = test_df
        else:
            raise Exception('Unrecognised train_test label = {}'.format(train_test))
        #=================================
        
        X_and_ids_df,fps_df,data_y,data_ids = getInputRequiredForModellingAndAD_OneSubset(data_df,id_col,smiles_col,class_label_col,class_1,fp_col=fp_col,bond_radius=bond_radius,bitInfo=bitInfo,nBits=nBits,pre_calc_stand_mol_col=pre_calc_stand_mol_col,pre_calc_fps=pre_calc_fps,type_of_fp=type_of_fp)
        
        #==================================
        if 'Train' == train_test:
            #------------------------
            fps_train = fps_df
            
            train_ids = data_ids
            
            train_y = data_y
            
            X_train_and_ids_df = X_and_ids_df
            #------------------------
        elif 'Test' == train_test:
            #------------------------
            fps_test = fps_df
            
            test_ids = data_ids
            
            test_y = data_y
            
            X_test_and_ids_df = X_and_ids_df
            #------------------------
        else:
            raise Exception('Unrecognised train_test label = {}'.format(train_test))
        #=================================
    
    
    return fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids

def getADSubsetTestIDsInOrder(AD_subset,test_id_status_dict,test_ids):
    
    #----------------------
    assert all([_id in test_id_status_dict.keys() for _id in test_ids])
    #---------------------

    try:
        if 'All' == AD_subset:
            subset_test_ids = test_ids[:]
        elif 'Inside' == AD_subset:
            subset_test_ids = [id_ for id_ in test_ids if test_id_status_dict[id_]['InsideAD']]
        elif 'Outside' == AD_subset:
            subset_test_ids = [id_ for id_ in test_ids if not test_id_status_dict[id_]['InsideAD']]
        else:
            raise Exception('Unrecognised subset label = {}'.format(AD_subset))
    except KeyError as err:
        print(f'getADSubsetTestIDsInOrder(...): problem AD subset={AD_subset}. test_id_status_dict[test_ids[0]].keys()={list(test_id_status_dict[test_ids[0]].keys())}')
        raise Exception(err)
    
    return subset_test_ids
