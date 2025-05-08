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
#######################
import sys,re,os
import numpy as np,pandas as pd
from pandas.testing import assert_frame_equal
from collections import defaultdict
from rdkit.Chem import AllChem,MolToInchi
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.MolStandardize import rdMolStandardize
try:
    from standardiser import unsalt #eTox standardiser module
except ImportError:
    print('Warning: standardiser module is not installed! Some functionality will not work!')
from .standardize_mols import standardize
from .globals_for_checking import simple_smiles,simple_smiles_salt,complex_smiles,simple_smiles_inchi,problem_mol,expected_fp_bits_list_of_simple_mol,expected_fp_bits_list_of_complex_mol
from ..utils.average_all_bioact_vals_for_same_mol_after_exclude_outliers import getAveragePlusDeviationEstimateForBioactAfterRemovingOutliers
from ..utils.basic_utils import findDups
##############################
mol_parse_status_col = 'MolParsedOK'

def is_valid_smiles(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if not mol is None:
        return True
    else:
        return False

def getInChIFromMol(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    
    if mol is None:
        return None
    else:
        return MolToInchi(mol)

def check_getInChIFromMol():
    i = getInChIFromMol(simple_smiles)
    
    assert i == simple_smiles_inchi,"simple_smiles = {}. Expected InChI = {}. InChI = {}.".format(simple_smiles,simple_smiles_inchi,i)
    
    print('RAN check_getInChIFromMol()')

def getInChIFromStandardisedMol(smiles_string):
    standardised_mol = standardize(smiles_string)
    
    if standardised_mol is None:
        return None
    else:
        i = MolToInchi(standardised_mol)
        return i

def check_getInChIFromStandardisedMol_simple_smiles_salt():
    i = getInChIFromStandardisedMol(simple_smiles_salt)
    
    assert i == simple_smiles_inchi,"simple_smiles_salt = {}. Expected InChI = {}. InChI = {}.".format(simple_smiles_salt,simple_smiles_inchi,i)
    
    print('RAN check_getInChIFromStandardisedMol_simple_smiles_salt()')

def getFPFromStandardisedMol(smiles_string,bond_radius=2,bitInfo={},nBits=1024,pre_calc_stand_mol=None,type_of_fp='Morgan'):
    if pre_calc_stand_mol is None:
        standardised_mol = standardize(smiles_string)
    else:
        standardised_mol = pre_calc_stand_mol
    
    if standardised_mol is None:
        return None
    else:
        try:
            if 'Morgan' == type_of_fp:
                fp = AllChem.GetMorganFingerprintAsBitVect(standardised_mol, bond_radius, bitInfo=bitInfo, nBits=nBits)
            else:
                raise Exception(f'type_of_fp={type_of_fp} is not yet implemented!')
            return fp
        except Exception as err:
            print('Problem generating fingerprint after standardizing %s: %s' % (smiles_string,str(err)))
            return None

def convert_bit_vector_fingerprint_to_bits_list(fp):
    if not fp is None:
        m=np.asmatrix(fp)
        
        fp_bits_list = m.tolist()[0]
        
        
        return fp_bits_list
    else:
        return None


def getFPFromStandardisedMol_BitsList(smiles_string,bond_radius=2,bitInfo={},nBits=1024,pre_calc_stand_mol=None,pre_calc_fp=None,type_of_fp='Morgan'):
    
    if pre_calc_fp is None:
        fp = getFPFromStandardisedMol(smiles_string,bond_radius=bond_radius,bitInfo=bitInfo,nBits=nBits,pre_calc_stand_mol=pre_calc_stand_mol,type_of_fp=type_of_fp)
    else:
        fp = pre_calc_fp
    
    return convert_bit_vector_fingerprint_to_bits_list(fp)

def compareTwoFPBitLists(fp_bits_list,expected_fp_bits_list):
    
    fp_bits_list_df = pd.DataFrame(fp_bits_list).transpose()
    
    expected_fp_bits_list_df = pd.DataFrame(expected_fp_bits_list).transpose()
    
    assert_frame_equal(fp_bits_list_df,expected_fp_bits_list_df)


def check_getFPFromStandardisedMol_BitsList_SimpleMol():
    
    print('Running check_getFPFromStandardisedMol_BitsList_SimpleMol()')
    
    fp_bits_list = getFPFromStandardisedMol_BitsList(simple_smiles)
    
    assert 1024 == len(fp_bits_list),len(fp_bits_list)
    
    expected_fp_bits_list = expected_fp_bits_list_of_simple_mol
    
    
    compareTwoFPBitLists(fp_bits_list,expected_fp_bits_list)
    
    print('RAN check_getFPFromStandardisedMol_BitsList_SimpleMol()')

#check_getFPFromStandardisedMol_BitsList_SimpleMol()

def check_getFPFromStandardisedMol_BitsList_SimpleMolSalt():
    
    print('Running check_getFPFromStandardisedMol_BitsList_SimpleMolSalt()')
    
    fp_bits_list = getFPFromStandardisedMol_BitsList(simple_smiles_salt)
    
    assert 1024 == len(fp_bits_list),len(fp_bits_list)
    
    expected_fp_bits_list = expected_fp_bits_list_of_simple_mol
    
    compareTwoFPBitLists(fp_bits_list,expected_fp_bits_list)
    
    print('RAN check_getFPFromStandardisedMol_BitsList_SimpleMolSalt()')

#check_getFPFromStandardisedMol_BitsList_SimpleMolSalt()

def check_getFPFromStandardisedMol_BitsList_ProblemMol():
    
    print('Running check_getFPFromStandardisedMol_BitsList_ProblemMol()')
    
    fp_bits_list = getFPFromStandardisedMol_BitsList(problem_mol)
    
    assert fp_bits_list is None,fp_bits_list
    
    print('RAN check_getFPFromStandardisedMol_BitsList_ProblemMol()')

#check_getFPFromStandardisedMol_BitsList_ProblemMol()

def addFPBitsColsToDf_Row_WithSMILES(row,smiles_col,nBits=1024,bond_radius=2,bitInfo={},report_problem_smiles=False,mol_parse_status_col=mol_parse_status_col,pre_calc_stand_mol_col=None,pre_calc_fp_col=None,type_of_fp='Morgan'):
    
    if (pre_calc_stand_mol_col is None and pre_calc_fp_col is None):
        fp_bits_list = getFPFromStandardisedMol_BitsList(smiles_string=row[smiles_col],bond_radius=bond_radius,bitInfo=bitInfo,nBits=nBits,type_of_fp=type_of_fp)
    elif (pre_calc_fp_col is None and not pre_calc_stand_mol_col is None):
        fp_bits_list = getFPFromStandardisedMol_BitsList(smiles_string=None,bond_radius=bond_radius,bitInfo=bitInfo,nBits=nBits,pre_calc_stand_mol=row[pre_calc_stand_mol_col],type_of_fp=type_of_fp)
    elif (not pre_calc_fp_col is None):
        fp_bits_list = getFPFromStandardisedMol_BitsList(smiles_string=None,bond_radius=bond_radius,bitInfo=bitInfo,nBits=nBits,pre_calc_fp=row[pre_calc_fp_col],type_of_fp=type_of_fp)
    else:
        raise Exception(f'Unrecognised scenario: pre_calc_stand_mol_col={pre_calc_stand_mol_col}, pre_calc_fp_col={pre_calc_fp_col}')
        
    
    if not fp_bits_list is None:
    
        fp_bits_series = pd.Series(fp_bits_list,index=[str(i) for i in range(0,nBits)])
    
        row = pd.concat([row,fp_bits_series])
        
        row[mol_parse_status_col] = True
    else:
        row[mol_parse_status_col] = None
        
        if report_problem_smiles:
            print('This SMILES could not be converted to a fingerprint and will be dropped: {}'.format(row[smiles_col]))
        
        row = pd.concat([row,pd.Series([0]*nBits,index=[str(i) for i in range(0,nBits)])])
    
    return row

def addFPBitsColsToDfWithSMILES(df_with_smiles,smiles_col,nBits=1024,bond_radius=2,bitInfo={},mol_parse_status_col=mol_parse_status_col,report_problem_smiles=False,pre_calc_stand_mol_col=None,pre_calc_fp_col=None,type_of_fp='Morgan'):
    
    df = df_with_smiles.apply(addFPBitsColsToDf_Row_WithSMILES,axis=1,args=(smiles_col,nBits,bond_radius,bitInfo,report_problem_smiles,mol_parse_status_col,pre_calc_stand_mol_col,pre_calc_fp_col,type_of_fp))
    
    df = df[~df[mol_parse_status_col].isna()]
    
    #==================================
    cols_to_drop = [mol_parse_status_col] #mol_parse_status_col is redundant if dropping problems!
    
    if not pre_calc_stand_mol_col is None:
        cols_to_drop.append(pre_calc_stand_mol_col)
    
    if not pre_calc_fp_col is None:
        cols_to_drop.append(pre_calc_fp_col)
    #==================================
    
    df = df.drop(labels=cols_to_drop,axis=1) 
    
    df.reset_index(drop=True,inplace=True)
    
    return df

def check_addFPBitsColsToDfWithSMILES():
    print('Running check_addFPBitsColsToDfWithSMILES()')
    
    smiles_col = 'SMILES'
    
    df_with_smiles = pd.DataFrame({smiles_col:[simple_smiles,simple_smiles_salt],'ID':['Mol','Salt']})
    
    df_with_fps_too = addFPBitsColsToDfWithSMILES(df_with_smiles,smiles_col)
    
    dict_of_fp_bits = {'ID':['Mol','Salt']}
    for fp_bit_index in range(0,1024):
        dict_of_fp_bits[str(fp_bit_index)] = [expected_fp_bits_list_of_simple_mol[fp_bit_index]]
        dict_of_fp_bits[str(fp_bit_index)]*=2
    
    df_of_fp_bits = pd.DataFrame(dict_of_fp_bits)
    
    
    expected_df_with_fps_too = df_with_smiles.merge(df_of_fp_bits,on='ID')
    
    cols_in_final_order = ['ID',smiles_col]+[str(i) for i in range(0,1024)]
    
    df_with_fps_too = df_with_fps_too[cols_in_final_order]
    expected_df_with_fps_too = expected_df_with_fps_too[cols_in_final_order]
    
    
    assert_frame_equal(df_with_fps_too,expected_df_with_fps_too)
    
    print('RAN check_addFPBitsColsToDfWithSMILES()')


#check_addFPBitsColsToDfWithSMILES()

def check_addFPBitsColsToDfWithProblemSMILES():
    print('Running check_addFPBitsColsToDfWithProblemSMILES()')
    
    smiles_col = 'SMILES'
    
    df_with_smiles = pd.DataFrame({smiles_col:[simple_smiles,problem_mol,simple_smiles_salt],'ID':['Mol','Problem','Salt']})
    
    df_with_fps_too = addFPBitsColsToDfWithSMILES(df_with_smiles,smiles_col)
    
    dict_of_fp_bits = {'ID':['Mol','Salt']}
    for fp_bit_index in range(0,1024):
        dict_of_fp_bits[str(fp_bit_index)] = [expected_fp_bits_list_of_simple_mol[fp_bit_index]]
        dict_of_fp_bits[str(fp_bit_index)]*=2
    
    df_of_fp_bits = pd.DataFrame(dict_of_fp_bits)
    
    
    expected_df_with_fps_too = df_with_smiles.merge(df_of_fp_bits,on='ID')
    
    cols_in_final_order = ['ID',smiles_col]+[str(i) for i in range(0,1024)]
    
    df_with_fps_too = df_with_fps_too[cols_in_final_order]
    expected_df_with_fps_too = expected_df_with_fps_too[cols_in_final_order]
    
    
    assert_frame_equal(df_with_fps_too,expected_df_with_fps_too)
    
    print('RAN check_addFPBitsColsToDfWithProblemSMILES()')


#check_addFPBitsColsToDfWithProblemSMILES()

def getFpsDfAndDataDfSkippingFailedMols(data_df,smiles_col,id_col,fp_col='fp',bond_radius=2,bitInfo={},nBits=1024,pre_calc_stand_mol_col=None,type_of_fp='Morgan'):
    if pre_calc_stand_mol_col is None:
        pre_calc_stand_mol = None
        
        fps_list = [getFPFromStandardisedMol(smiles_string,bond_radius=bond_radius,bitInfo=bitInfo,nBits=nBits,pre_calc_stand_mol=pre_calc_stand_mol,type_of_fp=type_of_fp) for smiles_string in data_df[smiles_col].tolist()]
    else:
        smiles_string = None
        
        fps_list = [getFPFromStandardisedMol(smiles_string,bond_radius=bond_radius,bitInfo=bitInfo,nBits=nBits,pre_calc_stand_mol=pre_calc_stand_mol,type_of_fp=type_of_fp) for pre_calc_stand_mol in data_df[pre_calc_stand_mol_col].tolist()]
    
    fps_df = pd.DataFrame(fps_list,columns=[fp_col])
    
    data_ids = data_df[id_col].tolist()
    
    #--------------------
    assert len(data_ids) == len(set(data_ids)),"Duplicate IDs: {}".format(findDups(data_ids))
    assert len(data_ids) == len(fps_list)
    assert len(data_ids) == data_df.shape[0]
    assert len(data_ids) == fps_df.shape[0]
    #-------------------
    
    indices_of_mols_with_failed_fp_calc = [i for i in range(0,len(data_ids)) if pd.isna(fps_list[i])]# is None]
    
    print('indices_of_mols_with_failed_fp_calc = {}'.format(indices_of_mols_with_failed_fp_calc))
    
    ids_of_mols_with_failed_fp_calc = [data_ids[i] for i in indices_of_mols_with_failed_fp_calc]
    
    #--------------------------------
    assert len(ids_of_mols_with_failed_fp_calc) == len(indices_of_mols_with_failed_fp_calc)
    #--------------------------------
    
    fps_df = fps_df[~fps_df[fp_col].isna()]
    
    data_df = data_df[~data_df[id_col].isin(ids_of_mols_with_failed_fp_calc)]
    
    #-------------------------
    fps_df.reset_index(drop=True,inplace=True)
    data_df.reset_index(drop=True,inplace=True)
    #-------------------------
    
    #---------------------------------
    assert data_df.shape[0] == fps_df.shape[0],"data_df.shape[0] = {} vs. fps_df.shape[0] = {}".format(data_df.shape[0],fps_df.shape[0])
    #---------------------------------
    
    return data_df,fps_df

def check_fps_df(fps_df):
    #-------------------------
    assert isinstance(fps_df,pd.DataFrame),type(fps_df)
    assert 1 == len(fps_df.columns.values.tolist()),len(fps_df.columns.values.tolist()) #This is required for selection of fp from row (for index,row in fps_df.iterrows()) via index 0 to work in dataset_distributions.py and SingleTcDist1NNAD.py!
    assert list(range(fps_df.shape[0]))==fps_df.index.tolist(),f'fps_df.index.tolist()={fps_df.index.tolist()}' #This is required if we want to get the corresponding ID, from a list of corresponding dataset IDs in the same order as rows in the dataframe, using the index returned by index,row in fps_df.iterrows(),e.g. in SingleTcDist1NNAD.py!
    #--------------------------

def addInChIToRow(row,smiles_col,inchi_col):
    row[inchi_col] = getInChIFromMol(row[smiles_col])
    
    return row

def addUniqueMolIds(df,smiles_col,unique_mol_id_col):
    
    if 'InChI' == unique_mol_id_col:
        inchi_col = unique_mol_id_col
        
        df_with_unique_mol_ids = df.apply(addInChIToRow,axis=1,args=(smiles_col,inchi_col))
        
        df_with_unique_mol_ids = df_with_unique_mol_ids[~df_with_unique_mol_ids[inchi_col].isna()]
    else:
        raise Exception('Only InChI unique molecule IDs supported! You supplied unique_mol_id_col={}'.format(unique_mol_id_col))
    
    return df_with_unique_mol_ids

def matchMolIdsToActivityVals(df_with_unique_mol_ids,unique_mol_id_col,activity_col):
    
    id_to_act_vals_dict = defaultdict(list)
    
    record_dicts = df_with_unique_mol_ids.to_dict('records')
    
    for record in record_dicts:
        id_to_act_vals_dict[record[unique_mol_id_col]].append(record[activity_col])
    
    return id_to_act_vals_dict

def matchIdToOneActValIfPossible(id_to_act_vals_dict,type_of_activities,regression_averaging,remove_outliers_prior_to_averaging,n_sd_extreme,bioactVal_deviation_thresh):
    
    retained_id_to_single_act_val_dict = {}
    
    if 'classification' == type_of_activities:
        for mol_id in id_to_act_vals_dict.keys():
            
            unique_act_vals = list(set(id_to_act_vals_dict[mol_id]))
            
            #=================================
            if len(unique_act_vals) > 1:
                continue
            #==================================
            
            retained_id_to_single_act_val_dict[mol_id] = unique_act_vals[0]
    elif 'regression' == type_of_activities:
        if remove_outliers_prior_to_averaging:
            
            for mol_id in id_to_act_vals_dict.keys():
                
                dict_of_bio_data_points_for_one_mol = dict(zip(range(0,len(id_to_act_vals_dict[mol_id])),id_to_act_vals_dict[mol_id]))
                
                
                bioactVal_average,bioactVal_deviation = getAveragePlusDeviationEstimateForBioactAfterRemovingOutliers(dict_of_bio_data_points_for_one_mol=dict_of_bio_data_points_for_one_mol,n_sd_extreme=n_sd_extreme,type_of_averaging=regression_averaging)
                
                #======================
                if not bioactVal_deviation_thresh is None:
                    if bioactVal_deviation > bioactVal_deviation_thresh:
                        continue
                #======================
                #======================
                if pd.isna(bioactVal_average):
                    continue
                #======================
                
                retained_id_to_single_act_val_dict[mol_id] = bioactVal_average
        else:
            raise Exception('remove_outliers_prior_to_averaging=False is not currently supported!')
    
    else:
        raise Exception('Unrecognised type_of_activities={}'.format(type_of_activities))
    
    return retained_id_to_single_act_val_dict

def onlyKeepOneOccurenceOfRetainedIds(df_with_unique_mol_ids,unique_mol_id_col,retained_id_to_single_act_val_dict):
    #Keep only one occurence of each molecule ID if present in retained_id_to_single_act_val_dict, or drop all occurences
    
    df_filtered = df_with_unique_mol_ids[df_with_unique_mol_ids[unique_mol_id_col].isin(list(retained_id_to_single_act_val_dict.keys()))]
    
    df_filtered = df_filtered.drop_duplicates(unique_mol_id_col)
    
    return df_filtered

def updateRowActivity(row,unique_mol_id_col,activity_col,retained_id_to_single_act_val_dict):
    row[activity_col] = retained_id_to_single_act_val_dict[row[unique_mol_id_col]]
    
    return row

def updateActivities(df_filtered,unique_mol_id_col,activity_col,retained_id_to_single_act_val_dict):
    
    df_final_precursor = df_filtered.apply(updateRowActivity,axis=1,args=(unique_mol_id_col,activity_col,retained_id_to_single_act_val_dict))
    
    return df_final_precursor

def dropAllIntermediateCols(df_final_precursor,needed_to_assign_unique_mol_id_col,unique_mol_id_col):
    
    cols_to_drop = []
    
    if needed_to_assign_unique_mol_id_col:
        cols_to_drop.append(unique_mol_id_col)
    
    df_final = df_final_precursor.drop(cols_to_drop,axis=1)
    
    return df_final


def removeDuplicateMolsTakingAccountOfActivities(df,smiles_col,activity_col,unique_mol_id_col=None,type_of_activities="classification",regression_averaging="ArithmeticMean",remove_outliers_prior_to_averaging=True,drop_intermediate_cols=True,n_sd_extreme=3,bioactVal_deviation_thresh=None):
    #####################################
    #If we have some identical molecules, either based upon computing InChIs from SMILES OR using a pre-existing unique molecule ID (e.g. corporate database identifier), it is not sufficient to just drop all but one of the duplicates. This is because we may be arbitarily discarding all but one of the measurements of the endpoint ( = activity) for that compound. 
    #For a given set of duplicates, we should proceed as follows:
    ###For classification (categorical) activities, we should drop all occurences if the class labels differ. Otherwise, we should drop all but one occurence and assign the single class label for all duplicates.
    ###For regression (continuous, numerical) activities, we should produce some kind of average, possibly after removing outliers which have been automatically identified, based upon being n_sd_extreme number of standard deviations away from the mean of the remaining values - if there are more than two values. (We may also want to exclude any molecules where the standard deviation or other measure of variation between records which we average if greater than bioactVal_deviation_thresh.) We should subsequently assign this average and drop all but one occurence.
    ######################################
    
    original_number_of_records = df.shape[0]
    
    print('Prior to removing duplicates, there are {} records'.format(original_number_of_records))
    
    if unique_mol_id_col is None:
        unique_mol_id_col = 'InChI'
        
        needed_to_assign_unique_mol_id_col = True
        
        df_with_unique_mol_ids = addUniqueMolIds(df,smiles_col,unique_mol_id_col)
    else:
        needed_to_assign_unique_mol_id_col = False
        
        df_with_unique_mol_ids = df
    
    id_to_act_vals_dict = matchMolIdsToActivityVals(df_with_unique_mol_ids,unique_mol_id_col,activity_col)
    
    retained_id_to_single_act_val_dict = matchIdToOneActValIfPossible(id_to_act_vals_dict,type_of_activities,regression_averaging,remove_outliers_prior_to_averaging,n_sd_extreme,bioactVal_deviation_thresh)
    
    df_filtered = onlyKeepOneOccurenceOfRetainedIds(df_with_unique_mol_ids,unique_mol_id_col,retained_id_to_single_act_val_dict)
    
    df_final_precursor = updateActivities(df_filtered,unique_mol_id_col,activity_col,retained_id_to_single_act_val_dict)
    
    if drop_intermediate_cols:
    
        df_final = dropAllIntermediateCols(df_final_precursor,needed_to_assign_unique_mol_id_col,unique_mol_id_col)
    else:
        df_final = df_final_precursor
    
    df_final.reset_index(drop=True, inplace=True)
    
    final_number_of_records = df_final.shape[0]
    
    print('After removing duplicates, there are {} records'.format(final_number_of_records))
    
    assert final_number_of_records <= original_number_of_records
    
    if final_number_of_records < original_number_of_records:
        print('{} records were identified as duplicates and dropped'.format((original_number_of_records-final_number_of_records)))
    
    return df_final


def check_removeDuplicateMolsTakingAccountOfActivities():
    
    examples_dict = defaultdict(dict)
    
    smiles_col = 'SMILES'
    
    activity_col = 'Activity'
    
    regression_averaging="ArithmeticMean"
    
    remove_outliers_prior_to_averaging=True
    
    unique_mol_id_col = None
    
    #------------------------
    examples_dict[1]['df'] = pd.DataFrame({'ID':[1,2,3,4,5,6],smiles_col:['CCO','[CH3]CO','[OH2]','O','CCN','NCC'],activity_col:[1,0,0,0,1,1]})
    
    examples_dict[1]['type_of_activities'] = 'classification'
    
    examples_dict[1]['expected_df_final'] = pd.DataFrame({'ID':[3,5],smiles_col:['[OH2]','CCN'],activity_col:[0,1]})
    #-------------------------
    examples_dict[2]['df'] = pd.DataFrame({'ID':[1,2,3,4,5,6],smiles_col:['CCO','[CH3]CO','[OH2]','O','CCN','NCC'],activity_col:[1.0,0.0,0.9,1.1,2,1.0]})
    
    examples_dict[2]['type_of_activities'] = 'regression'
    
    examples_dict[2]['expected_df_final'] = pd.DataFrame({'ID':[1,3,5],smiles_col:['CCO','[OH2]','CCN'],activity_col:[0.5,1.0,1.5]})
    #----------------------------
    examples_dict[3]['df'] = pd.DataFrame({'ID':[0,1,2,3,4,5,6],smiles_col:['?','CCO','[CH3]CO','[OH2]','O','CCN','NCC'],activity_col:[0.78,1.0,0.0,0.9,1.1,2,1.0]})
    
    examples_dict[3]['type_of_activities'] = 'regression'
    
    examples_dict[3]['expected_df_final'] = pd.DataFrame({'ID':[1,3,5],smiles_col:['CCO','[OH2]','CCN'],activity_col:[0.5,1.0,1.5]})
    #----------------------------
    examples_dict[4]['df'] = pd.DataFrame({'ID':[0,1,2,3,4,5,6,7,8],smiles_col:['?','CCO','[CH3]CO','[OH2]','O','CCN','NCC','CCN','NCC'],activity_col:[0.78,1.0,0.0,0.9,1.1,2,1.0,20.0,1.6]})
    
    examples_dict[4]['type_of_activities'] = 'regression'
    
    examples_dict[4]['expected_df_final'] = pd.DataFrame({'ID':[1,3,5],smiles_col:['CCO','[OH2]','CCN'],activity_col:[0.5,1.0,1.8]})
    #----------------------------
    
    print('Running check_removeDuplicateMolsTakingAccountOfActivities(...)')
    
    for eg in examples_dict.keys():
        print('-'*20)
        print('Checking example {}'.format(eg))
        print('-'*20)
        
        df=examples_dict[eg]['df']
        
        type_of_activities=examples_dict[eg]['type_of_activities']
        
        expected_df_final=examples_dict[eg]['expected_df_final']
        
        
        df_final = removeDuplicateMolsTakingAccountOfActivities(df=df,smiles_col=smiles_col,activity_col=activity_col,unique_mol_id_col=unique_mol_id_col,type_of_activities=type_of_activities,regression_averaging=regression_averaging,remove_outliers_prior_to_averaging=remove_outliers_prior_to_averaging)
        
        #############
        #Debug:
        print(f'df_final=')
        print(df_final)
        ############
        
        assert_frame_equal(df_final,expected_df_final)
        
        
        print('-'*20)
        print('CHECKED example {}'.format(eg))
        print('-'*20)
    
    
    print('RAN check_removeDuplicateMolsTakingAccountOfActivities()')

def convert_sdf_to_DataFrame_with_SMILES(sdf_name,mol_id='ID',smiles_col_name='RDKit_SMILES',drop_rdkit_mol=True):
    if drop_rdkit_mol:
        mol_name = None
    else:
        mol_name = 'RDKit_Mol'
    
    df = PandasTools.LoadSDF(sdf_name,idName=mol_id,molColName=mol_name,smilesName=smiles_col_name)
    
    
    return df

def get_all_mol_components_as_molecules(mol):
    return [c for c in Chem.GetMolFrags(mol,asMols=True)]
    

def get_all_non_salt_mol_component_as_molecules(mol):
    mol = remove_salt_counterions(mol)
    
    return get_all_mol_components_as_molecules(mol)

def at_least_one_component_is_non_organic(components_to_evaluate):
    non_organic_components = [c for c in components_to_evaluate if unsalt.is_nonorganic(c)]
    
    return (len(non_organic_components)>=1)

def remove_salt_counterions(mol,do_not_discard_all_components=True):
    ############################################################################
    #https://www.rdkit.org/docs/source/rdkit.Chem.SaltRemover.html
    #This seems more robust than standardiser module unsalt.is_salt(component)
    ############################################################################
    remover = SaltRemover()
    
    remaining_mol = remover.StripMol(mol,dontRemoveEverything=do_not_discard_all_components)
    
    if remaining_mol is None:
        print(f'WARNING: molecule with title = {mol.GetProp("_Name")}, smiles = {Chem.MolToSmiles(mol)} comprised only salt counterions and these have all been discarded!')
    
    return remaining_mol

def this_is_a_non_organic_compound(mol,ignore_salt_counterions=True,verbose=True):
    ############################
    #This is intended for application to molecules prior to standardization.
    #All compounds, after removing salt atoms, which contain non-organic atoms, should be flagged.
    ############################
    
    mol = rdMolStandardize.Cleanup(mol) #See comments on purpose of this function call in standardize_mols.py: def standardize(...):
    
    if ignore_salt_counterions:
        components_to_evaluate = get_all_non_salt_mol_component_as_molecules(mol)
    else:
        components_to_evaluate = get_all_mol_components_as_molecules(mol)
    
    return at_least_one_component_is_non_organic(components_to_evaluate)
