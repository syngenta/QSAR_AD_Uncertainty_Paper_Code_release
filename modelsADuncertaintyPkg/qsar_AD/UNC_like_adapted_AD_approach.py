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
#######################
#This code is designed to implement an applicability domain approach which builds upon the UNC and ULP approaches previously reported in the literature: https://doi.org/10.1021/ci800151m
#In brief, , for each training set compound, the distance to the k-nearest neighbours was computed and the average distance determined. The distribution of these training set compound specific values was summarized in terms of the mean (d_average_train_mean) and standard deviation calculated (d_average_train_std) across all training set compounds.
#For a new compound, the average distance to its k-nearest neighbours in the training set (d_average_new) is also computed, along with a z-score based upon the training set distribution statistics (equation 1). Assuming a normal distribution, the corresponding p-value was computed, i.e. the probability, given the null-hypothesis that the average distance to its nearest neighbours came from the same distribution as the training set. If this p-value was less than or equal to 5%, the compound was deemed outside the domain.
#Equation 1: z_score = (d_average_new-d_average_train_mean)/(d_average_train_std)
#####################
import sys,re,os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import scipy.stats as stats
from collections import defaultdict
#===================================
#top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from .dk_NN_thresholds import dk_NN_thresholds as BaseClass_kNN_AD_Approaches
from ..utils.basic_utils import findDups,convertDefaultDictDictIntoDataFrame
from ..CheminformaticsUtils import chem_data_parsing_utils as chem_utils
#===================================

class UNC_like_AD_approach(BaseClass_kNN_AD_Approaches):
    
    def getTrainingSetStats(self):
        self.getTrainingSetAveragekNNdistances()
        
        average_distance_to_k_NN_for_all_train_set_cmpds = self.d_k_av_i_array
        
        ##############################
        #Debug:
        #print('self.d_k_av_i_array={}'.format(self.d_k_av_i_array))
        ##############################
        
        self.d_average_train_mean = np.mean(average_distance_to_k_NN_for_all_train_set_cmpds)
        
        self.d_average_train_std = np.std(average_distance_to_k_NN_for_all_train_set_cmpds)
    
    def getNewCmpdsDistancesToTrainKNN(self,test_compounds_X_df,id_col):
        return self.getNewCmpdsDistancesToTrainNN(test_compounds_X_df,id_col,no_neighbours_to_consider=self.k)

    def _getNewCmpdsAverageDistances(self,new_cmpds_ids_x,ids_col):
        '''
        Inputs:
        
        new_cmpds_ids_x : [no. new cmpds, no. descriptors + 1] dataframe of descriptor values, along with an IDs column
        
        Computes:
        
        Average distance between new compound and k nearest training set neighbours
        '''
        #=====================
        if self.knn_model is None: raise Exception('You need to build the k-NN model by calling getTrainingSetStats() first, where the k-NN options (value of k, distance metric etc.) were specified when intializing the instance of this class!')
        #=====================
        
        new_cmpds_neigh_dist_array,new_cmpds_neigh_ind_array,new_ids = self.getNewCmpdsDistancesToTrainKNN(test_compounds_X_df=new_cmpds_ids_x,id_col=ids_col)
        
        new_cmpds_average_distances_array = np.mean(new_cmpds_neigh_dist_array,axis=1)
        
        
        #------------------------
        if not len(new_ids) == new_cmpds_average_distances_array.shape[0]: raise Exception('len(new_ids) =%d, new_cmpds_average_distances_array.shape[0] = %d' % (len(new_ids),new_cmpds_average_distances_array.shape[0]))
        if not 1 == len(list(new_cmpds_average_distances_array.shape)): raise Exception(len(list(new_cmpds_average_distances_array.shape)))
        #-------------------------
        
        return dict(zip(new_ids,new_cmpds_average_distances_array.tolist()))
    
    def insideAD(self,d_average_new,p_value_thresh,avoid_zero_division_offset=10**-10):
        
        try:
            z_score_numerator = (d_average_new-self.d_average_train_mean)
            
            z_score_denominator = self.d_average_train_std
            
            if 0 == z_score_denominator:
                z_score_denominator += avoid_zero_division_offset
            
            z_score = z_score_numerator/z_score_denominator
            
            ########################
            #Debug:
            #print("d_average_new={}".format(d_average_new))
            #print("self.d_average_train_mean={}".format(self.d_average_train_mean))
            #print("self.d_average_train_std={}".format(self.d_average_train_std))
            #print("z_score={}".format(z_score))
            ########################
            
        except AttributeError as err:
            print('You need to call getTrainingSetStats() first!')
            raise Exception(err)
        
        p_value = 1 - stats.norm.cdf(z_score)
        
        return (p_value > p_value_thresh),p_value
    
    def getADstatusOfTestSetCompounds(self,test_compounds_X_df,id_col,p_value_thresh=0.05,debug=False):
        
        if debug:
            print('test_compounds_X_df[id_col].tolist()={}'.format(test_compounds_X_df[id_col].tolist()))
        
        new_id_to_AverageDistance_dict = self._getNewCmpdsAverageDistances(new_cmpds_ids_x=test_compounds_X_df,ids_col=id_col)
        
        if debug:
            print('list(new_id_to_AverageDistance_dict.keys()) = {}'.format(list(new_id_to_AverageDistance_dict.keys())))
        
        test_id_ad_status_dict = defaultdict(dict)
        
        for new_id in new_id_to_AverageDistance_dict.keys():
            
            inside_AD,p_value = self.insideAD(d_average_new=new_id_to_AverageDistance_dict[new_id],p_value_thresh=p_value_thresh)
            
            test_id_ad_status_dict[new_id]['InsideAD'] = inside_AD
            test_id_ad_status_dict[new_id]['P-value'] = p_value
        
        return test_id_ad_status_dict



def apply_UNC_like_AD_approach_with_ECFP_FPs(train_df_with_SMILES,test_df_with_SMILES,smiles_col,id_col,endpoint_col,k=5,distance_metric='jaccard',scale=False,p_value_thresh=0.05):
    #=============================
    #train_df_with_SMILES or test_df_with_SMILES must include a column of IDs and a column of SMILES. 
    #an OPTIONAL column of endpoint values may also be included, but this will be removed by this function, rather than by UNC_like_AD_approach()
    #===========================
    
    fps_df_dict = {}
    
    for smiles_df_name in ['train_df_with_SMILES','test_df_with_SMILES']:
        
        if 'train_df_with_SMILES' == smiles_df_name:
            smiles_df = train_df_with_SMILES
        elif 'test_df_with_SMILES' == smiles_df_name:
            smiles_df = test_df_with_SMILES
        else:
            raise Exception('Unrecognised smiles_df_name=%s' % smiles_df_name)
        
        df_with_fps_too = chem_utils.addFPBitsColsToDfWithSMILES(df_with_smiles=smiles_df,smiles_col=smiles_col,report_problem_smiles=True)
        
        fps_df_dict[smiles_df_name] = df_with_fps_too.drop(labels=[smiles_col],axis=1)
    
    train_df_ready_for_AD_code = fps_df_dict['train_df_with_SMILES']
    
    ###############
    #Debug:
    #print('train_df_ready_for_AD_code={}'.format(train_df_ready_for_AD_code))
    ###############
    
    test_df_ready_for_AD_code = fps_df_dict['test_df_with_SMILES'].drop(labels=[endpoint_col],axis=1)
    
    
    UNC_like_AD_Estimator = UNC_like_AD_approach(train_df=train_df_ready_for_AD_code,id_col=id_col,k=k,distance_metric=distance_metric,scale=False,endpoint_col=endpoint_col)
    
    UNC_like_AD_Estimator.getTrainingSetStats()
    
    test_id_ad_status_dict = UNC_like_AD_Estimator.getADstatusOfTestSetCompounds(test_compounds_X_df=test_df_ready_for_AD_code,id_col=id_col,p_value_thresh=p_value_thresh)
    
    return test_id_ad_status_dict

def check_UNC_like_AD_approach():
    
    #=======================
    #Values used for all examples:
    id_col = 'ID'
    endpoint_col = 'Endpoint'
    smiles_col = 'SMILES'
    #========================
    
    examples = defaultdict(dict)
    #=============================
    examples[1]['train_df_with_SMILES'] = pd.DataFrame({id_col:[1,2,3,4,5,6],endpoint_col:['Active','Inactive','Active','Active','?','?'],smiles_col:['CO','CCO','CCCO','CCCCO','C(C)O','C(C)(C)O']})
    examples[1]['test_df_with_SMILES'] = pd.DataFrame({id_col:[10,20,30],endpoint_col:['?','?','Active'],smiles_col:['CO','CCCCCO','c1ccccc1']})
    examples[1]['k'] = 5
    #--------
    examples[1]['expected_test_id_ad_status_dict'] = defaultdict(dict)
    #-------
    examples[1]['expected_test_id_ad_status_dict'][10]['InsideAD'] = True #Iniutively expect and hope for this if the compound is inside the training set!
    examples[1]['expected_test_id_ad_status_dict'][10]['P-value'] = 0.6738043275661227 #1.0 #Other than being high, I think this exact value would depend upon the diversity of the training set!
    #--------
    examples[1]['expected_test_id_ad_status_dict'][20]['InsideAD'] = True
    examples[1]['expected_test_id_ad_status_dict'][20]['P-value'] = 0.7636642420443979 #0.6 #Other than being high and lower than a value for a compound present in the training set (but might that depend upon how representative that one compound was?), it is difficult to know if this exact value is correct.
    #--------
    examples[1]['expected_test_id_ad_status_dict'][30]['InsideAD'] = False
    examples[1]['expected_test_id_ad_status_dict'][30]['P-value'] = 0.0006275199000248532#0.01 #Other than being low, it is difficult to know if this exact value is correct.
    #==============================
    #=============================
    examples[2]['train_df_with_SMILES'] = pd.DataFrame({id_col:[1,2,3,4,5,6],endpoint_col:['Active','Inactive','Active','Active','?','?'],smiles_col:['CO','CCO','CCCO','CCCCO','C(C)O','C(C)(C)O']})
    examples[2]['test_df_with_SMILES'] = pd.DataFrame({id_col:[10,20,30],endpoint_col:['?','?','Active'],smiles_col:['CO','CCCCCO','c1ccccc1']})
    examples[2]['k'] = 1
    #--------
    examples[2]['expected_test_id_ad_status_dict'] = defaultdict(dict)
    #-------
    examples[2]['expected_test_id_ad_status_dict'][10]['InsideAD'] = True #Iniutively expect and hope for this if the compound is inside the training set!
    examples[2]['expected_test_id_ad_status_dict'][10]['P-value'] = 0.900925383424712  #1.0 #Why would this value not be 1.0 for an exact training set match and k=1? #Because, being in the training set and k=1 simply means that d_average_new =0 and p-value is based upon the z-score (d_average_new-self.d_average_train_mean)/(self.d_average_train_std)
    #--------
    examples[2]['expected_test_id_ad_status_dict'][20]['InsideAD'] = True
    examples[2]['expected_test_id_ad_status_dict'][20]['P-value'] = 0.691148097595966 #0.6 #Other than being high and lower than a value for a compound present in the training set (but might that depend upon how representative that one compound was?), it is difficult to know if this exact value is correct.
    #--------
    examples[2]['expected_test_id_ad_status_dict'][30]['InsideAD'] = False
    examples[2]['expected_test_id_ad_status_dict'][30]['P-value'] = 0.01672349196932843#0.01 #Other than being low, it is difficult to know if this exact value is correct.
    #==============================
    #=============================
    examples[3]['train_df_with_SMILES'] = pd.DataFrame({id_col:[1,2,3,4,5,6],endpoint_col:['Active','Inactive','Active','Active','?','?'],smiles_col:['CO','CO','CCCO','CCCO','C(C)O','C(C)O']}) #KEY CHANGE FROM EXAMPLE 2: EACH OF THE TRAINING SET COMPOUND SMILES OCCURS TWICE!
    examples[3]['test_df_with_SMILES'] = pd.DataFrame({id_col:[10,20,30],endpoint_col:['?','?','Active'],smiles_col:['CO','CCCCCO','c1ccccc1']})
    examples[3]['k'] = 1
    #--------
    examples[3]['expected_test_id_ad_status_dict'] = defaultdict(dict)
    #-------
    examples[3]['expected_test_id_ad_status_dict'][10]['InsideAD'] = True #Iniutively expect and hope for this if the compound is inside the training set!
    examples[3]['expected_test_id_ad_status_dict'][10]['P-value'] = 0.5 #If k=1 and this compound is also in the training set and all training set compounds appear twice, this must surelly be 0.5 now!
    #--------
    examples[3]['expected_test_id_ad_status_dict'][20]['InsideAD'] = False #With k=1 and all compounds in the training set appearing twice, the z_score denominator would approach zero, meaning the p-value should tend to zero for any compound which is not an exact match for a training set compound!
    examples[3]['expected_test_id_ad_status_dict'][20]['P-value'] = 0.0
    #--------
    examples[3]['expected_test_id_ad_status_dict'][30]['InsideAD'] = False #With k=1 and all compounds in the training set appearing twice, the z_score denominator would approach zero, meaning the p-value should tend to zero for any compound which is not an exact match for a training set compound!
    examples[3]['expected_test_id_ad_status_dict'][30]['P-value'] = 0.0
    #==============================
    
    for eg in examples.keys():
        print('Running check_UNC_like_AD_approach() for example %d' % eg)
        
        test_id_ad_status_dict = apply_UNC_like_AD_approach_with_ECFP_FPs(train_df_with_SMILES=examples[eg]['train_df_with_SMILES'],test_df_with_SMILES=examples[eg]['test_df_with_SMILES'],smiles_col=smiles_col,id_col=id_col,endpoint_col=endpoint_col,k=examples[eg]['k'])
        
        df = convertDefaultDictDictIntoDataFrame(test_id_ad_status_dict,col_name_for_first_key=id_col)
        
        #Debug:
        #print('test_id_ad_status_dict=')
        #print(test_id_ad_status_dict)
        
        expected_df = convertDefaultDictDictIntoDataFrame(examples[eg]['expected_test_id_ad_status_dict'],col_name_for_first_key=id_col)
        
        assert_frame_equal(df,expected_df)
        
        print('RAN check_UNC_like_AD_approach() for example %d' % eg)

#check_UNC_like_AD_approach()
