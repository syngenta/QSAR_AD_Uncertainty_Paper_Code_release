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
#Contact richard.marchese_robinson [at] syngenta.com
#######################
#####################################################
#This code is designed to derive the underlying k-NN model from the training set required to compute the dk-NN thresholds then derive the dk-NN thresholds as part of the RDN AD approach described in this paper:
#Aniceto et al. J Cheminform (2016) 8:69
#DOI 10.1186/s13321-016-0182-y
#N.B. The feature selection steps should have been performed prior to the calculations implemented herein.
#The dk-NN algorithm was introduced in the following paper:
#Sahigara et al. Journal of Cheminformatics 2013, 5:27
#http://www.jcheminf.com/content/5/1/27
#####################################################
import sys,re,os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors,DistanceMetric
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
#https://pubs.acs.org/doi/full/10.1021/ci9800211
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import DataConversionWarning


#===================================
# dir_of_file = os.path.dirname(os.path.abspath(__file__))
# top_dir = os.path.dirname(dir_of_file)
from ..utils.basic_utils import findDups
#===================================

def scaleDescs(test_X,train_X=None,fitted_scaler=None):
    #--------------------------------
    if not type(pd.DataFrame({'ID':[1]})) == type(test_X): raise Exception(type(test_X))
    if not train_X is None:
        if not train_X.shape[0] == test_X.shape[0]: raise Exception('train_X.shape[0] =%d, test_X.shape[0]=%d' % (train_X.shape[0], test_X.shape[0]))
        if not train_X.columns.values.tolist() == test_X.columns.values.tolist(): raise Exception('train_X.columns.values.tolist() =%s, test_X.columns.values.tolist()%d' % (str(train_X.columns.values.tolist()), str(test_X.columns.values.tolist())))
        if not fitted_scaler is None: raise Exception(fitted_scaler)
    #--------------------------------
    
    if fitted_scaler is None:
        
        scaler = MinMaxScaler()
        
        fitted_scaler = scaler.fit(train_X)
    
    trans_test_X = pd.DataFrame(fitted_scaler.transform(test_X),columns=test_X.columns.values.tolist())
    
    return trans_test_X,fitted_scaler

def Qn(np_array,n):
    #-----------------------
    if not type(np.array([[1.0]])) == type(np_array): raise Exception(type(np_array))
    if not 1 == len(list(np_array.shape)): raise Exception(len(list(np_array.shape)))
    #------------------------
    
    if 1 == n:
        return np.percentile(np_array,q=25)
    elif 2 == n:
        return np.percentile(np_array,q=50)
    elif 3 == n:
        return np.percentile(np_array,q=75)
    else:
        raise Exception('It is not possible to compute the %d quartile!' % n)

def check_Qn():
    #Based on Excel QUARTILE.INC(...) function:
    l = [0.5,1,98.5,0.5]
    a = np.array(l)
    #------------------------
    print('Checking Qn')
    if not 0.5==Qn(a,1): raise Exception(Qn(a,1))
    if not 0.75 == Qn(a,2): raise Exception(Qn(a,2))
    if not 25.375 == Qn(a,3): raise Exception(Qn(a,3))
    print('CHECKED Qn')
    #-----------------------

#check_Qn()

def Q3(np_array):
    return Qn(np_array,3)

def IQR(np_array):
    iqr = Q3(np_array) - Qn(np_array,1)
    #----------------------
    if not iqr >=0: raise Exception(iqr)
    #----------------------
    return iqr

class dk_NN_thresholds():
    def __init__(self, train_df, id_col, k, distance_metric='jaccard', scale=False, endpoint_col=None,expected_descs_no=None,ignore_warnings=True,only_consider_k_neighbours_for_final_threshold_calculations=False,debug=False):
        '''
        train_df: data frame containing ONLY id_col (numeric IDs) and maybe endpoint_col and columns with descriptors to be used to compute the necessary distances.
        id_col: numeric IDs
        
        distance_metric="jaccard" (= 1-Tanimoto similarity) is recommended for fingerprints [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html] -> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html#sklearn.metrics.DistanceMetric]
        '''
        #-------------------------
        if not isinstance(train_df,pd.DataFrame): raise Exception(f'type(train_df)={type(train_df)}')
        if not isinstance(id_col,str): raise Exception(f'type(id_col)={type(id_col)}')
        if not isinstance(k,int): raise Exception(f'type(k)={type(k)}')
        #-------------------------
        
        if endpoint_col is None:
            cols_to_drop = [id_col]
        else:
            cols_to_drop = [id_col,endpoint_col]
        
        self.train_df = train_df.copy(deep=True)
        del train_df

        self.train_df.reset_index(drop=True,inplace=True)
        
        self.train_X = self.train_df.drop(labels=cols_to_drop,axis=1)
        
        #--------------------
        if not expected_descs_no is None:
            self.expected_descs_no = expected_descs_no
        
            if not self.train_X.shape[1] == self.expected_descs_no: raise Exception("self.train_X.shape[1] ={}, self.expected_descs_no={}".format(self.train_X.shape[1],self.expected_descs_no))
        else:
            self.expected_descs_no = self.train_X.shape[1]
        #---------------------
        
        #-------------------------
        if not self.train_X.shape[0] == self.train_df.shape[0]: raise Exception(r'self.train_X.shape[0] =%d, self.train_df.shape[0]=%d' % (self.train_X.shape[0],self.train_df.shape[0]))
        
        if not self.train_X.shape[1] == (self.train_df.shape[1]-len(cols_to_drop)): raise Exception(r'self.train_X.shape[1] =%d, self.train_df.shape[1]=%d' % (self.train_X.shape[1],self.train_df.shape[1]))
        #-------------------------
        
        #-------------------------
        #Allow for the possibility that each training set row is duplicated, e.g. identical molecules or molecules with the same fingerprints:
        if all(self.train_X.duplicated(keep=False).tolist()):
            self.all_train_X_are_duplicates = True
        else:
            self.all_train_X_are_duplicates = False
        #-------------------------
        
        self.train_ids = self.train_df[id_col]
        
        #-------------------------
        if not self.train_ids.shape[0] == self.train_X.shape[0]: raise Exception('self.train_ids.shape[0] =%d, self.train_X.shape[0]=%d' % (self.train_ids.shape[0],self.train_X.shape[0]))
        
        ids = self.train_ids.tolist()
        
        if not self.train_ids.shape[0] == len(ids): raise Exception('self.train_ids.shape[0] =%d, len(ids)=%d' % (self.train_ids.shape[0],len(ids)))
        
        if not len(ids) == len(set(ids)): raise Exception('ids must be unique values! - \n %s' % str(ids))
        
        if not all([isinstance(id_,int) for id_ in ids]): raise Exception('ids types:%s' % str([type(ID) for ID in ids]))
        
        #-------------------------
        
        self.train_index_to_id_dict = dict(zip(self.train_ids.index.tolist(),self.train_ids.tolist()))

        #-----------------------
        if not len(self.train_index_to_id_dict) == self.train_df.shape[0]: raise Exception(f'How can there be duplicates in self.train_df as any duplicates in the original indices should have been fixed above? - {findDups(self.train_df.index.tolist())}')
        #-----------------------

        self.k = k

        self.knn_model = None
        
        self.distance_metric = distance_metric
        
        self.scale = scale
        
        if self.scale:
            trans_test_X,fitted_scaler = scaleDescs(test_X=self.train_X,train_X=self.train_X,fitted_scaler=None)
            
            self.fitted_scaler = fitted_scaler
            
            self.train_X = trans_test_X
            
            #------------------------------
            if not self.train_X.shape[1] == self.expected_descs_no: raise Exception("self.train_X.shape[1] ={}, self.expected_descs_no={}".format(self.train_X.shape[1],self.expected_descs_no))
            #------------------------------
        else:
            self.fitted_scaler = None
        
        ############################
        #We initially considered computing the training set compound-specific distances based upon the mean distance to the original k-nearest neighbours that lay within a distance of RefVal from the training set compound in question
        #We speculated that this might better emphasize that new compounds that were only close to training set outliers should be much less likely to lie inside the domain
        #However, the original approach of  Sahigara et al. (2013) [http://www.jcheminf.com/content/5/1/27] is probably better at ensuring a new compound would not be considered outside the domain if it was close to a cluster of close analogues in the training set.
        #############################
        self.only_consider_k_neighbours_for_final_threshold_calculations = only_consider_k_neighbours_for_final_threshold_calculations
        #############################

        #-----------------------
        self.ignore_warnings = ignore_warnings
        if self.ignore_warnings:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore",category=PendingDeprecationWarning)
        #-----------------------

        self.debug = debug
    
    def buildNNmodel(self,no_neighbours_to_consider):
        if self.ignore_warnings:
            warnings.filterwarnings("ignore", category=DataConversionWarning) #This does not work?!
        
        #----------------------------------
        if not no_neighbours_to_consider in [self.k,(self.train_X.shape[0]-1),(self.train_X.shape[0])]:
            raise Exception(f'no_neighbours_to_consider={no_neighbours_to_consider},self.k={self.k},self.train_X.shape[0]={self.train_X.shape[0]}')
        #------------------------------------

        model = NearestNeighbors(n_neighbors=no_neighbours_to_consider,metric=self.distance_metric).fit(X=self.train_X,y=None)

        #--------------------------
        if not model.n_features_in_ == self.train_X.shape[1]: raise Exception(f'model.n_features_in_={model.n_features_in_} vs. self.train_X.shape[1]={self.train_X.shape[1]}')
        
        if no_neighbours_to_consider == self.k:
            self.knn_model = model
        
        if no_neighbours_to_consider == (self.train_X.shape[0]-1): #This may be equal to self.k for a small, trial training set!
            self.all_other_training_set_compounds_nn_model = model
        
        if no_neighbours_to_consider == (self.train_X.shape[0]): #This may be equal to self.k for a small, trial training set!
            self.all_training_set_compounds_nn_model = model #To support training set - new (test) compound distance calculations
        
        
    
    def run_checks_which_are_appropriate_for_all_pairwise_training_set_nearest_neighbour_distances(self,train_nn_dist_array):
        #-------------------------
        if not self.train_ids.shape[0] == train_nn_dist_array.shape[0]: raise Exception('self.train_ids.shape[0] =%d, train_nn_dist_array.shape[0] =%d' % (self.train_ids.shape[0],train_nn_dist_array.shape[0]))
        #The distance of the training set instances to themselves should not be computed:
        all_closest_distances = [train_nn_dist_array[i][0] for i in range(0,self.train_ids.shape[0])]
        unique_closest_distances = list(set(all_closest_distances))
        if not self.all_train_X_are_duplicates: #Stop this failing when we specifically design the training set to comprise pairs of duplicates!
            if 1 == len(unique_closest_distances) and 0 == unique_closest_distances[0]: raise Exception('The training set instances appear to be compared to themselves???')
        #-------------------------

    def run_checks_which_are_appropriate_for_kNN_pairwise_training_set_nearest_neighbour_distances(self,train_nn_dist_array):
        #-------------------------
        if not self.k == train_nn_dist_array.shape[1]: raise Exception('k=%d nearest-neighbours specified. But training set instances appear to be linked to %d nearest neighbors.' % (self.k,train_nn_dist_array.shape[1]))
        #-------------------------

    def _getTrainingSetNNdistances(self,only_consider_k_nn=False):
        
        if only_consider_k_nn:
            self.buildNNmodel(self.k)
        else:
            self.buildNNmodel((self.train_X.shape[0]-1)) #For a small, e.g. practice, training set, self.k could be equal to (self.train_X.shape[0]-1)
        
        if only_consider_k_nn:
            
            train_nn_dist_array, train_neigh_ind_array = self.knn_model.kneighbors()

            self.train_knn_dist_array = train_nn_dist_array

            self.run_checks_which_are_appropriate_for_kNN_pairwise_training_set_nearest_neighbour_distances(train_nn_dist_array)
        else:
            
            train_nn_dist_array, train_neigh_ind_array = self.all_other_training_set_compounds_nn_model.kneighbors()
            
            self.all_other_training_set_cmpds_dist_array = train_nn_dist_array
        
        
        
        self.run_checks_which_are_appropriate_for_all_pairwise_training_set_nearest_neighbour_distances(train_nn_dist_array)
        
        
        
        
    
    def getTrainingSetAveragekNNdistances(self):
        self._getTrainingSetNNdistances(only_consider_k_nn=True)
        
        d_k_av_i_array = np.mean(self.train_knn_dist_array,axis=1)
        
        
        #------------------------
        if not self.train_ids.shape[0] == d_k_av_i_array.shape[0]: raise Exception('self.train_ids.shape[0] =%d, d_k_av_i_array.shape[0] = %d' % (self.train_ids.shape[0],d_k_av_i_array.shape[0]))
        if not 1 == len(list(d_k_av_i_array.shape)): raise Exception(len(list(d_k_av_i_array.shape)))
        #-------------------------
        
        self.d_k_av_i_array = d_k_av_i_array
    
    def computeRefVal(self):
        self.getTrainingSetAveragekNNdistances()
        
        RefVal = Q3(self.d_k_av_i_array) + 1.5*IQR(self.d_k_av_i_array)
        
        
        self.RefVal = RefVal
    
    def getInitialTrainingSetThresholds(self):
        self.computeRefVal()
        
        ##################
        if self.debug:
            print(f'self.train_df.shape[0]={self.train_df.shape[0]}')
            print(f'len(self.train_ids.index.tolist())={len(self.train_ids.index.tolist())}')
        ###################


        self.TrainIdsMatchedToThresholds = {}
        
        if self.only_consider_k_neighbours_for_final_threshold_calculations:
            neighbours_distance_list_of_lists = self.train_knn_dist_array.tolist()
        else:
            self._getTrainingSetNNdistances(only_consider_k_nn=False)
                
            neighbours_distance_list_of_lists = self.all_other_training_set_cmpds_dist_array.tolist()

        for i in range(self.train_df.shape[0]):
            neighbour_distances_within_RefVal = np.array([d for d in neighbours_distance_list_of_lists[i] if d <= self.RefVal])
           
            #------------------------------------
            if not 1 == len(neighbour_distances_within_RefVal.shape): raise Exception(neighbour_distances_within_RefVal.shape)
            #------------------------------------
           
            ###############################
            #When zero nearest neighbour distances were retained within RefVal, Sahigara et al. (2013) [http://www.jcheminf.com/content/5/1/27] set the threshold to the smallest threshold in the training set where at least one neighbour was retained.            
            #However, we initially set this to zero - then update this if we subsequently call updateZeroValuedTrainingSetThresholds(...)
            ##############################
            if 0 < neighbour_distances_within_RefVal.shape[0]:
                self.TrainIdsMatchedToThresholds[self.train_ids.tolist()[i]] = np.mean(neighbour_distances_within_RefVal)
            elif 0 == neighbour_distances_within_RefVal.shape[0]:
                self.TrainIdsMatchedToThresholds[self.train_ids.tolist()[i]] = 0.0 #In this scenario, I observed np.mean(neighbour_distances_within_RefVal) to actually give nan!
            else:
                raise Exception('neighbour_distances_within_RefVal.shape[0]=%s' % str(neighbour_distances_within_RefVal.shape[0]))
    
    def updateZeroValuedTrainingSetThresholds(self):
        ###############################
        #When zero nearest neighbour distances were retained within RefVal, which would indicate the threshold should be set to zero, Sahigara et al. (2013) [http://www.jcheminf.com/content/5/1/27] set the threshold to the smallest threshold in the training set where at least one neighbour was retained.
        ############################
        non_zero_thresholds = [v for v in self.TrainIdsMatchedToThresholds.values() if not 0 == v]
        
        if not 0 == len(non_zero_thresholds):
            if not len(list(self.TrainIdsMatchedToThresholds.values()))==len(non_zero_thresholds):
                min_non_zero_threshold = min(non_zero_thresholds)
                for key in self.TrainIdsMatchedToThresholds.keys():
                    if 0 == self.TrainIdsMatchedToThresholds[key]: self.TrainIdsMatchedToThresholds[key] = min_non_zero_threshold
                
                new_non_zero_thresholds = [v for v in self.TrainIdsMatchedToThresholds.values() if not 0 == v]
                assert len(list(self.TrainIdsMatchedToThresholds.values()))==len(new_non_zero_thresholds)

        else:
            print('Warning: All thresholds are zero!')
    
    def getNewCmpdsDistancesToTrainNN(self,test_compounds_X_df,id_col,no_neighbours_to_consider):
        #********************************
        if not isinstance(test_compounds_X_df,pd.DataFrame): raise Exception(type(test_compounds_X_df))
        #********************************
        
        test_ids = test_compounds_X_df[id_col].tolist()
        #-------------------------
        if not len(test_ids)==len(set(test_ids)): raise Exception('Duplicate IDs=%s' % findDups(test_ids))
        #-------------------------
        
        test_X = test_compounds_X_df.copy()
        test_X = test_X.drop(labels=[id_col],axis=1)
        
        #------------------------------
        if not test_X.shape[1] == self.expected_descs_no: raise Exception("test_X.shape[1] ={}, self.expected_descs_no={}".format(test_X.shape[1],self.expected_descs_no))
        #------------------------------
        
        if self.scale:
            trans_test_X,fitted_scaler = scaleDescs(test_X=test_X,train_X=None,fitted_scaler=self.fitted_scaler)
            test_X = trans_test_X
        
        if self.k == no_neighbours_to_consider:
            
            nd,ni = self.knn_model.kneighbors(test_X)
        
        elif self.train_X.shape[0] == no_neighbours_to_consider:
            
            self.buildNNmodel(no_neighbours_to_consider)
            
            nd,ni = self.all_training_set_compounds_nn_model.kneighbors(test_X)
        else:
            raise Exception(f'Why are you trying to compute the distances between new compounds and their closest {no_neighbours_to_consider} compounds in the training set, when k={self.k} and training set size={self.train_X.shape[0]}?')
        
        #-------------------------
        if not nd.shape[0] == len(test_ids): raise Exception('nd.shape[0] =%d, len(test_ids)=%d' % (nd.shape[0],len(test_ids)))
        if not ni.shape[0] == len(test_ids): raise Exception('ni.shape[0] =%d, len(test_ids)=%d' % (ni.shape[0],len(test_ids)))
        if not nd.shape[1]==no_neighbours_to_consider: raise Exception('nd.shape[1]=%d,no_neighbours_to_consider=%d' % (nd.shape[1],no_neighbours_to_consider))
        if not ni.shape[1]==no_neighbours_to_consider: raise Exception('ni.shape[1]=%d,no_neighbours_to_consider=%d' % (ni.shape[1],no_neighbours_to_consider))
        #--------------------------
        
        nd_list = nd.tolist()
        ni_list = ni.tolist()
        
        return nd_list,ni_list,test_ids
    
    def getADstatusOfTestSetCompounds(self,test_compounds_X_df,id_col):
        
        
        nd_list,ni_list,test_ids = self.getNewCmpdsDistancesToTrainNN(test_compounds_X_df,id_col,no_neighbours_to_consider=self.train_X.shape[0])
        
        test_id_ad_status_dict = defaultdict(dict)
        
        for test_cmpd_index in range(0,len(test_ids)):
            
            test_ID = test_ids[test_cmpd_index]
            
            #============================
            #This could be updated to document the extent to which the test compound can be considered to lie inside the AD:
            test_id_ad_status_dict[test_ID]['InsideAD']=False
            #=============================
            
            #thresholds_met = 0
            
            for nn_index in range(0,self.train_X.shape[0]):
                
                train_set_index = ni_list[test_cmpd_index][nn_index]
                
                train_set_id = self.train_index_to_id_dict[train_set_index]
                
                train_set_threshold = self.TrainIdsMatchedToThresholds[train_set_id]
                
                train_test_distance = nd_list[test_cmpd_index][nn_index]
                
                if train_set_threshold >= train_test_distance:
                    #thresholds_met += 1
                    
                    #This could be updated to document the extent to which the test compound can be considered to lie inside the AD:
                    
                    test_id_ad_status_dict[test_ID]['InsideAD']=True

                    if self.debug:
                        print(f'train_set_id={train_set_id}, test_ID={test_ID}, train_test_distance={train_test_distance}, train_set_threshold={train_set_threshold}')
                        print(f'self.distance_metric={self.distance_metric}')
                        
                    if not self.debug:
                        break
                
        return test_id_ad_status_dict

def check_dk_NN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations():
    ###########################
    examples = defaultdict(dict)
    #-----------------------------
    examples[1]['train_df'] = pd.DataFrame({'ID':[1,2,4,5],'x1':[1.0,2.0,100.0,1.0]})
    examples[1]['id_col'] = 'ID'
    examples[1]['k'] = 2
    examples[1]['distance_metric'] = 'manhattan'
    examples[1]['test_compounds_X_df']=pd.DataFrame({'ID':[10,20,30,50],'x1':[101.0,2.0,101.0,1.5]})
    #======
    examples[1]['expected: train_knn_dist_array'] = np.array([[0,1],[1,1],[98.0,99.0],[0,1]])
    examples[1]['expected: d_k_av_i_array'] = np.array([0.5,1.0,98.5,0.5])
    examples[1]['expected: RefVal'] = 62.6875
    examples[1]['expected: TrainIdsMatchedToThresholds'] = dict(zip([1,2,4,5],[0.5,1.0,0.0,0.5]))
    examples[1]['expected: updated TrainIdsMatchedToThresholds'] = dict(zip([1,2,4,5],[0.5,1.0,0.5,0.5]))
    examples[1]['expected: test_id_ad_status_dict'] = defaultdict(dict)
    examples[1]['expected: test_id_ad_status_dict'][10]['InsideAD']=False
    examples[1]['expected: test_id_ad_status_dict'][20]['InsideAD']=True
    examples[1]['expected: test_id_ad_status_dict'][30]['InsideAD']=False
    examples[1]['expected: test_id_ad_status_dict'][50]['InsideAD']=True
    examples[1]['scale'] = False
    examples[1]['descs_no']=1
    examples[1]['only_consider_k_neighbours_for_final_threshold_calculations'] = True
    #------------------------------
    #-----------------------------
    examples[2]['train_df'] = pd.DataFrame({'ID':[1,2,4,5],'x1':[0,0,0,1]})
    examples[2]['id_col'] = 'ID'
    examples[2]['k'] = 2
    examples[2]['distance_metric'] = 'manhattan'
    examples[2]['test_compounds_X_df']=pd.DataFrame({'ID':[10,20,30,50],'x1':[1000.0,2.0,101.0,1.5]})
    #======
    examples[2]['expected: train_knn_dist_array'] = np.array([[0,0],[0,0],[0,0],[1,1]])
    examples[2]['expected: d_k_av_i_array'] = np.array([0,0,0,1])
    examples[2]['expected: RefVal'] = 0.625
    examples[2]['expected: TrainIdsMatchedToThresholds'] = dict(zip([1,2,4,5],[0,0,0,0]))
    examples[2]['expected: updated TrainIdsMatchedToThresholds'] = dict(zip([1,2,4,5],[0,0,0,0]))
    examples[2]['expected: test_id_ad_status_dict'] = defaultdict(dict)
    examples[2]['expected: test_id_ad_status_dict'][10]['InsideAD']=False
    examples[2]['expected: test_id_ad_status_dict'][20]['InsideAD']=False
    examples[2]['expected: test_id_ad_status_dict'][30]['InsideAD']=False
    examples[2]['expected: test_id_ad_status_dict'][50]['InsideAD']=False
    examples[2]['scale'] = False
    examples[2]['descs_no']=1
    examples[2]['only_consider_k_neighbours_for_final_threshold_calculations'] = True
    #------------------------------
    #*******************************
    for example_no in examples.keys():
        print('Running check_dk_NN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations: example=%d' % example_no)
        dk_NN_thresholds_instance = dk_NN_thresholds(train_df=examples[example_no]['train_df'],id_col=examples[example_no]['id_col'],k=examples[example_no]['k'],distance_metric=examples[example_no]['distance_metric'],scale=examples[example_no]['scale'],expected_descs_no=examples[example_no]['descs_no'],only_consider_k_neighbours_for_final_threshold_calculations=examples[example_no]['only_consider_k_neighbours_for_final_threshold_calculations'],debug=True)
        
        dk_NN_thresholds_instance.getInitialTrainingSetThresholds()
        
        if not np.array_equal(examples[example_no]['expected: train_knn_dist_array'],dk_NN_thresholds_instance.train_knn_dist_array): raise Exception("examples[example_no]['expected: train_knn_dist_array'] =%s, dk_NN_thresholds_instance.train_knn_dist_array=%s" % (str(examples[example_no]['expected: train_knn_dist_array']),str(dk_NN_thresholds_instance.train_knn_dist_array)))
        if not np.array_equal(examples[example_no]['expected: d_k_av_i_array'],dk_NN_thresholds_instance.d_k_av_i_array): raise Exception("examples[example_no]['expected: d_k_av_i_array']=%s, dk_NN_thresholds_instance.d_k_av_i_array=%s" % (str(examples[example_no]['expected: d_k_av_i_array']),str(dk_NN_thresholds_instance.d_k_av_i_array)))
        if not examples[example_no]['expected: RefVal'] == dk_NN_thresholds_instance.RefVal: raise Exception("examples[example_no]['expected: RefVal'] =%s, dk_NN_thresholds_instance.RefVal = %s" % (str(examples[example_no]['expected: RefVal']),str(dk_NN_thresholds_instance.RefVal)))
        if not examples[example_no]['expected: TrainIdsMatchedToThresholds'] == dk_NN_thresholds_instance.TrainIdsMatchedToThresholds: raise Exception("examples[example_no]['expected: TrainIdsMatchedToThresholds'] =%s, dk_NN_thresholds_instance.TrainIdsMatchedToThresholds=%s" % (str(examples[example_no]['expected: TrainIdsMatchedToThresholds']),str(dk_NN_thresholds_instance.TrainIdsMatchedToThresholds)))
        
        dk_NN_thresholds_instance.updateZeroValuedTrainingSetThresholds()
        if not examples[example_no]['expected: updated TrainIdsMatchedToThresholds'] == dk_NN_thresholds_instance.TrainIdsMatchedToThresholds: raise Exception("examples[example_no]['expected: updated TrainIdsMatchedToThresholds'] =%s, dk_NN_thresholds_instance.TrainIdsMatchedToThresholds=%s" % (str(examples[example_no]['expected: updated TrainIdsMatchedToThresholds']),str(dk_NN_thresholds_instance.TrainIdsMatchedToThresholds)))
        
        test_id_ad_status_dict=dk_NN_thresholds_instance.getADstatusOfTestSetCompounds(test_compounds_X_df=examples[example_no]['test_compounds_X_df'],id_col=examples[example_no]['id_col'])
        if not examples[example_no]['expected: test_id_ad_status_dict'] == test_id_ad_status_dict: raise Exception("examples[example_no]['expected: test_id_ad_status_dict'] =%s, test_id_ad_status_dict=%s" % (str(examples[example_no]['expected: test_id_ad_status_dict']),str(test_id_ad_status_dict)))
        print('RAN check_dk_NN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations: example=%d' % example_no)
    #********************************

#check_dk_NN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations()
