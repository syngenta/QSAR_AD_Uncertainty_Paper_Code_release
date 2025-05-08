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
#Copyright (c) 2020-2023 Syngenta
#Contact richard.marchese_robinson@syngenta.com
#######################
#########################
#This code is designed to update the dk_NN_thresholds to obtain the RDN thresholds as part of -  a variation on - the following applicability domain approach, as well as enable new test chemicals to be evaluated against these thresholds, by computing their distances to all test compounds using the underlying k-NN model:
#Aniceto et al. J Cheminform (2016) 8:69
#DOI 10.1186/s13321-016-0182-y
#N.B. See additional references to 'Aniceto' in the code for details of differences between the original approach and the approach implemented herein.
#######################
import sys,re,os,functools,math
import numpy as np
import pandas as pd
from collections import defaultdict
from .dk_NN_thresholds import dk_NN_thresholds
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import StratifiedKFold,ShuffleSplit,StratifiedShuffleSplit, KFold
from copy import deepcopy
import pickle
from math import exp
#=================================
#dir_of_file = os.path.dirname(os.path.abspath(__file__))
#top_dir = os.path.dirname(dir_of_file)
#==================================
from ..utils.load_example_datasets_for_code_checks import generateExampleDatasetForChecking_BCWD
from ..utils.ML_utils import get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance,k_fold_cv_with_stratified_sampling_for_continuous_or_categorical_y
#======================================


class RDN_thresholds(dk_NN_thresholds):
    
    def adHocScaling(self):
        if self.k < 31:
            scaling_factor = (1/3) #Python 3 required for this to return one third!
        elif self.k < 41:
            scaling_factor = 0.5
        else:
            scaling_factor = 1
        
        for key in self.TrainIdsMatchedToThresholds.keys():
            self.TrainIdsMatchedToThresholds[key]*=scaling_factor
    
    def get_x__y_ids_from_df_which_only_contains_these(self,df,id_col,endpoint_col):
        x = df.copy()
        x = x.drop(labels=[id_col],axis=1)
        x = x.drop(labels=[endpoint_col],axis=1)
        
        y = df[endpoint_col]

        ids = df[id_col].tolist()

        return x,y,ids


    def getEnsembleTrainXandYandIds(self,ensemble_train_df,id_col,endpoint_col):
        #----------------------------
        if not type(pd.DataFrame({'a':[1]})) == type(ensemble_train_df): raise Exception(type(ensemble_train_df))
        if not id_col in ensemble_train_df.columns.values.tolist(): raise Exception("id_col =%s, ensemble_train_df.columns.values.tolist()=%s" % (id_col,str(ensemble_train_df.columns.values.tolist())))
        if not endpoint_col in ensemble_train_df.columns.values.tolist(): raise Exception("endpoint_col =%s, ensemble_train_df.columns.values.tolist()=%s" % (endpoint_col,str(ensemble_train_df.columns.values.tolist())))
        #-----------------------------
        ensemble_train_df_ids = ensemble_train_df[id_col].tolist()
        #------------------------------
        if not ensemble_train_df_ids == self.train_ids.tolist(): raise Exception("ensemble_train_df_ids=%d,self.train_ids=%d" % (str(ensemble_train_df_ids),str(self.train_ids)))
        #-------------------------------
        
        ensemble_train_x,ensemble_train_y,ensemble_train_df_ids = self.get_x__y_ids_from_df_which_only_contains_these(ensemble_train_df,id_col,endpoint_col)
        
        return ensemble_train_x,ensemble_train_y,ensemble_train_df_ids
    
    def getCrossValInputs(self,id_col,endpoint_col,cv_folds_no=10,SEED=100,write_out_intermediate_res_for_checks=False, regression=False):
        
        ensemble_train_df = self.train_df
        
        ensemble_train_x,ensemble_train_y,ensemble_train_df_ids = self.getEnsembleTrainXandYandIds(ensemble_train_df,id_col,endpoint_col)
        
        fold2TrainAndTestDict = k_fold_cv_with_stratified_sampling_for_continuous_or_categorical_y(df=ensemble_train_df,y_col=endpoint_col,id_col=id_col,K=cv_folds_no,rand_state=SEED,reset_indices=True,regression=regression)
        
        self.cross_val_inputs_dict = defaultdict(functools.partial(defaultdict,dict))
        
        all_fold_ids = []

        for fold_count in range(cv_folds_no):
            for subset_label in ['cv_train','cv_test']:
                subset = subset_label

                subset_df = fold2TrainAndTestDict[fold_count][subset_label]

                subset_x,subset_y,subset_ids = self.get_x__y_ids_from_df_which_only_contains_these(subset_df,id_col,endpoint_col)

                if not regression:
                    assert 1 in subset_y.tolist() and 0 in subset_y.tolist(),f'Binary classification subset does not include one class label: subset={subset}, subset_y={subset_y.tolist()}'
                    extra_class_labels = [v for v in subset_y.tolist() if not v in [1,0]]
                    assert 0 == len(extra_class_labels),f'Binary classification only allows two class labels: extra_class_labels={extra_class_labels}'

                self.cross_val_inputs_dict[fold_count][subset]['x'] = subset_x
                self.cross_val_inputs_dict[fold_count][subset]['y'] = subset_y
                self.cross_val_inputs_dict[fold_count][subset]['ids'] = subset_ids
            
            if not 0 == len(set(self.cross_val_inputs_dict[fold_count]['cv_train']['ids']).intersection(set(self.cross_val_inputs_dict[fold_count]['cv_test']['ids']))): raise Exception('Overlap = %s' % str(list(set(self.cross_val_inputs_dict[fold_count]['cv_train']['ids']).intersection(set(self.cross_val_inputs_dict[fold_count]['cv_test']['ids'])))))
            all_fold_ids += self.cross_val_inputs_dict[fold_count]['cv_test']['ids']
        
        #-------------------------------------------
        all_fold_ids.sort()
        if not len(all_fold_ids) > cv_folds_no: raise Exception('not len(all_fold_ids) > cv_folds_no')
        if not len(all_fold_ids)==len(set(all_fold_ids)): raise Exception('not len(all_fold_ids)==len(set(all_fold_ids))')
        ensemble_train_df_ids.sort()
        if not all_fold_ids == ensemble_train_df_ids: raise Exception('not len(all_fold_ids)==len(set(all_fold_ids))')
        #-------------------------------------------
        
        #===========================================
        if write_out_intermediate_res_for_checks:
            f_o = open('cross_val_inputs_dict.pkl','wb')
            try:
                pickle.dump(self.cross_val_inputs_dict,f_o)
            finally:
                f_o.close()
        #============================================
    
    def getTreeEnsemble(self,train_x,train_y,n_models_in_RDN_ensemble,SEED,use_sklearn_rf,regression=False):
        
        if use_sklearn_rf:
            if not regression:
                rf = RandomForestClassifier(n_estimators=n_models_in_RDN_ensemble,random_state=SEED).fit(train_x,train_y) #default: max_features="auto" , meaning sqrt(number of features) [sklearn version 1.0.2 documentation]
            else:
                rf = RandomForestRegressor(n_estimators=n_models_in_RDN_ensemble,random_state=SEED,max_features=(1.0/3)).fit(train_x,train_y) #Ensure mtry default for regression, as for classification, matches Svetnik et al. (2003) [https://pubs.acs.org/doi/full/10.1021/ci034160g]
            
            list_of_ensemble_trees = [tree for tree in rf.estimators_]
            
            if not regression:
                all_classes_in_order = rf.classes_.tolist()
            else:
                all_classes_in_order = None
        else: 
            raise Exception('We are only interested in RandomForest decision tree ensemble!')
        
        return list_of_ensemble_trees,all_classes_in_order,rf
    
    def compute_stdev_measure(self,predictions,regression):
        if not regression:
            stdev = np.std(predictions)
            return stdev
        else:
            normalized_stdev = (1-exp(-np.std(predictions)))/(1+exp(-np.std(predictions)))
            assert normalized_stdev >=0 and normalized_stdev <= 1, normalized_stdev
            return normalized_stdev

    def get_agreement_and_prediction_for_a_single_instance_and_tree(self,test_y,rf,list_of_ensemble_trees,all_classes_in_order,tree_index,test_instance_row_x,instance_index,regression,round_dt_preds):
        index=instance_index

        tree = list_of_ensemble_trees[tree_index]
                
        if not regression:
            if all_classes_in_order is None:
                all_classes_in_order = tree.classes_.tolist()
            
            
            
            #----------------------------------
            if not type(all_classes_in_order) == type([]): raise Exception('type(all_classes_in_order)=%s' % str(type(all_classes_in_order)))
            if not len(all_classes_in_order) == len(set(all_classes_in_order)): raise Exception("all_classes_in_order=%s" % str(all_classes_in_order))
            #----------------------------------
            
            
            probs_for_all_classes = get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance(tree=tree,ensemble=rf,tree_index=tree_index,test_X=test_instance_row_x,expected_type_of_test_X=pd.DataFrame,want_class_probs=True,ignore_user_warning=self.ignore_warnings)
            #-------------------------
            if not 1 == probs_for_all_classes.shape[0]: raise Exception('probs_for_all_classes.shape[0]=%d' % probs_for_all_classes.shape[0])
            if not len(all_classes_in_order) == probs_for_all_classes.shape[1]: raise Exception('probs_for_all_classes.shape[1]=%d' % probs_for_all_classes.shape[1])
            #-------------------------------
            
            true_class = test_y[index]
            
            try:
                true_class_index = all_classes_in_order.index(true_class)
            except ValueError as err:
                raise Exception(f'err={err},true_class={true_class},all_classes_in_order={all_classes_in_order}')
            
            #--------------------------
            if not type(1) == type(true_class_index): raise Exception('type(true_class_index)=%s' % str(type(true_class_index)))
            #--------------------------
            
            
            try:
                prob_for_true_class = float(probs_for_all_classes[:,true_class_index])
            except TypeError:
                print(true_class_index)
                print(true_class)
                print(all_classes_in_order)
                print(probs_for_all_classes.tolist())
                raise Exception('?')
            
            #---------------------------
            if not prob_for_true_class >= 0 and prob_for_true_class <= 1: raise Exception('prob_for_true_class=%f' % prob_for_true_class)
            #---------------------------
            
            if round_dt_preds:
                prob_for_true_class = round(prob_for_true_class,0)
                #---------------------------
                #if not math.isclose(prob_for_true_class,0,abs_tol=10**-5) or math.isclose(prob_for_true_class,1,abs_tol=10**-5): raise Exception('prob_for_true_class=%f' % prob_for_true_class)
                #Exception: prob_for_true_class=1.000000 #???
                if not round(0.99,0) == 1: raise Exception('!')
                if not round(0.51,0) == 1: raise Exception('!')
                if not round(0.49,0) == 0: raise Exception('!')
                #---------------------------
            
            pred_val = prob_for_true_class
            
            #--------------------------------------
            if not 2 == len(all_classes_in_order): raise Exception('The code would require further adaptation to work with more than two categories!')
            #---------------------------------------
            
            if prob_for_true_class > 0.5: #This assumes that there are only two classes!
                agreement_val = 1
            else:
                agreement_val = 0
        else:
            preds_y = get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance(tree=tree,ensemble=rf,tree_index=tree_index,test_X=test_instance_row_x,expected_type_of_test_X=pd.DataFrame,want_class_probs=False,ignore_user_warning=self.ignore_warnings)
            
            if not 1 == len(preds_y): raise Exception('preds_y={}'.format(preds_y))
            if not type(0.5)==type(float(preds_y[0])): raise Exception('type(float(preds_y[0]))={}'.format(type(float(preds_y[0]))))
            
            y_pred = preds_y[0]
            
            pred_val = y_pred
            
            true_y = float(test_y[index])
            
            if not type(0.5)==type(true_y): raise Exception('type(true_y)={}'.format(type(true_y)))
            
            agreement_val = exp(-abs(y_pred-true_y))

        return agreement_val,pred_val                             
    
    def getEnsembleAgreementStdVals_ForASingleTrainTestDataPair(self,fold,train_x,train_y,test_x,test_y,test_ids,n_models_in_RDN_ensemble,SEED,use_sklearn_rf,round_dt_preds=False,write_out_intermediate_res_for_checks=False,regression=False):
        #-------------------
        ####################################
        #20/9/23: Using scikit-learn==1.0.2, I am still getting the following warning, even though I explicitly check the consistency of train_x and test_x, as well as the subset of test_x used for predictions below: "UserWarning: X has feature names, but DecisionTreeRegressor was fitted without feature names"
        #https://forums.fast.ai/t/randomforestregressor-estimators-method/93942
        ###################################
        if not type(pd.DataFrame({'a':[1]}))==type(train_x): raise Exception('type(train_x)=%s' % type(train_x))
        if not type(pd.DataFrame({'a':[1]}))==type(test_x): raise Exception('type(test_x)=%s' % type(test_x))
        if not train_x.shape[1] == test_x.shape[1]: raise Exception(f'train_x.shape[1] = {train_x.shape[1]} vs. test_x.shape[1] = {test_x.shape[1]}')
        if not list(range(test_x.shape[0]))==list(test_x.index): raise Exception(f'list(range(test_x.shape[1]))= {list(range(test_x.shape[1]))} vs. list(test_x.index) = {list(test_x.index)}') #If this is not true, there could be problems with get the instance specific dataframes and corresponding IDs for the test set below.
        if not type(pd.Series([1]))==type(train_y): raise Exception('type(train_y)=%s' % type(train_y))
        if not type(pd.Series([1]))==type(test_y): raise Exception('type(test_y)=%s' % type(test_y))
        if not type([]) == type(test_ids): raise Exception("type(test_ids)=%s" % type(test_ids))
        #-------------------
        
        #print('fold=%s' % str(fold))
        #print('train_y=%s' % str(train_y))
        
        list_of_ensemble_trees,all_classes_in_order,rf = self.getTreeEnsemble(train_x,train_y,n_models_in_RDN_ensemble,SEED,use_sklearn_rf,regression) #all_classes_in_order may just be a dummy value (None) - see below!
        
        #===========================================
        if write_out_intermediate_res_for_checks:
            f_o = open('list_of_ensemble_trees_fold={}.pkl'.format(fold),'wb')
            try:
                pickle.dump({'list_of_ensemble_trees':list_of_ensemble_trees,'all_classes_in_order':all_classes_in_order},f_o)
            finally:
                f_o.close()
        #============================================
        
        for index in test_x.index.tolist():
            
            #try:
            test_instance_row_x = test_x.iloc[[index]] #Double brackets gave the desired dataframe version of this row during testing of this syntax! 
            #except IndexError:
            #    print('index=%s' % str(index))
            #    print('type(test_x)=%s' % str(type(test_x)))
            #    raise Exception('What is the problem?') #Indices need to be reset for .iloc[[index]] to work!
            #-------------------------
            if not type(test_instance_row_x) == type(pd.DataFrame({'a':[1]})): raise Exception('type(test_instance_row_x)=%s' % str(type(test_instance_row_x)))
            #-------------------------
            
            test_instance_id =  test_ids[index]
            
            #agreement_list = []
            
            #prob_of_true_class_list = [] #Is this really what we should be considering, as opposed to the probabilities for a fixed 'class 1'? However, otherwise, how would this be applied to the multi-class case?
            #predictions = [] # = prob_of_true_class_list if not regression
            
            agreement_pred_tuples_list_for_all_trees = [self.get_agreement_and_prediction_for_a_single_instance_and_tree(test_y,rf,list_of_ensemble_trees,all_classes_in_order,tree_index,test_instance_row_x,instance_index=index,regression=regression,round_dt_preds=round_dt_preds) for tree_index in range(0,len(list_of_ensemble_trees))]

            agreement_list = [t[0] for t in agreement_pred_tuples_list_for_all_trees]
            predictions = [t[1] for t in agreement_pred_tuples_list_for_all_trees]

            self.id_to_A_STD_vals[test_instance_id]['STD']=self.compute_stdev_measure(predictions,regression)
            self.id_to_A_STD_vals[test_instance_id]['A']=sum(agreement_list)/len(agreement_list) #Relies on Python 3
            
        
    
    def getCrossValEnsembleAgreementStdVals(self,n_models_in_RDN_ensemble=10,SEED=100,use_sklearn_rf=True,round_dt_preds=False,write_out_intermediate_res_for_checks=False,regression=False):
        ##############################################################
        #This differs from the original work of Aniceto et al. (2016) [https://link.springer.com/article/10.1186/s13321-016-0182-y]#
        #In their work, they trained a single decision tree on a sample of the training data, then repeated this 10 times, in order to obtain the standard deviation values.
        #Here, the K-fold cross-validated models and their cross-validation test set predictions are used to compute the agreement and standard deviation values.
        ##For binary classification (multi-class classification is not supported):
        #Unlike the description in the paper (equation 1), we use M (no. models) as the denominator. This ensures STD remains in the range 0 to 1 for binary classification.
        ##For regression:
        #Aniceto et al. (2016) did not consider regression tasks.
        #Here, to ensure agreement remains in in the range [0 - 1], each contribution to agreement was defined as exp(-abs(pred - experi))
        #STD (standard deviation) was replaced with a normalized measure, to ensure that this also remains in the range 0 to 1:
        #see def compute_stdev_measure(...)
        ##############################################################
        
        #self.id_to_A_STD_vals = defaultdict(dict) #Create elsewhere!
        
        
        
        for fold in self.cross_val_inputs_dict.keys():
            train_x = self.cross_val_inputs_dict[fold]['cv_train']['x']
            train_y = self.cross_val_inputs_dict[fold]['cv_train']['y']
            test_x = self.cross_val_inputs_dict[fold]['cv_test']['x']
            test_y = self.cross_val_inputs_dict[fold]['cv_test']['y']
            
            test_ids = self.cross_val_inputs_dict[fold]['cv_test']['ids']
            
            self.getEnsembleAgreementStdVals_ForASingleTrainTestDataPair(fold,train_x,train_y,test_x,test_y,test_ids,n_models_in_RDN_ensemble,SEED,use_sklearn_rf,round_dt_preds,write_out_intermediate_res_for_checks,regression)
    
    def updateTrainingSetThresholds_Using_A_and_STD(self,write_out_intermediate_res_for_checks=False):
        self.dk_NN_TrainIdsMatchedToThresholds = deepcopy(self.TrainIdsMatchedToThresholds)
        
        for test_instance_id in self.TrainIdsMatchedToThresholds.keys():
            W_i = (1-self.id_to_A_STD_vals[test_instance_id]['STD'])*self.id_to_A_STD_vals[test_instance_id]['A']
            ################
            assert W_i >=0,W_i
            #################
            ##############
            #Debug:
            if write_out_intermediate_res_for_checks:
                print('test_instance_id={},W_i={}'.format(test_instance_id,W_i))
            ###############
            #Difference: Aniceto code indicates I should use a different value when W_i=0: the minimum of the non-zero values. However, I cannot find a description of this in the text and I think allowing W_i=0 is justified.
            
            self.TrainIdsMatchedToThresholds[test_instance_id]*=W_i
    
    def assignRdnTrainingSetSpecificThresholds(self,id_col,endpoint_col,n_models_in_RDN_ensemble=10,cv_folds_no=10,SEED=100,use_k_fold_cv_for_ensemble_preds=True,use_sklearn_rf=True,round_dt_preds=False,write_out_intermediate_res_for_checks=False,useAdHocScaling=False,regression=False):
        self.getInitialTrainingSetThresholds()
        
        self.updateZeroValuedTrainingSetThresholds()
        
        self.id_to_A_STD_vals = defaultdict(dict)
        
        if use_k_fold_cv_for_ensemble_preds:
        
            self.getCrossValInputs(id_col,endpoint_col,cv_folds_no,SEED,write_out_intermediate_res_for_checks, regression=regression)
            
            self.getCrossValEnsembleAgreementStdVals(n_models_in_RDN_ensemble,SEED,use_sklearn_rf,round_dt_preds,write_out_intermediate_res_for_checks,regression)
        else:
            
            raise Exception('We are only interested in use_k_fold_cv_for_ensemble_preds=True')
        
        self.updateTrainingSetThresholds_Using_A_and_STD(write_out_intermediate_res_for_checks)
        
        if useAdHocScaling:
            self.adHocScaling() #How general is adHocScaling() proposed by Aniceto et al. (2016), especially since, as per notes in the code, we are using a variation on their approach?

#generateExampleDatasetForChecking_BCWD(write_to_csv=True)

def check_RDN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations():
    write_out_intermediate_res_for_checks=False #03/11/23: set to True again to enable checking after new stratified CV routine used. #True #Set this to True as cannot compute all RDN steps by hand, but can at least retrieve details not possible to compute by hand and use these to check the remaining steps!
    
    ###########################
    examples = defaultdict(dict)
    # #=================================================
    # bc_df = generateExampleDatasetForChecking_BCWD()
    # bc_df_train = bc_df.iloc[range(0,560)]
    # bc_df_test = bc_df.iloc[range(560,569)]
    # #=================================================
    #For example 1, try and use the same, simple example as per first dk-NN check:
    examples[1]['train_df'] = pd.DataFrame({'ID':[1,2,4,5],'x1':[1.0,2.0,100.0,1.0],'Class':[1,0,0,1]})
    examples[1]['id_col'] = 'ID'
    examples[1]['k'] = 2
    examples[1]['distance_metric'] = 'manhattan'
    examples[1]['test_compounds_X_df']=pd.DataFrame({'ID':[10,20,30,50],'x1':[101.0,2.0,101.0,1.5]})
    examples[1]['endpoint_col']='Class'
    examples[1]['n_models_in_RDN_ensemble']=10
    examples[1]['cv_folds_no']=2
    examples[1]['SEED']=100
    examples[1]['use_k_fold_cv_for_ensemble_preds']=True
    examples[1]['use_sklearn_rf']=True
    examples[1]['round_dt_preds']=True
    examples[1]['scale']=False
    #======
    examples[1]['expected: dk_NN_TrainIdsMatchedToThresholds'] = dict(zip([1,2,4,5],[0.5,1.0,0.5,0.5]))
    w_i_dict = {4:0.379,5:0.630,1:0.630,2:0.070} #03/11/23: needed to update this after new stratified CV routine used. #parsing of 'cross_val_inputs_dict.pkl' and 'list_of_ensemble_trees_fold={}.pkl'.format(fold), plus Excel completion of calculations, yielded these w_i values.
    examples[1]['expected: RDN TrainIdsMatchedToThresholds'] = dict(zip([1,2,4,5],[examples[1]['expected: dk_NN_TrainIdsMatchedToThresholds'][id_]*w_i_dict[id_] for id_ in [1,2,4,5]]))
    examples[1]['expected: RDN test_id_ad_status_dict'] = defaultdict(dict)
    examples[1]['expected: RDN test_id_ad_status_dict'][10]['InsideAD']=False #From dkNN checking example [1]: must be consistent
    examples[1]['expected: RDN test_id_ad_status_dict'][20]['InsideAD']=True #From dkNN checking example [1]:True
    examples[1]['expected: RDN test_id_ad_status_dict'][30]['InsideAD']=False #From dkNN checking example [1]: must be consistent
    examples[1]['expected: RDN test_id_ad_status_dict'][50]['InsideAD']=False #From dkNN checking example [1]:True - OK to observe this go to False here
    #*******************************
    for example_no in examples.keys():
        print('Running check_RDN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations: example=%d' % example_no)
        RDN_thresholds_instance = RDN_thresholds(train_df=examples[example_no]['train_df'],id_col=examples[example_no]['id_col'],k=examples[example_no]['k'],distance_metric=examples[example_no]['distance_metric'],scale=examples[example_no]['scale'],endpoint_col=examples[example_no]['endpoint_col'],expected_descs_no=1,only_consider_k_neighbours_for_final_threshold_calculations=True)
        
        RDN_thresholds_instance.assignRdnTrainingSetSpecificThresholds(id_col=examples[example_no]['id_col'],endpoint_col=examples[example_no]['endpoint_col'],n_models_in_RDN_ensemble=examples[example_no]['n_models_in_RDN_ensemble'],cv_folds_no=examples[example_no]['cv_folds_no'],SEED=examples[example_no]['SEED'],use_k_fold_cv_for_ensemble_preds=examples[example_no]['use_k_fold_cv_for_ensemble_preds'],use_sklearn_rf=examples[example_no]['use_sklearn_rf'],round_dt_preds=examples[example_no]['round_dt_preds'],write_out_intermediate_res_for_checks=write_out_intermediate_res_for_checks)
        
        if not examples[example_no]['expected: dk_NN_TrainIdsMatchedToThresholds'] == RDN_thresholds_instance.dk_NN_TrainIdsMatchedToThresholds: raise Exception("examples[example_no]['expected: dk_NN_TrainIdsMatchedToThresholds'] =%s, RDN_thresholds_instance.dk_NN_TrainIdsMatchedToThresholds=%s" % (str(examples[example_no]['expected: dk_NN_TrainIdsMatchedToThresholds']),str(RDN_thresholds_instance.dk_NN_TrainIdsMatchedToThresholds)))
        
        
        if not list(examples[example_no]['expected: RDN TrainIdsMatchedToThresholds'].keys()) == list(RDN_thresholds_instance.TrainIdsMatchedToThresholds.keys()): raise Exception("examples[example_no]['expected: RDN TrainIdsMatchedToThresholds'] =%s, RDN_thresholds_instance.TrainIdsMatchedToThresholds=%s" % (str(examples[example_no]['expected: RDN TrainIdsMatchedToThresholds']),str(RDN_thresholds_instance.TrainIdsMatchedToThresholds)))
        
        for id_ in examples[example_no]['expected: RDN TrainIdsMatchedToThresholds'].keys():
            if not round(examples[example_no]['expected: RDN TrainIdsMatchedToThresholds'][id_],3)==round(RDN_thresholds_instance.TrainIdsMatchedToThresholds[id_],3):raise Exception("examples[example_no]['expected: RDN TrainIdsMatchedToThresholds'] =%s, RDN_thresholds_instance.TrainIdsMatchedToThresholds=%s" % (str(examples[example_no]['expected: RDN TrainIdsMatchedToThresholds']),str(RDN_thresholds_instance.TrainIdsMatchedToThresholds)))
        
        test_id_ad_status_dict=RDN_thresholds_instance.getADstatusOfTestSetCompounds(test_compounds_X_df=examples[example_no]['test_compounds_X_df'],id_col=examples[example_no]['id_col'])
        
        if not examples[example_no]['expected: RDN test_id_ad_status_dict'] == test_id_ad_status_dict: raise Exception("examples[example_no]['expected: RDN test_id_ad_status_dict'] =%s, test_id_ad_status_dict=%s" % (str(examples[example_no]['expected: RDN test_id_ad_status_dict']),str(test_id_ad_status_dict)))
        
        print('RAN check_RDN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations: example=%d' % example_no)
    #********************************

#check_RDN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations()

def check_RDN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations_regression():
    write_out_intermediate_res_for_checks=False
    
    ###########################
    examples = defaultdict(dict)
    #*******************************
    #For example 1, try and use the same, simple example as per first dk-NN check:
    examples[1]['train_df'] = pd.DataFrame({'ID':[1,2,4,5],'x1':[1.0,2.0,100.0,1.0],'Class':[1,0,0,1]})
    examples[1]['id_col'] = 'ID'
    examples[1]['k'] = 2
    examples[1]['distance_metric'] = 'manhattan'
    examples[1]['test_compounds_X_df']=pd.DataFrame({'ID':[10,20,30,50],'x1':[101.0,2.0,101.0,1.5]})
    examples[1]['endpoint_col']='Class'
    examples[1]['n_models_in_RDN_ensemble']=10
    examples[1]['cv_folds_no']=2
    examples[1]['SEED']=100
    examples[1]['use_k_fold_cv_for_ensemble_preds']=True
    examples[1]['use_sklearn_rf']=True
    examples[1]['round_dt_preds']=True
    examples[1]['scale']=False
    #======
    examples[1]['expected: dk_NN_TrainIdsMatchedToThresholds'] = dict(zip([1,2,4,5],[0.5,1.0,0.5,0.5]))
    examples[1]['expected: RDN TrainIdsMatchedToThresholds'] = {1: 0.18393972058572117, 2: 0.36787944117144233, 4: 0.18393972058572117, 5: 0.18393972058572117} #When switching from StratifiedKFold toKFold, the thresholds were updated. #When switched to regression, this was the observed outcome #dict(zip([1,2,4,5],[examples[1]['expected: dk_NN_TrainIdsMatchedToThresholds'][id_]*w_i_dict[id_] for id_ in [1,2,4,5]]))
    examples[1]['expected: RDN test_id_ad_status_dict'] = defaultdict(dict)
    examples[1]['expected: RDN test_id_ad_status_dict'][10]['InsideAD']=False 
    examples[1]['expected: RDN test_id_ad_status_dict'][20]['InsideAD']=True 
    examples[1]['expected: RDN test_id_ad_status_dict'][30]['InsideAD']=False 
    examples[1]['expected: RDN test_id_ad_status_dict'][50]['InsideAD']=False
    #*******************************
    #For example 2, check code works with non-integer endpoint values:
    examples[2]['train_df'] = pd.DataFrame({'ID':[1,2,4,5],'x1':[1.0,2.0,100.0,1.0],'Activity':[1.58,0.02,0.03,1.7]})
    examples[2]['id_col'] = 'ID'
    examples[2]['k'] = 2
    examples[2]['distance_metric'] = 'manhattan'
    examples[2]['test_compounds_X_df']=pd.DataFrame({'ID':[10,20,30,50],'x1':[101.0,2.0,101.0,1.5]})
    examples[2]['endpoint_col']='Activity'
    examples[2]['n_models_in_RDN_ensemble']=10
    examples[2]['cv_folds_no']=2
    examples[2]['SEED']=100
    examples[2]['use_k_fold_cv_for_ensemble_preds']=True
    examples[2]['use_sklearn_rf']=True
    examples[2]['round_dt_preds']=True
    examples[2]['scale']=False
    #======
    examples[2]['expected: dk_NN_TrainIdsMatchedToThresholds'] = dict(zip([1,2,4,5],[0.5,1.0,0.5,0.5]))
    examples[2]['expected: RDN TrainIdsMatchedToThresholds'] = {1: 0.105015870403904, 2: 0.19681031408389435, 4: 0.09939414531216254, 5: 0.09314072164086293} #03/11/23: these small changes not unexpected when introduced stratified splitting for regression {1: 0.10514335445719257, 2: 0.19214342935669887, 4: 0.09703725146292935, 5: 0.09325378985308012} #31/10/23: introduced normalized stdev for regression -so replaced expected with observed.#Previously: {1: 0.10490188707497346, 2: 0.18862107868431383, 4: 0.09525837602034358, 5: 0.0930396274969894}#Switching to non-integer endpoint values reduced all thresholds - OK as regression based on non-integer values is a harder task? #{1: 0.18393972058572117, 2: 0.36787944117144233, 4: 0.18393972058572117, 5: 0.18393972058572117} #When switching from StratifiedKFold toKFold, the thresholds were updated. #When switched to regression, this was the observed outcome #dict(zip([1,2,4,5],[examples[2]['expected: dk_NN_TrainIdsMatchedToThresholds'][id_]*w_i_dict[id_] for id_ in [1,2,4,5]]))
    examples[2]['expected: RDN test_id_ad_status_dict'] = defaultdict(dict)
    examples[2]['expected: RDN test_id_ad_status_dict'][10]['InsideAD']=False
    examples[2]['expected: RDN test_id_ad_status_dict'][20]['InsideAD']=True 
    examples[2]['expected: RDN test_id_ad_status_dict'][30]['InsideAD']=False 
    examples[2]['expected: RDN test_id_ad_status_dict'][50]['InsideAD']=False
    #*******************************
    for example_no in examples.keys():
        print('Running check_RDN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations_regression: example=%d' % example_no)
        RDN_thresholds_instance = RDN_thresholds(train_df=examples[example_no]['train_df'],id_col=examples[example_no]['id_col'],k=examples[example_no]['k'],distance_metric=examples[example_no]['distance_metric'],scale=examples[example_no]['scale'],endpoint_col=examples[example_no]['endpoint_col'],expected_descs_no=1,only_consider_k_neighbours_for_final_threshold_calculations=True)
        
        RDN_thresholds_instance.assignRdnTrainingSetSpecificThresholds(id_col=examples[example_no]['id_col'],endpoint_col=examples[example_no]['endpoint_col'],n_models_in_RDN_ensemble=examples[example_no]['n_models_in_RDN_ensemble'],cv_folds_no=examples[example_no]['cv_folds_no'],SEED=examples[example_no]['SEED'],use_k_fold_cv_for_ensemble_preds=examples[example_no]['use_k_fold_cv_for_ensemble_preds'],use_sklearn_rf=examples[example_no]['use_sklearn_rf'],round_dt_preds=examples[example_no]['round_dt_preds'],write_out_intermediate_res_for_checks=write_out_intermediate_res_for_checks,regression=True)
        
        if not examples[example_no]['expected: dk_NN_TrainIdsMatchedToThresholds'] == RDN_thresholds_instance.dk_NN_TrainIdsMatchedToThresholds: raise Exception("examples[example_no]['expected: dk_NN_TrainIdsMatchedToThresholds'] =%s, RDN_thresholds_instance.dk_NN_TrainIdsMatchedToThresholds=%s" % (str(examples[example_no]['expected: dk_NN_TrainIdsMatchedToThresholds']),str(RDN_thresholds_instance.dk_NN_TrainIdsMatchedToThresholds)))
        
        
        if not list(examples[example_no]['expected: RDN TrainIdsMatchedToThresholds'].keys()) == list(RDN_thresholds_instance.TrainIdsMatchedToThresholds.keys()): raise Exception("examples[example_no]['expected: RDN TrainIdsMatchedToThresholds'] =%s, RDN_thresholds_instance.TrainIdsMatchedToThresholds=%s" % (str(examples[example_no]['expected: RDN TrainIdsMatchedToThresholds']),str(RDN_thresholds_instance.TrainIdsMatchedToThresholds)))
        
        for id_ in examples[example_no]['expected: RDN TrainIdsMatchedToThresholds'].keys():
            if not round(examples[example_no]['expected: RDN TrainIdsMatchedToThresholds'][id_],3)==round(RDN_thresholds_instance.TrainIdsMatchedToThresholds[id_],3):raise Exception("examples[example_no]['expected: RDN TrainIdsMatchedToThresholds'] =%s, RDN_thresholds_instance.TrainIdsMatchedToThresholds=%s" % (str(examples[example_no]['expected: RDN TrainIdsMatchedToThresholds']),str(RDN_thresholds_instance.TrainIdsMatchedToThresholds)))
        
        test_id_ad_status_dict=RDN_thresholds_instance.getADstatusOfTestSetCompounds(test_compounds_X_df=examples[example_no]['test_compounds_X_df'],id_col=examples[example_no]['id_col'])
        
        if not examples[example_no]['expected: RDN test_id_ad_status_dict'] == test_id_ad_status_dict: raise Exception("examples[example_no]['expected: RDN test_id_ad_status_dict'] =%s, test_id_ad_status_dict=%s" % (str(examples[example_no]['expected: RDN test_id_ad_status_dict']),str(test_id_ad_status_dict)))
        
        print('RAN check_RDN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations_regression: example=%d' % example_no)
    #********************************

#check_RDN_thresholds_only_consider_k_neighbours_for_final_threshold_calculations_regression()
