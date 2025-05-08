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
####################
#Copyright (c) 2022-2023 Syngenta
# Contact: richard.marchese_robinson [at] syngenta.com
###################
import os,sys,re
from collections import defaultdict
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import _safe_indexing
from sklearn.model_selection import train_test_split
import warnings
import time
#--------------------
from .basic_utils import findDups
from .stratified_continuous_split import scsplit
from ..utils.time_utils import basic_time_task
from ..utils.basic_utils import report_name_of_function_where_this_is_called

def convertClassLabel(raw_class_label,class_1):
    if raw_class_label == class_1:
        return 1
    else:
        return 0

def predictedBinaryClassFromProbClass1(prob_class_1,thresh=0.5):
    if prob_class_1 > thresh:
        return 1
    else:
        return 0

def getXandY(df,descriptor_names,endpoint_name):
    
    #=========================
    if not type(pd.DataFrame({'x':[1.5]}))==type(df): raise Exception('df is not a pandas dataframe, but a {}'.format(type(df)))
    #========================
    
    x = df[descriptor_names]
    y = df[endpoint_name]
    
    return x,y

def getTestYFromTestIds(dataset_ids,dataset_y,test_ids):
    #==========================
    assert isinstance(dataset_ids,list),type(dataset_ids)
    assert isinstance(test_ids,list),type(test_ids)
    assert isinstance(dataset_y,(pd.Series,np.ndarray))
    #Checked in Python interpreter for this situation
    assert dataset_y.shape[0] == len(dataset_ids),"dataset_y.shape[0] = {}, len(dataset_ids) = {}".format(dataset_y.shape[0],len(dataset_ids))
    assert 0 == len([id_ for id_ in test_ids if not id_ in dataset_ids]),"Some test IDs are not in dataset_ids: {}".format([id_ for id_ in test_ids if not id_ in dataset_ids])
    #==========================
    
    if isinstance(dataset_y,pd.Series):
        dataset_y_reindexed = dataset_y.reset_index(drop=True) #For a Pandas Series, subsetting based upon list indices will only work if dataset_y indices start from 0 and no indices are skipped!
    else:
        dataset_y_reindexed = dataset_y
    
    test_indices = [dataset_ids.index(id_) for id_ in test_ids]
    
    test_y = dataset_y_reindexed[test_indices]
    
    if isinstance(dataset_y,pd.Series):
        test_y = test_y.reset_index(drop=True)
    
    return test_y

def get_consistent_n_random_seeds(original_seed,n):
    np.random.seed(original_seed)
    return np.random.randint(low=0,high=10**6,size=n).tolist()

def singleRandomSplit(data_x,data_y,test_fraction,random_state=42,stratified=False,reset_indices=True,data_ids=None,regression=True,check_no_mols_dropped=True,check_split_indices=True):
    #==================
    assert isinstance(data_x,pd.DataFrame),f'type(data_x)={type(data_x)}'
    assert isinstance(data_y,pd.Series),f'type(data_y)={type(data_y)}'
    if not data_ids is None:
        assert isinstance(data_ids,list),f'type(data_ids)={type(data_ids)}'
        assert stratified,'Splitting data_ids is only currently supported for stratified random splitting!'
    #==================

    #########
    #scsplit(...) will not allow these arguments to be passed in via keywords!
    X=data_x
    y=data_y
    #########

    if not stratified:
       train_x, test_or_calib_x, train_y, test_or_calib_y = train_test_split(X, y, test_size=test_fraction, random_state=random_state)
    else:
        train_x, test_or_calib_x, train_y, test_or_calib_y = scsplit(X, y, stratify = data_y,test_size = test_fraction, random_state = random_state, continuous=regression)
    
    #--------------------------
    if check_no_mols_dropped:
        assert isinstance(train_x,pd.DataFrame)
        assert isinstance(test_or_calib_x,pd.DataFrame)
        assert isinstance(train_y,pd.Series)
        assert isinstance(test_or_calib_y,pd.Series)
        assert train_x.shape[0] == train_y.shape[0]
        assert test_or_calib_x.shape[0] == test_or_calib_y.shape[0]

        assert (train_x.shape[0]+test_or_calib_x.shape[0]) == data_x.shape[0],f'train_x.shape[0]={train_x.shape[0]},test_or_calib_x.shape[0]={test_or_calib_x.shape[0]}, data_x.shape[0]={data_x.shape[0]}'
        assert (train_x.shape[0]+test_or_calib_x.shape[0]) == data_y.shape[0]
    #--------------------------
    #--------------------------
    if check_split_indices:
        print('Also checking split indices ....')
        assert 0 == len(set(train_x.index.tolist()).intersection(set(test_or_calib_x.index.tolist())))
        assert 0 == len(set(train_y.index.tolist()).intersection(set(test_or_calib_y.index.tolist())))
    #--------------------------
    if reset_indices:
        print('Resetting split indices')
        train_x.reset_index(drop=True, inplace=True)
        train_y.reset_index(drop=True, inplace=True)
        test_or_calib_x.reset_index(drop=True, inplace=True)
        test_or_calib_y.reset_index(drop=True, inplace=True)
    #--------------------------

    return train_x,train_y,test_or_calib_x,test_or_calib_y

def get_multiple_random_splits(no_splits,data_x,data_y,test_fraction,random_state=42,stratified=False,reset_indices=True):
    
    dict_of_splits = defaultdict(dict)
    
    split_rand_seeds = get_consistent_n_random_seeds(original_seed=random_state,n=no_splits)
    
    for split in range(no_splits):
        split_seed = split_rand_seeds[split]

        train_x,train_y,test_or_calib_x,test_or_calib_y = singleRandomSplit(data_x,data_y,test_fraction,split_seed,stratified,reset_indices)
        
        dict_of_splits[split]['train_x'] = train_x
        dict_of_splits[split]['train_y'] = train_y
        dict_of_splits[split]['test_or_calib_x'] = test_or_calib_x
        dict_of_splits[split]['test_or_calib_y'] = test_or_calib_y
    
    return dict_of_splits

def get_k_folds_with_stratified_sampling(df,y_col,id_col,K,rand_state,regression,check_no_mols_dropped=True):
    assert isinstance(df,pd.DataFrame)

    folds_dict = {}

    remainder = df

    for fold_no in range(K):
        
        if fold_no < (K-1):
            remainder, fold = scsplit(remainder, stratify=remainder[y_col], test_size = 1/(K-fold_no),random_state=rand_state,continuous=regression)
            #===============================
            if check_no_mols_dropped:
                print('Checking no molecules dropped inside get_k_folds_with_stratified_sampling(...)')
                assert isinstance(remainder,pd.DataFrame)
                assert isinstance(fold,pd.DataFrame)
                assert (remainder.shape[0]+fold.shape[0]) == df.shape[0]
            #===============================
        else:
            fold = remainder

        folds_dict[fold_no] = fold
    
    return folds_dict

def get_per_fold_train_test_subsets(folds_dict):

    fold2TrainAndTestDict = defaultdict(dict)

    for fold_no in folds_dict.keys():
        test_df = folds_dict[fold_no]

        other_folds = [folds_dict[fn] for fn in folds_dict.keys() if not fold_no == fn]

        train_df = pd.concat(other_folds)

        fold2TrainAndTestDict[fold_no]['cv_train'] = train_df

        fold2TrainAndTestDict[fold_no]['cv_test'] = test_df



    return fold2TrainAndTestDict

def reset_all_train_test_indices(fold2TrainAndTestDict):

    for fold_no in fold2TrainAndTestDict.keys():
        for subset_label in ['cv_train','cv_test']:
            fold2TrainAndTestDict[fold_no][subset_label].reset_index(drop=True, inplace=True)

    return fold2TrainAndTestDict

def k_fold_cv_with_stratified_sampling_for_continuous_or_categorical_y(df,y_col,id_col,K,rand_state,reset_indices=True,regression=True):
    assert isinstance(df,pd.DataFrame)

    folds_dict = get_k_folds_with_stratified_sampling(df,y_col,id_col,K,rand_state,regression)

    fold2TrainAndTestDict = get_per_fold_train_test_subsets(folds_dict)

    if reset_indices:
        fold2TrainAndTestDict = reset_all_train_test_indices(fold2TrainAndTestDict)

    return fold2TrainAndTestDict




def makeYLabelsNumeric(test_y, class_1="Active",class_0="Inactive",return_as_pandas_series=False):
    #if not "Active" == class_1: raise Exception('This whole function only makese sense if class_1=Active!')

    # experi_class_labels = test_y.tolist() #Trying to change an iterable during iteration might cause problems? #Actually, no, this worked fine for a toy list of strings!
    experi_class_labels_pre_ordering = []
    ############################################
    # i = 0
    for label in test_y.tolist():
        if label == class_0:
            experi_class_labels_pre_ordering.append(0)
        elif class_1 == label:
            experi_class_labels_pre_ordering.append(1)
        else:
            raise Exception("Unrecognised label in test_y={}.".format(label))
        # i += 1
    ############################################

    if return_as_pandas_series:
        experi_class_labels_pre_ordering = pd.Series(experi_class_labels_pre_ordering)

    return experi_class_labels_pre_ordering

def makeIDsNumeric(df,id_col,tmp_id_col='Old.ID'): #A limitation of current AD code is that the IDs must be numeric for some methods!
    
    df = df.rename({id_col:tmp_id_col},axis=1)
    
    df.insert(1,id_col,range(df.shape[0]))
    
    df = df.drop(tmp_id_col,axis=1)
    
    return df

def checkTrainTestHaveUniqueDistinctIDs(train_df,test_df,id_col):
    train_ids = train_df[id_col].tolist()
    
    test_ids = test_df[id_col].tolist()
    
    all_ids = train_ids+test_ids
    
    assert len(all_ids)==len(set(all_ids)),"Duplicate IDs: {}".format(findDups(all_ids))

def prepareInputsForModellingAndUncertainty(X_train_and_ids_df,train_y,X_test_and_ids_df,id_col):
    #----------------------
    assert isinstance(X_train_and_ids_df,pd.DataFrame),f'type(X_train_and_ids_df)={type(X_train_and_ids_df)}'
    assert isinstance(X_test_and_ids_df,pd.DataFrame),f'type(X_test_and_ids_df)={type(X_test_and_ids_df)}'
    #---------------------

    train_inc_calib_x = X_train_and_ids_df.drop([id_col],axis=1)
    
    train_inc_calib_y = train_y
    
    test_x = X_test_and_ids_df.drop([id_col],axis=1)
    
    return train_inc_calib_x,train_inc_calib_y,test_x

def checkEndpointLists(all_eps,exemplar_eps):
    assert len(all_eps)==len(set(all_eps)),f"Duplicates={findDups(all_eps)}"
    
    assert len(set(exemplar_eps).intersection(set(all_eps)))==len(exemplar_eps),f'exemplar ep not in all={[ep for ep in exemplar_eps if not ep in all_eps]}, duplicate exemplar eps = {findDups(exemplar_eps)}'


def get_predictions_of_bagged_ensemble_tree_from_full_features_test_instance(tree,ensemble,tree_index,test_X,expected_type_of_test_X=pd.DataFrame,want_class_probs=False,ignore_user_warning=True,monitor_time=False):
    ##########################
    #This is designed to avoid possible problems which may be associated with applying the predict or predict_proba functions directly, as reflected in the following warning message [SciKit-Learn version 1.0.2]:  " UserWarning: X has feature names, but DecisionTreeRegressor was fitted without feature names"
    #This is based upon the following documented issue and suggested work-arounds:
    #https://github.com/scikit-learn/scikit-learn/issues/21599
    #Maybe this issue is or will be fixed in future versions of SciKit-Learn?
    ##########################
    #########################
    #For now, just filter this warning by default to stop log files becoming clogged up:
    if ignore_user_warning:
        warnings.filterwarnings("ignore", category=UserWarning)
    ##########################
    
    if monitor_time:
        start = time.time()

    if not tree == ensemble.estimators_[tree_index]: raise Exception('Inconsistent tree and tree_index supplied!')
    
    if not want_class_probs:
        #preds =  tree.predict(_safe_indexing(test_X, ensemble.estimators_features_[tree_index], axis=1)) #[SciKit-Learn version 1.0.2]: This fails with the following error: "AttributeError: 'RandomForestRegressor' object has no attribute 'estimators_features_'"
        preds = tree.predict(test_X).tolist()
    else:
        #preds = tree.predict_proba(_safe_indexing(test_X, ensemble.estimators_features_[tree_index], axis=1)) #SciKit-Learn version 1.0.2]: This fails with the following error: "AttributeError: 'RandomForestClassifier' object has no attribute 'estimators_features_'"
        preds = tree.predict_proba(test_X)
    
    if monitor_time:
        end = time.time()

        task = report_name_of_function_where_this_is_called()

        basic_time_task(task,end,start,units='seconds')
        del task,end,start

    
    return preds

def compute_pred_intervals(y_pred,half_pred_interval_sizes):
    if isinstance(y_pred,np.ndarray):
        _y_pred = y_pred.tolist()
    else:
        assert isinstance(y_pred,list),type(y_pred)
        _y_pred = y_pred
    
    if isinstance(half_pred_interval_sizes,np.ndarray):
        _half_pred_interval_sizes = half_pred_interval_sizes.tolist()
    else:
        assert isinstance(half_pred_interval_sizes,list),type(half_pred_interval_sizes)
        _half_pred_interval_sizes = half_pred_interval_sizes

    assert len(_half_pred_interval_sizes)==len(_y_pred),f'len(_half_pred_interval_sizes)={len(_half_pred_interval_sizes)},len(_y_pred)={len(_y_pred)}'

    intervals = np.column_stack((_y_pred,_y_pred))

    intervals[:,0] -= _half_pred_interval_sizes
    intervals[:,1] += _half_pred_interval_sizes

    return intervals

def getKFoldCVTrainOtherXY(data_x,data_y,other_name,no_folds,rand_seed):
    
    #==========================
    if not data_x.shape[0] == data_y.shape[0]: raise Exception('data_x.shape[0]=%d,data_y.shape[0]=%d' % (data_x.shape[0],data_y.shape[0]))
    if not data_x.shape[0] > (2*no_folds): raise Exception('data_x.shape[0]=%d,no_folds=%d' % (data_x.shape[0],no_folds))
    #==========================
    
    fold2TrainOtherXY = {}#defaultdict(dict)
    
    splitter = StratifiedKFold(n_splits=no_folds,shuffle=True,random_state=rand_seed)
    
    fold_count = 0
    
    for train_indices,other_indices in splitter.split(data_x,data_y):
        fold_count += 1
        
        fold2TrainOtherXY[fold_count] = {}
        
        fold2TrainOtherXY[fold_count]['Train_x'] = data_x.iloc[train_indices,:]
        fold2TrainOtherXY[fold_count]['%s_x' % other_name] = data_x.iloc[other_indices,:]
        fold2TrainOtherXY[fold_count]['Train_y'] = data_y.iloc[train_indices]
        fold2TrainOtherXY[fold_count]['%s_y' % other_name] = data_y.iloc[other_indices]
        
        #=======================================
        if not fold2TrainOtherXY[fold_count]['Train_x'].shape[0] == fold2TrainOtherXY[fold_count]['Train_y'].shape[0]: raise Exception("fold2TrainOtherXY[fold_count]['Train_x'].shape[0] =%d, fold2TrainOtherXY[fold_count]['Train_y'].shape[0] =%d" % (fold2TrainOtherXY[fold_count]['Train_x'].shape[0],fold2TrainOtherXY[fold_count]['Train_y'].shape[0]))
        if not fold2TrainOtherXY[fold_count]['%s_x' % other_name].shape[0] == fold2TrainOtherXY[fold_count]['%s_y' % other_name].shape[0]: raise Exception("fold2TrainOtherXY[fold_count]['%s_x'].shape[0] =%d, fold2TrainOtherXY[fold_count]['%s_y'].shape[0] =%d" % (fold2TrainOtherXY[fold_count]['%s_x' % other_name].shape[0],fold2TrainOtherXY[fold_count]['%s_y' % other_name].shape[0]))
        if not (fold2TrainOtherXY[fold_count]['Train_x'].shape[0]+fold2TrainOtherXY[fold_count]['%s_x' % other_name].shape[0])==data_x.shape[0]: raise Exception('sum=%d,data_x.shape[0]=%d' % ((fold2TrainOtherXY[fold_count]['Train_x'].shape[0]+fold2TrainOtherXY[fold_count]['%s_x' % other_name].shape[0]),data_x.shape[0]))
        #=======================================
    
    if not fold_count == no_folds: raise Exception('fold_count =%d no_folds =%d' % (fold_count,no_folds))
    
    return fold2TrainOtherXY

def probOK(p):
    if type(0.5) == type(p) and p >=0 and p<=1:
        return True
    else:
        return False

def getXandYFromDataset(dataset_df,id_col,y_col):
    assert isinstance(dataset_df,pd.DataFrame)
        
    dataset_x = dataset_df.drop(labels=[id_col,y_col],axis=1)
    
    dataset_y = dataset_df[y_col]#.tolist()
    
    return dataset_x,dataset_y

def unexpectedScore(score,ml_alg,always_normalized):
    if 'RandomForestClassifier' == ml_alg or always_normalized:
        if not (score >= 0 and score <= 1 and type(0.5)==type(score)):
            return True
    else:
        raise Exception('The following ML algorithm is not currently supported: %s' % ml_alg)
    
    return False

def getClass1ProbsRFModel(model,data_x,class_1=1): 
    array_of_scores_per_class = model.predict_proba(data_x)
    
    list_of_all_classes_in_order_considered_by_model = model.classes_.tolist()
    
    if not len(list_of_all_classes_in_order_considered_by_model) == len(set(list_of_all_classes_in_order_considered_by_model)): raise Exception('Not unique classes?! - %s' % str(list_of_all_classes_in_order_considered_by_model))
    
    class_1_index = list_of_all_classes_in_order_considered_by_model.index(class_1)
    
    return [float(v) for v in array_of_scores_per_class[:,class_1_index].tolist()]

def isRFClass1ScoreConsistent(score,pred):
    
    if not probOK(score):
        return False
    
    if not predictedBinaryClassFromProbClass1(prob_class_1=score) == pred:
        return False
    
    return True

def getScoresForClass1(model,data_x,ml_alg,always_normalized=True):
    
    if 'RandomForestClassifier' == ml_alg:
        scores_for_class_1 = getClass1ProbsRFModel(model,data_x)
        
        #=========================
        #Checking:
        predicted_classes = model.predict(data_x)
        
        assert all([isRFClass1ScoreConsistent(score=scores_for_class_1[i],pred=predicted_classes[i]) for i in range(0,data_x.shape[0])])
        #=========================
    
    else:
        raise Exception('The following ML algorithm is not currently supported: %s' % ml_alg)
    
    #===========================
    #Checking
    
    if not len(scores_for_class_1) == data_x.shape[0]: raise Exception('len(scores_for_class_1) =%d, data_x.shape[0]=%d' % (len(scores_for_class_1),data_x.shape[0]))
    
    unexpected_scores = [score for score in scores_for_class_1 if unexpectedScore(score,ml_alg,always_normalized)]
    
    if not 0 == len(unexpected_scores): raise Exception('ml_alg=%s, unexpected_scores = %s' % (ml_alg,str(unexpected_scores))) 
    #============================
    
    return scores_for_class_1
