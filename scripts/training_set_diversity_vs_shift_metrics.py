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
############################################################
import os,sys,re,glob
import pandas as pd
import numpy as np
import statistics
import itertools
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
#======================================================
#07/11/25: support debugging using logging
#https://docs.python.org/3/howto/logging.html
#https://docs.python.org/3/library/logging.html#logging.Logger.setLevel
#https://docs.python.org/3/library/logging.html#logging-levels
import logging
from logging import DEBUG,INFO
#===========================================================


this_dir = os.path.dirname(os.path.abspath(__file__))
top_code_dir = os.path.dirname(this_dir)
sys.path.append(top_code_dir)
#-
from scripts.summary_plots_script_functions import get_shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals
from scripts.summary_plots_script_functions import make_label_ready_for_file_name
from scripts.summary_plots_script_functions import get_consistent_metric_limits
from summary_plots_global_min_max import globally_observed_min_max_metric_and_shifts_dict
from scripts.consistent_parameters_for_all_modelling_runs import endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col
from scripts.consistent_parameters_for_all_modelling_runs import classifcation_metric_to_expected_sub_1_minus_sub_2_sign,regression_metric_to_expected_sub_1_minus_sub_2_sign
from scripts.recommended_defaults import ep_type_matched_to_default_AD_uncertainty_methods
from scripts.PublicDataModelling.general_purpose.common_globals import regression_dataset_names as public_regression_dataset_names
from scripts.PublicDataModelling.general_purpose.common_globals import ds_matched_to_ep_list as public_ds_matched_to_ep_list
#If we try to use relative imports, this causes problems with imports inside the script we are importing from
sys.path.append(os.path.sep.join([top_code_dir,'scripts','PublicDataModelling','general_purpose']))
from summarize_dataset_pairwise_distance_distributions import get_fold
#If we try to use relative imports, this causes problems with imports inside the script we are importing from
sys.path.append(os.path.sep.join([top_code_dir,'scripts','SynExamples']))
from summary_plots_for_syn_updates import regression_dataset_names as syn_regression_dataset_names
from summary_plots_for_syn_updates import classification_dataset_names as syn_classification_dataset_names
from summary_plots_for_syn_updates import reg_test_names_in_order as syn_reg_test_names_in_order
from summary_plots_for_syn_updates import class_test_names_in_order as syn_class_test_names_in_order


#-
from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict
#-

dir_containing_all_res = os.path.dirname(os.path.dirname(top_code_dir))

syn_reg_stats_csv = os.path.sep.join([dir_containing_all_res,'SyngentaData','Merged_Stats','SYN_logP_Regression_Stats_EndpointTestNamesSwapped.csv'])
syn_class_stats_csv = os.path.sep.join([dir_containing_all_res,'SyngentaData','Merged_Stats','SYN_DT50_Classification_Stats_EndpointTestNamesSwapped.csv'])
public_reg_stats_csv = os.path.sep.join([dir_containing_all_res,'PublicData','AllMergedStats','All_Endpoints_Regression_Stats.csv'])
public_class_stats_csv = os.path.sep.join([dir_containing_all_res,'PublicData','AllMergedStats','All_Endpoints_Classification_Stats.csv'])

all_metrics_csv_files = [syn_reg_stats_csv,syn_class_stats_csv,public_reg_stats_csv,public_class_stats_csv]

#---------------------------
classification_stats = list(classifcation_metric_to_expected_sub_1_minus_sub_2_sign.keys())
regression_stats = list(regression_metric_to_expected_sub_1_minus_sub_2_sign.keys())
#----------------------------

syn_dataset_matched_to_endpoints_or_timepoints_for_syn_timesplits = {syn_regression_dataset_names[0]:syn_reg_test_names_in_order,syn_classification_dataset_names[0]:syn_class_test_names_in_order}

def get_df_with_extra_metadata(dist_stats_csv,dataset):

    if dataset in ['logP','DT50']:
        endpoint = dataset
        train_set_label = os.path.basename(dist_stats_csv).split('_')[1].split('_')[0]
        test_set_label = train_set_label
        fold = 'N/A'
    elif dataset in ['Morger_ChEMBL','Morger_Tox21','Wang_ChEMBL']:
        endpoint = os.path.basename(dist_stats_csv).split('ep=')[1].split('_')[0]
        raw_train_set_label =  os.path.basename(dist_stats_csv).split('tr=')[1].split('_vs')[0].replace(endpoint,'')
        raw_test_set_label = os.path.basename(dist_stats_csv).split('_vs_te=')[1].split('_FPs_fract_stats.csv')[0].replace(f'{endpoint}_','').replace('_test','')

        train_test_label_pair = (raw_train_set_label, raw_test_set_label)

        fold = get_fold(train_test_label_pair=train_test_label_pair,dataset_name=dataset,merged_across_folds=False)

        train_set_label = re.sub('(_f=[0-9+])','',raw_train_set_label)
        test_set_label = re.sub('(_f=[0-9+])','',raw_test_set_label)
    
    else:
        raise Exception(f'Unexpected dataset name={dataset}')

    df = pd.read_csv(dist_stats_csv)

    #======================================
    #Train-vs-Test mean distance doesn't really tell you much. 
    #You could have a diverse training set and a less diverse test set from a gap within that chemical space and the mean train-test distance could be smaller than the training-training distance even though they were separated in chemical space.
    #The t-SNE plot is arguably more informative.
    df.drop(['Train-vs-Test'],axis=1,inplace=True)
    #=========================================
      

    df.insert(df.shape[1],'Endpoint',[endpoint]*df.shape[0],allow_duplicates=False)
    df.insert(df.shape[1],'Train-Name',[train_set_label]*df.shape[0],allow_duplicates=False)
    df.insert(df.shape[1],'Test-Name',[test_set_label]*df.shape[0],allow_duplicates=False)
    df.insert(df.shape[1],'fold',[fold]*df.shape[0],allow_duplicates=False)

    return df

def get_merged_training_set_diversity_stats_df(dataset,diversity_stats_dir):

    logging.info(f'Merging dataset diversity statistics for dataset={dataset},diversity_stats_dir={diversity_stats_dir}')
    
    all_dfs_to_merge = [get_df_with_extra_metadata(dist_stats_csv,dataset) for dist_stats_csv in glob.glob(os.path.sep.join([diversity_stats_dir,'*_fract_stats.csv']))]


    merged_training_set_diversity_stats_df = pd.concat(all_dfs_to_merge)
    
    merged_training_set_diversity_stats_df.reset_index(drop=True,inplace=True)

    return merged_training_set_diversity_stats_df

def populate_unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance(row,unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance):
    
    

    endpoint = row['Endpoint']

    if endpoint in ['logP','DT50']:
        endpoint_or_timesplit = row['Test-Name']
        test_set_name = endpoint
    else:
        endpoint_or_timesplit = endpoint
        test_set_name = row['Test-Name']
    
    #We also need to handle the fact that there could be different values matched to different folds!
    unique_test_set_type_name_precursor = f"{test_set_name}_FOLD={row['fold']}"
    
    mean_train_vs_train_dist_val = row['Train-vs-Train']

    assert isinstance(mean_train_vs_train_dist_val,float),type(mean_train_vs_train_dist_val)


    unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance[unique_test_set_type_name_precursor][endpoint_or_timesplit] = mean_train_vs_train_dist_val



def get_unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance(merged_training_set_diversity_stats_df):

    mean_vals_only_merged_training_set_diversity_stats_df = merged_training_set_diversity_stats_df[merged_training_set_diversity_stats_df['Name of Statistic'].isin(['mean'])]

    logging.debug(f'mean_vals_only_merged_training_set_diversity_stats_df.head()={mean_vals_only_merged_training_set_diversity_stats_df.head()}')

    unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance = neverEndingDefaultDict()

    mean_vals_only_merged_training_set_diversity_stats_df.apply(populate_unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance,axis=1,args=(unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance,))

    return unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance

def get_all_unique_endpoint_or_timesplit_vals(unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance):

    all_unique_endpoint_or_timesplit_vals = list(set(list(itertools.chain(*[[endpoint_or_timesplit for endpoint_or_timesplit in unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance[unique_test_set_type_name_precursor].keys()] for unique_test_set_type_name_precursor in unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance.keys()]))))

    return all_unique_endpoint_or_timesplit_vals

def average_training_set_diversity_over_folds(unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance):

    all_unique_endpoint_or_timesplit_vals = get_all_unique_endpoint_or_timesplit_vals(unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance)

    unique_test_set_type_name_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance = neverEndingDefaultDict()

    for endpoint_or_timesplit in all_unique_endpoint_or_timesplit_vals:
        unique_test_type_name_matched_to_list_of_mean_training_set_distances = defaultdict(list)
        for unique_test_set_type_name_precursor in unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance.keys():
            mean_training_set_distance = unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance[unique_test_set_type_name_precursor][endpoint_or_timesplit]

            #==================
            if not isinstance(mean_training_set_distance,float):
                logging.info(f'WARNING: mean_training_set_distance={mean_training_set_distance} does not appear to be valid, presumably because endpoint_or_timesplit={endpoint_or_timesplit} and unique_test_set_type_name_precursor={unique_test_set_type_name_precursor} are not compatible!')
                continue
            #===================

            unique_test_set_type_name = unique_test_set_type_name_precursor.split('_FOLD=')[0]

            unique_test_type_name_matched_to_list_of_mean_training_set_distances[unique_test_set_type_name].append(mean_training_set_distance)

        for unique_test_set_type_name in unique_test_type_name_matched_to_list_of_mean_training_set_distances.keys():
            values_to_average = unique_test_type_name_matched_to_list_of_mean_training_set_distances[unique_test_set_type_name]

            assert isinstance(values_to_average,list)

            values_to_average_msg = f'unique_test_set_type_name={unique_test_set_type_name},endpoint_or_timesplit={endpoint_or_timesplit}: averaging these mean training set distances across folds={[round(v,2) for v in values_to_average]},range={round(min(values_to_average),2)}-{round(max(values_to_average),2)}'

            assert all([isinstance(v,float) for v in values_to_average]),values_to_average_msg

            if 5 == len(values_to_average):
                logging.info(values_to_average_msg)
            elif 1 == len(values_to_average):
                pass
            else:
                raise Exception(values_to_average_msg)


            fold_mean_of_mean_training_set_distance = statistics.mean(np.asarray(values_to_average,dtype=np.float32))

            unique_test_set_type_name_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance[unique_test_set_type_name][endpoint_or_timesplit] = fold_mean_of_mean_training_set_distance


    return unique_test_set_type_name_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance

def get_mean_training_set_distance_paired_with_all_shift_metric_values_df(metric,unique_test_set_type_name_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance,metric_names_matched_to_test_set_type_to_endpoint_or_timesplit_names_to_all_fold_seed_shift_metric_vals):

    #================================
    logging.debug(f'unique_test_set_type_name_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance={unique_test_set_type_name_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance}')
    logging.debug(f'metric_names_matched_to_test_set_type_to_endpoint_or_timesplit_names_to_all_fold_seed_shift_metric_vals={metric_names_matched_to_test_set_type_to_endpoint_or_timesplit_names_to_all_fold_seed_shift_metric_vals}')

    #================================
    
    dict_of_mean_training_set_distance_paired_with_all_shift_metric_values = defaultdict(list)

    for unique_test_set_type_name in metric_names_matched_to_test_set_type_to_endpoint_or_timesplit_names_to_all_fold_seed_shift_metric_vals[metric].keys():
        
        #==========================================
        #Skip results obtained on less realistic randomly sampled test sets:
        if 'train' in unique_test_set_type_name.lower() or 'ivit' in unique_test_set_type_name.lower():
            logging.info(f'SKIPPING results for this randomly sampled unique_test_set_type_name={unique_test_set_type_name}')
            continue
        else:
            logging.info(f'Not skipping results for this non-randomly sampled unique_test_set_type_name={unique_test_set_type_name}')
        #==========================================

        for endpoint_or_timesplit_name in metric_names_matched_to_test_set_type_to_endpoint_or_timesplit_names_to_all_fold_seed_shift_metric_vals[metric][unique_test_set_type_name].keys():

            all_shift_metric_vals = metric_names_matched_to_test_set_type_to_endpoint_or_timesplit_names_to_all_fold_seed_shift_metric_vals[metric][unique_test_set_type_name][endpoint_or_timesplit_name]['Inside - Outside']

            assert isinstance(all_shift_metric_vals,list)
            assert all([isinstance(v,float) for v in all_shift_metric_vals])
            
            mean_training_set_distance = unique_test_set_type_name_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance[unique_test_set_type_name][endpoint_or_timesplit_name]

            assert isinstance(mean_training_set_distance,np.float32),f'unique_test_set_type_name={unique_test_set_type_name},endpoint_or_timesplit_name={endpoint_or_timesplit_name},mean_training_set_distance={mean_training_set_distance},type={type(mean_training_set_distance)}'
            
            for shift_metric_val in all_shift_metric_vals:

                dict_of_mean_training_set_distance_paired_with_all_shift_metric_values['Mean training set distance (averaged over folds where applicable)'].append(mean_training_set_distance)
                dict_of_mean_training_set_distance_paired_with_all_shift_metric_values['Shift-metric value'].append(shift_metric_val)

    mean_training_set_distance_paired_with_all_shift_metric_values_df = pd.DataFrame(dict_of_mean_training_set_distance_paired_with_all_shift_metric_values)

    return mean_training_set_distance_paired_with_all_shift_metric_values_df

def plot_training_set_diversity_vs_shift_metric_values(mean_training_set_distance_paired_with_all_shift_metric_values_df,plot_file_name,shift_metric_lower_limit,shift_metric_upper_limit):
    #===================
    #Set based on visual inspection of initial plots acroos all dataset groups (seem to extend beyond range of observed values):
    mean_dist_lower_lim = 0.75
    mean_dist_upper_lim = 0.95
    #===================
    mean_dist_col_label = 'Mean training set distance (averaged over folds where applicable)'
    #===================
    #Check:
    current_min_mean_dist = mean_training_set_distance_paired_with_all_shift_metric_values_df[mean_dist_col_label].min()
    current_max_mean_dist = mean_training_set_distance_paired_with_all_shift_metric_values_df[mean_dist_col_label].max()
    assert current_min_mean_dist > mean_dist_lower_lim,plot_file_name
    assert current_max_mean_dist < mean_dist_upper_lim,plot_file_name
    #===================

    sns.set_style("white")

    #https://seaborn.pydata.org/generated/seaborn.regplot.html
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    dict_of_args_to_pass_to_plt_scatter = {'s':5}
    matplot_lib_axes_obj = sns.regplot(data=mean_training_set_distance_paired_with_all_shift_metric_values_df,x=mean_dist_col_label, y='Shift-metric value',marker="x", scatter_kws=dict_of_args_to_pass_to_plt_scatter)

    matplot_lib_axes_obj.set_xlim(xmin=mean_dist_lower_lim,xmax=mean_dist_upper_lim)

    matplot_lib_axes_obj.set_xlabel('Average training set distance')

    matplot_lib_axes_obj.set_ylim(ymin=shift_metric_lower_limit,ymax=shift_metric_upper_limit)

    plt.tight_layout()

    plt.savefig(plot_file_name, transparent=True)
    plt.close('all')
    plt.clf()

def main():
    print('THE START')

    for public_datasets in [False,True]:
        
        #=========================
        if not public_datasets:
            regression_dataset_names = syn_regression_dataset_names
            dataset_matched_to_endpoints_or_timepoints_for_syn_timesplits = syn_dataset_matched_to_endpoints_or_timepoints_for_syn_timesplits
        else:
            regression_dataset_names = public_regression_dataset_names
            dataset_matched_to_endpoints_or_timepoints_for_syn_timesplits = public_ds_matched_to_ep_list
        #=========================

        for dataset in dataset_matched_to_endpoints_or_timepoints_for_syn_timesplits.keys():
            logging.info(f'dataset={dataset}')
                         
            #-------------------------------
            if dataset in regression_dataset_names:
                if public_datasets:
                    merged_raw_stats_file = public_reg_stats_csv
                else:
                    merged_raw_stats_file = syn_reg_stats_csv
            else:
                if public_datasets:
                    merged_raw_stats_file = public_class_stats_csv
                else:
                    merged_raw_stats_file = syn_class_stats_csv
            #---------------------------------
            
            #---------------------------------
            if 'logP' == dataset:
                diversity_stats_dir = os.path.sep.join([dir_containing_all_res,'SyngentaData','logP_updates','Dist'])
            elif 'DT50' == dataset:
                diversity_stats_dir = os.path.sep.join([dir_containing_all_res,'SyngentaData','DT50_Updates','Dist'])
            elif 'Morger_ChEMBL' == dataset:
                diversity_stats_dir = os.path.sep.join([dir_containing_all_res,'PublicData','Morger_ChEMBL','Distributions'])
            elif 'Morger_Tox21' == dataset:
                diversity_stats_dir = os.path.sep.join([dir_containing_all_res,'PublicData','Tox21','Distributions'])
            elif 'Wang_ChEMBL' == dataset:
                diversity_stats_dir = os.path.sep.join([dir_containing_all_res,'PublicData','Wang_ChEMBL','data','dataset','Distributions'])
            else:
                raise Exception(f'dataset={dataset}')
            #---------------------------------

            logging.info(f'os.path.basename(merged_raw_stats_file)={os.path.basename(merged_raw_stats_file)},dataset={dataset},regression_dataset_names={regression_dataset_names}')
            
            #Note: this return value name is clearer than the original defined in the following function and other scripts!
            metric_names_matched_to_test_set_type_to_endpoint_or_timesplit_names_to_all_fold_seed_shift_metric_vals = get_shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals(dataset,dataset_matched_to_endpoints_or_timepoints_for_syn_timesplits,merged_raw_stats_file,regression_dataset_names,classification_stats,regression_stats,ep_type_matched_to_default_AD_uncertainty_methods,endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col)


            #unique_test_set_type_name,endpoint names need to align with contents of metric_names_matched_to_test_set_type_to_endpoint_or_timesplit_names_to_all_fold_seed_shift_metric_vals
            merged_training_set_diversity_stats_df = get_merged_training_set_diversity_stats_df(dataset,diversity_stats_dir)

            merged_training_set_diversity_stats_df.to_csv(os.path.sep.join([diversity_stats_dir,f'{dataset}_merged_diversity_stats.csv']))

            unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance = get_unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance(merged_training_set_diversity_stats_df)

            logging.debug(f'unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance={unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance}')
            
            unique_test_set_type_name_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance = average_training_set_diversity_over_folds(unique_test_set_type_name_precursor_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance)

            for metric in metric_names_matched_to_test_set_type_to_endpoint_or_timesplit_names_to_all_fold_seed_shift_metric_vals.keys():
                plot_file_name = os.path.sep.join([diversity_stats_dir,f'{dataset}_train_diversity_vs_{make_label_ready_for_file_name(metric)}_shift_metric_values.tiff'])

                mean_training_set_distance_paired_with_all_shift_metric_values_df = get_mean_training_set_distance_paired_with_all_shift_metric_values_df(metric,unique_test_set_type_name_matched_to_endpoint_or_timesplit_matched_to_mean_training_set_distance,metric_names_matched_to_test_set_type_to_endpoint_or_timesplit_names_to_all_fold_seed_shift_metric_vals)

                logging.info(f'dataset={dataset},metric={metric}')
                logging.debug(f'mean_training_set_distance_paired_with_all_shift_metric_values_df={mean_training_set_distance_paired_with_all_shift_metric_values_df}')
                
                shift_metric_lower_limit,shift_metric_upper_limit = get_consistent_metric_limits(metric=metric,metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals=metric_names_matched_to_test_set_type_to_endpoint_or_timesplit_names_to_all_fold_seed_shift_metric_vals,shift_metrics=True,globally_observed_min_max_metric_and_shifts_dict=globally_observed_min_max_metric_and_shifts_dict)
                
                plot_training_set_diversity_vs_shift_metric_values(mean_training_set_distance_paired_with_all_shift_metric_values_df,plot_file_name,shift_metric_lower_limit,shift_metric_upper_limit)






    print('THE END')

    return 0

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    #logger.setLevel(DEBUG)
    logging.basicConfig(filename='training_set_diversity_vs_shift_metrics_debug.log', level=INFO,filemode='w',force=True)
    sys.exit(main())