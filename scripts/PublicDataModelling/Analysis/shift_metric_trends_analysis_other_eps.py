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
#Copyright (c) 2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#############################################################
import os,sys,glob,re
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import argparse
##################################
#from AD_ranking import average_over_folds_and_seed
from default_ad_sanity_checking import compute_fraction_with_expected_sign
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
#----------------------------------------------------------------------------
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict,createOrReplaceDir,doubleDefaultDictOfLists
#----------------------------------------------------------------------------
top_scripts_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
sys.path.append(top_scripts_dir)
from PublicDataModelling.general_purpose.common_globals import regression_dataset_names,ds_matched_to_ep_list,ds_matched_to_exemplar_ep_list,get_ds_matched_to_other_endpoints_of_interest
from recommended_defaults import ep_type_matched_to_default_AD_uncertainty_methods
from consistent_parameters_for_all_modelling_runs import regression_metric_to_expected_sub_1_minus_sub_2_sign,classifcation_metric_to_expected_sub_1_minus_sub_2_sign
from consistent_parameters_for_all_modelling_runs import endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col
from summary_plots_script_functions import get_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals
from summary_plots_script_functions import get_shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals
from summary_plots_script_functions import get_p_values_info_to_support_plot_annotation
from summary_plots_script_functions import one_tail_key,two_tail_key,raw_p_val_col,adjusted_p_val_col,p_vals_label_col
from summary_plots_script_functions import get_plot_input_df
#-------------
top_res_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData'])
out_dir = os.path.sep.join([top_res_dir,'Summaries_Other_EPs_Models'])
dir_with_merged_raw_stats = os.path.sep.join([top_res_dir,'OtherMergedStats'])
classification_merged_raw_stats_csv = os.path.sep.join([dir_with_merged_raw_stats,'Other_Endpoints_Classification_Stats.csv'])
regression_merged_raw_stats_csv = os.path.sep.join([dir_with_merged_raw_stats,'Other_Endpoints_Regression_Stats.csv'])



dict_of_one_tail_adjusted_p_vals_files = defaultdict(list)
dict_of_two_tail_adjusted_p_vals_files = defaultdict(list)
for modelling_type in ['Classification','Regression']:
    dict_of_one_tail_adjusted_p_vals_files[modelling_type].append(os.path.sep.join([top_res_dir,'Other_EPs_AD_P_vals',f'one_tail_{modelling_type}_PVals_GlobalAdjusted.csv']))
    dict_of_two_tail_adjusted_p_vals_files[modelling_type].append(os.path.sep.join([top_res_dir,'Other_EPs_AD_P_vals',f'{modelling_type}_PVals_GlobalAdjusted.csv']))


def filter_by_focus_endpoints(ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,list_of_endpoints_to_focus_on):

    new_ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list = defaultdict(list)

    for ds in ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list.keys():
        for ep in ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list[ds]:
            if ep in list_of_endpoints_to_focus_on:
                print(f'Considering this endpoint = {ep}')
                new_ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list[ds].append(ep)
            else:
                print(f'Skipping this endpoint={ep}')

    return new_ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list

def get_expected_sign_of_shift_metric(metric):

    if metric in regression_metric_to_expected_sub_1_minus_sub_2_sign.keys():
        expected_sign = regression_metric_to_expected_sub_1_minus_sub_2_sign[metric]
    elif metric in classifcation_metric_to_expected_sub_1_minus_sub_2_sign.keys():
        expected_sign = classifcation_metric_to_expected_sub_1_minus_sub_2_sign[metric]
    else:
        raise Exception(f'Unexpected metric={metric}!')


    return expected_sign

def compute_number_of_non_missing_metrics(sub_df,metric):
    non_blank_subset_df = sub_df[~sub_df[metric].isna()]

    values = non_blank_subset_df[metric].tolist()

    return len(values)

def compute_average_shift_metric_percentage_with_the_right_sign(average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df,val_col,metric):

    expected_sign = get_expected_sign_of_shift_metric(metric)

    fraction_with_the_right_sign_str = compute_fraction_with_expected_sign(sub_df=average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df,metric=val_col,expected_sign=expected_sign)

    fraction_with_the_right_sign_str = fraction_with_the_right_sign_str.replace('(','').replace(')','')

    ##########################
    #Debug:
    print(f'metric={metric}')
    print(f'fraction_with_the_right_sign_str={fraction_with_the_right_sign_str}')
    ##########################

    fraction_with_the_right_sign = float(fraction_with_the_right_sign_str.split('/')[0])/float(fraction_with_the_right_sign_str.split('/')[1])
    
    percentage_with_the_right_sign = round(100*fraction_with_the_right_sign,2)

    return percentage_with_the_right_sign

def compute_significant_percentage(average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df,val_col,p_vals_label_col,sig_label_of_interest):

    #################
    #c.f. filtering of NAs inside compute_fraction_with_expected_sign(....):
    rows_before_filtering = average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df.shape[0]

    sub_df = average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df[~average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df[val_col].isna()]

    rows_after_filtering = sub_df.shape[0]

    if not (rows_after_filtering==rows_after_filtering):
        print(f"WARNING: {rows_before_filtering-rows_after_filtering} rows dropped due to missing average shift-metric values!")
    #################

    sub_with_significant_p_vals_df = sub_df[sub_df[p_vals_label_col].isin([sig_label_of_interest])]

    return round(100*(sub_with_significant_p_vals_df.shape[0]/rows_after_filtering),2)

def average_over_folds_and_seed(df,val_col):
    #####################
    #From debugging:
    #raw_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df.columns.values.tolist()=['Endpoint', 'AD Subset', 'Shift-metric: Balanced Accuracy', 'Statistical Significance Scenario Label']
    ################
    ###############
    #To keep the p-values significance labels (one per endpoint), we need column specific aggregations:
    #https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html
    #############

    for col_which_should_be_const in ['AD Subset']:
        assert 1 == len(df[col_which_should_be_const].unique().tolist()),f'col_which_should_be_const={col_which_should_be_const},values={df[col_which_should_be_const].tolist()}'
        df = df.drop(labels=[col_which_should_be_const],axis=1)

    groupby_cols = ['Endpoint']

    grouped = df.groupby(groupby_cols, as_index=False)

    averaged_df = grouped.agg({val_col:'mean','Statistical Significance Scenario Label':'first'})
    return  averaged_df

def analyse_shift_metric_signs_and_p_values(dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig,shift_metric_endpoints_to_all_fold_seed_in_and_out_ad_vals,ds,metric,test_set_type,p_values_info,p_vals_label_col=p_vals_label_col,endpoint_col=endpoint_col,ad_subset_col=ad_subset_col,raw_one_tail_sig_label="*",adj_one_tail_sig_label="**",raw_two_tail_sig_label="x",adj_two_tail_sig_label="xx"):

    val_col = f'Shift-metric: {metric}'

    #=====================================
    print(f'Dataset={ds}')
    print(f'Metric={metric}')
    print(f'Test set type={test_set_type}')
    #=====================================
    
    raw_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df = get_plot_input_df(shift_metric_endpoints_to_all_fold_seed_in_and_out_ad_vals,endpoint_col,val_col,ad_subset_col,p_values_info=p_values_info,p_vals_label_col=p_vals_label_col)

    ###################
    #Debugging:
    print(f'raw_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df.columns.values.tolist()={raw_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df.columns.values.tolist()}')
    print(f'raw_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df.head()={raw_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df.head()}')
    ###################

    
    average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df = average_over_folds_and_seed(raw_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df,val_col)

    ###################
    #Debugging:
    print(f'average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df.columns.values.tolist()={average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df.columns.values.tolist()}')
    print(f'average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df.head()={average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df.head()}')
    ###################

    dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig[ds][test_set_type][metric]['no. metric values'] = compute_number_of_non_missing_metrics(sub_df=average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df,metric=val_col)

    dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig[ds][test_set_type][metric]['Average shift-metric has the right sign (%)'] = compute_average_shift_metric_percentage_with_the_right_sign(average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df,val_col,metric)

    dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig[ds][test_set_type][metric]['Raw one-tail p-value is significant [average shift-metric has right sign] (%)'] = compute_significant_percentage(average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df,val_col,p_vals_label_col,sig_label_of_interest=raw_one_tail_sig_label)
    
    dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig[ds][test_set_type][metric]['Adjusted one-tail p-value is significant [average shift-metric has right sign] (%)'] = compute_significant_percentage(average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df,val_col,p_vals_label_col,sig_label_of_interest=adj_one_tail_sig_label)

    dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig[ds][test_set_type][metric]['Raw two-tail p-value is significant [average shift-metric has wrong sign]  (%)'] = compute_significant_percentage(average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df,val_col,p_vals_label_col,sig_label_of_interest=raw_two_tail_sig_label)
    
    dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig[ds][test_set_type][metric]['Adjusted two-tail p-value is significant [average shift-metric has wrong sign] (%)'] = compute_significant_percentage(average_shift_metrics_matched_to_p_val_labels_for_average_shift_metrics_df,val_col,p_vals_label_col,sig_label_of_interest=adj_two_tail_sig_label)
    
    return dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig

def convert_results_to_df(dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig):

    res_dol = defaultdict(list)

    for ds in dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig.keys():
        for test_set_type in dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig[ds].keys():
            for metric in dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig[ds][test_set_type].keys():
                res_dol['Dataset'].append(ds)
                res_dol['Test set type'].append(test_set_type)
                res_dol['Metric'].append(metric)
                for res_type in ['no. metric values','Average shift-metric has the right sign (%)','Raw one-tail p-value is significant [average shift-metric has right sign] (%)','Adjusted one-tail p-value is significant [average shift-metric has right sign] (%)','Raw two-tail p-value is significant [average shift-metric has wrong sign]  (%)','Adjusted two-tail p-value is significant [average shift-metric has wrong sign] (%)']:
                    res_dol[res_type].append(dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig[ds][test_set_type][metric][res_type])
    
    res_df = pd.DataFrame(res_dol)


    return res_df


def main():

    print('THE START')

    ###########################
    parser = argparse.ArgumentParser()
    parser.add_argument('-e',dest='list_of_endpoints_to_focus_on',action='store',default=None)
    dict_of_cmd_line_opts = vars(parser.parse_args())
    if not dict_of_cmd_line_opts['list_of_endpoints_to_focus_on'] is None:
        list_of_endpoints_to_focus_on = dict_of_cmd_line_opts['list_of_endpoints_to_focus_on'].split(".")
        print(f'list_of_endpoints_to_focus_on={list_of_endpoints_to_focus_on}')
    else:
        list_of_endpoints_to_focus_on = None
    ###########################

    createOrReplaceDir(out_dir)

    ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list = get_ds_matched_to_other_endpoints_of_interest(ds_matched_to_ep_list,ds_matched_to_exemplar_ep_list)

    if not list_of_endpoints_to_focus_on is None:
        ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list = filter_by_focus_endpoints(ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,list_of_endpoints_to_focus_on)

    dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig = neverEndingDefaultDict()

    for dataset in ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list.keys():

        print(f'dataset={dataset}')

        if dataset in regression_dataset_names:
            merged_raw_stats_file = regression_merged_raw_stats_csv
            modelling_type = 'Regression'
        else:
            merged_raw_stats_file = classification_merged_raw_stats_csv
            modelling_type = 'Classification'
        
        #---------------------------
        classification_stats = list(classifcation_metric_to_expected_sub_1_minus_sub_2_sign.keys())
        regression_stats = list(regression_metric_to_expected_sub_1_minus_sub_2_sign.keys())
        #----------------------------

        metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals = get_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals(dataset,ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,merged_raw_stats_file,regression_dataset_names,classification_stats,regression_stats,ep_type_matched_to_default_AD_uncertainty_methods,endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col)

        shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals = get_shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals(dataset,ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,merged_raw_stats_file,regression_dataset_names,classification_stats,regression_stats,ep_type_matched_to_default_AD_uncertainty_methods,endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col)
        
        for metric in metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals.keys():
            
            for test_set_type in metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric].keys():
                
                print(f'test_set_type={test_set_type}')

                p_values_info = get_p_values_info_to_support_plot_annotation(dict_of_one_tail_adjusted_p_vals_files,dict_of_two_tail_adjusted_p_vals_files,one_tail_key,two_tail_key,raw_p_val_col,adjusted_p_val_col,p_vals_label_col,modelling_type,metric,test_set_type)



                ########################################
                shift_metric_endpoints_to_all_fold_seed_in_and_out_ad_vals = shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric][test_set_type]
                
                
                dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig = analyse_shift_metric_signs_and_p_values(dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig,shift_metric_endpoints_to_all_fold_seed_in_and_out_ad_vals,dataset,metric,test_set_type,p_values_info)
                #########################################
    
    
    res_df = convert_results_to_df(dict_of_ds_to_test_set_type_to_metric_to_proportions_with_right_sign_and_one_tail_stat_sig)

    if list_of_endpoints_to_focus_on is None:
        res_df.to_csv(os.path.sep.join([out_dir,'Analysis_of_Trends.csv']),index=False)
    else:
        res_df.insert(0,column='Endpoints of interest',value=f'{".".join(list_of_endpoints_to_focus_on)}',allow_duplicates=False)
        res_df.to_csv(os.path.sep.join([out_dir,f'Analysis_of_Trends_Endpoints_of_Interest.csv']),index=False)

    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())

