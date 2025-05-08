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
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
#----------------------------------------------------------------------------
pkg_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict,createOrReplaceDir,doubleDefaultDictOfLists
#---------------------------------------------------------------------------
from class_syn_updates import updates_in_order as class_test_names_in_order
from reg_syn_updates import updates_in_order as reg_test_names_in_order
#----------------------------------------------------------------------------
top_scripts_dir = os.path.dirname(dir_of_this_script)
sys.path.append(top_scripts_dir)

from recommended_defaults import ep_type_matched_to_default_AD_uncertainty_methods
#from consistent_parameters_for_all_modelling_runs import classification_stats_in_desired_order,regression_stats_in_desired_order
from consistent_parameters_for_all_modelling_runs import regression_metric_to_expected_sub_1_minus_sub_2_sign,classifcation_metric_to_expected_sub_1_minus_sub_2_sign
from consistent_parameters_for_all_modelling_runs import endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col
from summary_plots_script_functions import get_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals,make_label_ready_for_file_name,plot_metric_average_and_variability_summary_inside_vs_outside_domain_across_endpoints,get_consistent_metric_limits
from summary_plots_script_functions import get_shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals
from summary_plots_script_functions import get_p_values_info_to_support_plot_annotation
from summary_plots_script_functions import one_tail_key,two_tail_key,raw_p_val_col,adjusted_p_val_col,p_vals_label_col
from summary_plots_global_min_max import globally_observed_min_max_metric_and_shifts_dict
#-------------
top_res_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'SyngentaData'])
out_dir = os.path.sep.join([top_res_dir,'Summary_Plots'])
dir_with_merged_raw_stats = os.path.sep.join([top_res_dir,'Merged_Stats'])
classification_merged_raw_stats_csv = os.path.sep.join([dir_with_merged_raw_stats,'SYN_DT50_Classification_Stats.csv'])
regression_merged_raw_stats_csv = os.path.sep.join([dir_with_merged_raw_stats,'SYN_logP_Regression_Stats.csv'])
regression_dataset_names = ['logP']
classification_dataset_names = ['DT50']

def treat_timepoint_like_endpoint_for_syn_data(merged_raw_stats_file,test_set_type_col,endpoint_col):
    #merge_syn_modelling_statistics.py assigns timepoint label to test_set_type_col column!

    modified_merged_raw_stats_file = re.sub('(\.csv$)','_EndpointTestNamesSwapped.csv',merged_raw_stats_file)
    assert not modified_merged_raw_stats_file == merged_raw_stats_file

    df = pd.read_csv(merged_raw_stats_file)

    #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
    #explains how to swap a pair of column names

    df[[endpoint_col,test_set_type_col]] = df[[test_set_type_col,endpoint_col]]

    df.to_csv(modified_merged_raw_stats_file,index=False)

    return modified_merged_raw_stats_file

dict_of_one_tail_adjusted_p_vals_files = defaultdict(list)
dict_of_two_tail_adjusted_p_vals_files = defaultdict(list)
for modelling_type in ['Classification','Regression']:
    if 'Classification' == modelling_type:
        res_sub_dir = f'{classification_dataset_names[0]}_Updates'
    elif 'Regression' == modelling_type:
        res_sub_dir = f'{regression_dataset_names[0]}_updates'
    else:
        raise Exception(f'Unrecognised modelling_type={modelling_type}')
    
    dict_of_one_tail_adjusted_p_vals_files[modelling_type].append(os.path.sep.join([top_res_dir,res_sub_dir,'Calc','AD.stat.sig',f'one_tail_{modelling_type}_PVals_GlobalAdjusted.csv']))
    dict_of_two_tail_adjusted_p_vals_files[modelling_type].append(os.path.sep.join([top_res_dir,res_sub_dir,'Calc','AD.stat.sig',f'{modelling_type}_PVals_GlobalAdjusted.csv']))

def main():

    print('THE START')

    createOrReplaceDir(out_dir)

    ###################
    #Here, we treat timepoints (= temporal validation test set names) like endpoints for summary plot purposes:
    ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list = {regression_dataset_names[0]:reg_test_names_in_order,classification_dataset_names[0]:class_test_names_in_order}
    ###################

    for dataset in ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list.keys():
        print(f'dataset={dataset}')

        if dataset in regression_dataset_names:
            merged_raw_stats_file = regression_merged_raw_stats_csv
            modelling_type = 'Regression'
        else:
            merged_raw_stats_file = classification_merged_raw_stats_csv
            modelling_type = 'Classification'
        
        merged_raw_stats_file = treat_timepoint_like_endpoint_for_syn_data(merged_raw_stats_file,test_set_type_col,endpoint_col)
        
        #---------------------------
        classification_stats = list(classifcation_metric_to_expected_sub_1_minus_sub_2_sign.keys())
        regression_stats = list(regression_metric_to_expected_sub_1_minus_sub_2_sign.keys())
        #----------------------------

        metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals = get_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals(dataset,ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,merged_raw_stats_file,regression_dataset_names,classification_stats,regression_stats,ep_type_matched_to_default_AD_uncertainty_methods,endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col)

        shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals = get_shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals(dataset,ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,merged_raw_stats_file,regression_dataset_names,classification_stats,regression_stats,ep_type_matched_to_default_AD_uncertainty_methods,endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col)

        for metric in metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals.keys():
            
            metric_lower_limit,metric_upper_limit = get_consistent_metric_limits(metric,metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals,globally_observed_min_max_metric_and_shifts_dict=globally_observed_min_max_metric_and_shifts_dict)

            shift_metric_lower_limit,shift_metric_upper_limit = get_consistent_metric_limits(metric,shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals,shift_metrics=True,globally_observed_min_max_metric_and_shifts_dict=globally_observed_min_max_metric_and_shifts_dict)
            
            for test_set_type in metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric].keys():
                print(f'test_set_type={test_set_type}')

                p_values_info = get_p_values_info_to_support_plot_annotation(dict_of_one_tail_adjusted_p_vals_files,dict_of_two_tail_adjusted_p_vals_files,one_tail_key,two_tail_key,raw_p_val_col,adjusted_p_val_col,p_vals_label_col,modelling_type,metric,test_set_type)
                
                endpoints_to_all_fold_seed_in_and_out_ad_vals = metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric][test_set_type]

                #==============================================================
                if 'Spearman coefficient (PIs vs. residuals)' == metric:
                    alt_y_label_raw = 'SCC'
                    alt_y_label_shift = f'Shift-metric: SCC'
                elif 'Stratified Brier Score' == metric:
                    alt_y_label_raw = 'SBS'
                    alt_y_label_shift = f'Shift-metric: SBS'
                elif 'Validity' == metric:
                    alt_y_label_raw = 'Coverage'
                    alt_y_label_shift = f'Shift-metric: Coverage'
                else:
                    alt_y_label_raw = None
                    alt_y_label_shift = None
                #===============================================================
                
                plot_name = os.path.sep.join([out_dir,f'{dataset}_{make_label_ready_for_file_name(metric)}_{make_label_ready_for_file_name(test_set_type)}.tiff'])

                plot_metric_average_and_variability_summary_inside_vs_outside_domain_across_endpoints(plot_name,endpoints_to_all_fold_seed_in_and_out_ad_vals,endpoint_col,ad_subset_col,val_col=metric,y_min=metric_lower_limit,y_max=metric_upper_limit,alt_x_label='Timepoint',alt_y_label=alt_y_label_raw)

                ########################################
                
                
                shift_metric_endpoints_to_all_fold_seed_in_and_out_ad_vals = shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric][test_set_type]
                
                shift_plot_name = os.path.sep.join([out_dir,f'SHIFT_METRIC_{dataset}_{make_label_ready_for_file_name(metric)}_{make_label_ready_for_file_name(test_set_type)}.tiff'])

                plot_metric_average_and_variability_summary_inside_vs_outside_domain_across_endpoints(shift_plot_name,shift_metric_endpoints_to_all_fold_seed_in_and_out_ad_vals,endpoint_col,ad_subset_col,val_col=f'Shift-metric: {metric}',y_min=shift_metric_lower_limit,y_max=shift_metric_upper_limit,p_values_info=p_values_info,alt_x_label='Timepoint',alt_y_label=alt_y_label_shift)
                #########################################
    
    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())

