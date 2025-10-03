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
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_of_this_script)
from consistent_parameters_for_all_modelling_runs import sig_level_of_interest
#----------------------------------------------------------------------------
pkg_dir = os.path.dirname(dir_of_this_script)
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict,doubleDefaultDictOfLists,findDups
#----------------------------------------------------------------------------

one_tail_key = 'one-tail'
two_tail_key = 'two-tail'
raw_p_val_col='Shift-Metric P-value'
adjusted_p_val_col='GLOBAL Adjusted P-value'
p_vals_label_col = 'Statistical Significance Scenario Label'

def get_subset_of_endpoints_for_this_dataset(dataset,ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,merged_raw_stats_file,endpoint_col):
    df = pd.read_csv(merged_raw_stats_file)

    relevant_endpoints = ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list[dataset]

    df = df[df[endpoint_col].isin(relevant_endpoints)]

    return df

def make_na_fold_numeric(row,fold_col):

    fold = row[fold_col]

    if pd.isna(fold):
        row[fold_col] = 0
    
    return row

def get_metric_names_row_records_and_df_with_relevant_metrics(dataset,ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,merged_raw_stats_file,regression_dataset_names,classification_stats,regression_stats,ep_type_matched_to_default_AD_uncertainty_methods,endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col):
    #**************************************
    if dataset in regression_dataset_names:
        ds_type = 'Regression'
        metric_names = regression_stats
    else:
        ds_type = 'Classification'
        metric_names = classification_stats
    #**************************************

    df = get_subset_of_endpoints_for_this_dataset(dataset,ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,merged_raw_stats_file,endpoint_col)

    #############################
    #We are not going to plot the overall statistics prior to AD status splitting:
    #'N/A' label refers to 'All' AD subset c.f. merge_exemplar_modelling_stats.py, where the relevant function is re-used to merge the other endpoint modelling statistics
    df = df.dropna(subset=[ad_col])
    assert ['Inside','Outside'] == df[ad_subset_col].unique().tolist(),f'df[ad_col].unique().tolist()={df[ad_subset_col].unique().tolist()}'
    #############################

    df = df.apply(make_na_fold_numeric,axis=1,args=(fold_col,))

    #-------------------------------------
    assert 1 == len(df[alg_col].unique().tolist()),f'unique modelling/uncertainty algorithms used to generate these results on other endpoints = {df[alg_col].unique().tolist()}'
    assert 1 == len(df[ad_col].unique().tolist()),f'unique AD methods considered = {df[ad_col].unique().tolist()}'
    #--------------------------------------
    #--------------------------------------
    assert ep_type_matched_to_default_AD_uncertainty_methods[ds_type]['AD_method_name'] == df[ad_col].unique().tolist()[0],f'df[ad_col].unique().tolist()[0]={df[ad_col].unique().tolist()[0]}'
    assert ep_type_matched_to_default_AD_uncertainty_methods[ds_type]['uncertainty_method'] == df[alg_col].unique().tolist()[0],f'df[alg_col].unique().tolist()[0]={df[alg_col].unique().tolist()[0]}'
    #---------------------------------------

    row_records = df.to_dict(orient='records')

    return metric_names,row_records,df


def get_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals(dataset,ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,merged_raw_stats_file,regression_dataset_names,classification_stats,regression_stats,ep_type_matched_to_default_AD_uncertainty_methods,endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col):
    

    metric_names,row_records,df = get_metric_names_row_records_and_df_with_relevant_metrics(dataset,ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,merged_raw_stats_file,regression_dataset_names,classification_stats,regression_stats,ep_type_matched_to_default_AD_uncertainty_methods,endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col)
    

    metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals = {}

    for metric in metric_names:
        metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric] = {}
        for unique_test_set_type_name in df[test_set_type_col].unique().tolist():
            metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric][unique_test_set_type_name] = doubleDefaultDictOfLists()

            relevant_rows = [row for row in row_records if row[test_set_type_col] == unique_test_set_type_name]
            
            for row in relevant_rows:
                metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric][unique_test_set_type_name][row[endpoint_col]][row[ad_subset_col]].append(row[metric])
    
    return metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals

def get_shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals(dataset,ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,merged_raw_stats_file,regression_dataset_names,classification_stats,regression_stats,ep_type_matched_to_default_AD_uncertainty_methods,endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col):
    

    metric_names,row_records,df = get_metric_names_row_records_and_df_with_relevant_metrics(dataset,ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,merged_raw_stats_file,regression_dataset_names,classification_stats,regression_stats,ep_type_matched_to_default_AD_uncertainty_methods,endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col)
    

    shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals = {}

    for metric in metric_names:
        shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric] = {}
        for unique_test_set_type_name in df[test_set_type_col].unique().tolist():
            shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric][unique_test_set_type_name] = {}

            for endpoint in df[endpoint_col].unique().tolist():

                shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric][unique_test_set_type_name][endpoint] = defaultdict(list)

                for fold in df[fold_col].unique().tolist():
                    for seed in df[rnd_seed_col].unique().tolist():
                        relevant_rows = [row for row in row_records if (row[test_set_type_col] == unique_test_set_type_name and row[endpoint_col] == endpoint and row[fold_col] == fold and row[rnd_seed_col] == seed)]

                        assert 2 == len(relevant_rows),f'relevant_rows={relevant_rows},metric={metric},unique_test_set_type_name={unique_test_set_type_name},endpoint={endpoint},fold={fold},seed={seed}'

                        inside_rows = [r for r in relevant_rows if 'Inside'==r[ad_subset_col]]
                        outside_rows = [r for r in relevant_rows if 'Outside'==r[ad_subset_col]]

                        assert 1==len(inside_rows),f'inside_rows={inside_rows}'
                        assert 1==len(outside_rows),f'outside_rows={outside_rows}'

                        shift_metric = inside_rows[0][metric] - outside_rows[0][metric]
                        

                        shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric][unique_test_set_type_name][endpoint]['Inside - Outside'].append(shift_metric)
        
    return shift_metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals

def make_label_ready_for_file_name(label):
    v =  re.sub('(\s+|\(|\))','_',label)
    v = v.replace('%','')
    return v

def add_stat_sig_label_to_this_row(row,endpoint_col,p_values_info,p_vals_label_col):
    endpoint = row[endpoint_col]

    stat_sig_label = p_values_info[endpoint][p_vals_label_col]

    row[p_vals_label_col] = stat_sig_label

    return row

def add_statistical_significance_labels_to_plot_input_df(plot_input_df,endpoint_col,p_values_info,p_vals_label_col):

    plot_input_df = plot_input_df.apply(add_stat_sig_label_to_this_row,axis=1,args=(endpoint_col,p_values_info,p_vals_label_col))

    return plot_input_df

def get_plot_input_df(endpoints_to_all_fold_seed_in_and_out_ad_vals,endpoint_col,val_col,ad_subset_col,p_values_info=None,p_vals_label_col=p_vals_label_col):

    plot_input_dol = defaultdict(list)

    for ep in endpoints_to_all_fold_seed_in_and_out_ad_vals.keys():
        for subset in endpoints_to_all_fold_seed_in_and_out_ad_vals[ep].keys():
            for val in endpoints_to_all_fold_seed_in_and_out_ad_vals[ep][subset]:
                plot_input_dol[endpoint_col].append(ep)
                plot_input_dol[ad_subset_col].append(subset)
                plot_input_dol[val_col].append(val)

    plot_input_df = pd.DataFrame(plot_input_dol)

    if not p_values_info is None:
        plot_input_df = add_statistical_significance_labels_to_plot_input_df(plot_input_df,endpoint_col,p_values_info,p_vals_label_col)

    return plot_input_df

def update_x_not_to_keep(row,x_not_to_keep,x_label,subset_label,y_label,unique_subset_labels):

    x = row[x_label]
    subset = row[subset_label]
    y_count = row[y_label]

    assert subset in unique_subset_labels,f'unique_subset_labels={unique_subset_labels}. subset={subset}'

    assert isinstance(y_count,int),y_count
    assert y_count >= 0,y_count

    if 0 == y_count:
        x_not_to_keep.append(x)


def filter_x_with_no_ys_for_at_least_one_subset(plot_input_df,x_label,y_label,subset_label):

    unique_subset_labels = plot_input_df[subset_label].unique().tolist()

    counts_x_subset_combinations_df = plot_input_df.groupby(by=[x_label,subset_label],as_index=False).count()

    #####################
    #Debug:
    #print(f'counts_x_subset_combinations_df={counts_x_subset_combinations_df}')
    #print(f'counts_x_subset_combinations_df.columns.values.tolist()={counts_x_subset_combinations_df.columns.values.tolist()}')
    ######################

    x_not_to_keep = []

    counts_x_subset_combinations_df.apply(update_x_not_to_keep,axis=1,args=(x_not_to_keep,x_label,subset_label,y_label,unique_subset_labels))

    if not 0 == len(x_not_to_keep):
        print(f'For y-property-name = {y_label}, these x-properties have ZERO values for one of the relevant subsets={x_not_to_keep}')

    plot_input_df = plot_input_df[~plot_input_df[x_label].isin(x_not_to_keep)]



    return plot_input_df

def get_noteworthy_y_value(y_label,sig_level_used_to_get_predictions_and_PIs_for_most_regression_metrics=sig_level_of_interest):
    if 'Shift' == y_label.split('-')[0]:
        return 0.0
    elif y_label in ['Balanced Accuracy','AUC']:
        return 0.50
    elif y_label in ['R2','Pearson coefficient','Spearman coefficient','Spearman coefficient (PIs vs. residuals)','MCC','Kappa','R2 (cal)','Pearson coefficient (cal)','Spearman coefficient (cal)']:
        return 0.0
    elif y_label in ['SCC']: #A shorter name for Spearman coefficient (PI vs. residuals) was introduced for the plots
        return 0.0
    elif y_label in ['Validity','Coverage']: #A clearer name for Validity was introduce for the plots
        return (1.0-sig_level_used_to_get_predictions_and_PIs_for_most_regression_metrics)
    elif y_label in ['RMSE','Efficiency','ENCE','ECE','Stratified Brier Score','RMSE (cal)']:
        return None
    elif y_label in ['SBS']: #A shorter name for Stratified Brier Score was introduced for the plots
        return None
    else:
        raise Exception(f'Unexpected y_label={y_label}')

def get_y_ticks(y_min,y_max,interval=0.2):
       
    adjusted_y_min = (y_min//interval)*interval

    #if (y_max - y_min) <= 2:

    no_steps = int((y_max-adjusted_y_min)//interval)

    y_tick_vals = list(set([(adjusted_y_min + (i*interval)) for i in range(no_steps)]+[y_max]))
    #else:
    #    y_tick_vals = None

    return adjusted_y_min,y_tick_vals

def make_x_axis_tick_labels_nice(matplot_lib_axes_obj,x_axis_tickmarks_rotation=60.0,x_axis_tickmarks_size=12.0,spacing_factor=1.8):
    #####################
    #Consulted documentation:
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xticklabels.html (16/09/25)
    #https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text (16/09/25)
    #####################
    

    individual_x_axis_tick_label_objects = matplot_lib_axes_obj.get_xticklabels(which='both')

    for x_axis_tick_label_obj in individual_x_axis_tick_label_objects:
        x_axis_tick_label_obj.set_rotation(x_axis_tickmarks_rotation)
        x_axis_tick_label_obj.set_horizontalalignment('center')
        x_axis_tick_label_obj.set_size(x_axis_tickmarks_size)
        x_axis_tick_label_obj.set_linespacing(spacing_factor)

def plot_distribution_of_y_vals_per_subset_across_different_x_vals(plot_name,plot_input_df,x_label,y_label,subset_label,y_min=0.0,y_max=1.0,central_estimator='mean',error_bar_settings=("pi",95),x_axis_tickmarks_rotation=70.0,x_axis_tickmarks_size=11.0,y_axis_tickmarks_size=9.0,x_label_size=9.0,y_label_size=9.0,p_values_info=None,p_vals_label_col=p_vals_label_col,should_filter_x_with_no_ys_for_at_least_one_subset=False,alt_x_label=None,alt_y_label=None,legend_fontsize=14):
    ##############################
    #Documentation consulted to understand options:
    #
    #https://seaborn.pydata.org/generated/seaborn.pointplot.html
    #https://seaborn.pydata.org/generated/seaborn.lineplot.html (looks nicer than pointplot, but subsets for intermediate endpoints on the x-axis with no values will not be clearly displayed)
    #https://seaborn.pydata.org/tutorial/error_bars.html (16/03/24)
    #
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylim.html
    #
    #Annotation using p-values info., if applicable, is informed by this:
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
    #https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html#sphx-glr-gallery-statistics-boxplot-demo-py
    #
    #y = 0 line is useful for discriminating between positive and negative shift-metrics:
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axhline.html
    #https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def
    #
    #We want to ensure the legend does not obscure the statistical significance labels:
    #https://seaborn.pydata.org/generated/seaborn.move_legend.html
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html#matplotlib.axes.Axes.legend
    #
    #We want to control which y-axis ticks we see:
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yticks.html
    #
    #Make sure x,y labels fit on plot:
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html
    #
    #Trying to improve (main text) figure plot aesthetics in response to peer-review:
    #https://seaborn.pydata.org/tutorial/aesthetics.html (16/09/25)
    ##############################

    sns.set_style("white")
    #sns.set_theme()
    #sns.set_context("paper")

    #=========================
    if "ci" == error_bar_settings[0]:
        ###################
        #These parameters should only be relevant to bootstrap confidence intervals
        _seed = 42
        _n_boot = 5000
    else:
        _seed = None
        _n_boot = None
    #========================

    if should_filter_x_with_no_ys_for_at_least_one_subset:
        plot_input_df = filter_x_with_no_ys_for_at_least_one_subset(plot_input_df,x_label,y_label,subset_label)
    
    if not alt_x_label is None:
        plot_input_df = plot_input_df.rename(mapper={x_label:alt_x_label},axis=1)
        x_label = alt_x_label
    
    if not alt_y_label is None:
        plot_input_df = plot_input_df.rename(mapper={y_label:alt_y_label},axis=1)
        y_label = alt_y_label

    
    matplot_lib_axes_obj = sns.lineplot(data=plot_input_df,x=x_label,y=y_label,hue=subset_label,style=subset_label,estimator=central_estimator,seed=_seed,n_boot=_n_boot,errorbar=error_bar_settings,markers=True)

    sns.move_legend(bbox_to_anchor=(0.5, 1),obj=matplot_lib_axes_obj,loc='lower center',ncol=2,frameon=False,title=None,fontsize=legend_fontsize)
    #matplot_lib_axes_obj.legend(fontsize=legend_fontsize)

    #matplot_lib_axes_obj.tick_params(axis='x',labelrotation=x_axis_tickmarks_rotation,labelsize=x_axis_tickmarks_size)
    make_x_axis_tick_labels_nice(matplot_lib_axes_obj,x_axis_tickmarks_rotation=x_axis_tickmarks_rotation,x_axis_tickmarks_size=x_axis_tickmarks_size)

    matplot_lib_axes_obj.tick_params(axis='y',labelsize=y_axis_tickmarks_size)#,which='both',labelbottom=True, labeltop=True, labelleft=True, labelright=True)

    matplot_lib_axes_obj.set_xlabel(xlabel=x_label,fontsize=x_label_size)
    matplot_lib_axes_obj.set_ylabel(ylabel=y_label,fontsize=y_label_size)

    #y_min,y_tick_vals = get_y_ticks(y_min,y_max)

    matplot_lib_axes_obj.set_ylim(ymin=y_min,ymax=y_max)

    #if not y_tick_vals is None:
    #    plt.yticks(ticks=y_tick_vals,labels=[f'{v:.1f}' for v in y_tick_vals])

    noteworthy_y_val = get_noteworthy_y_value(y_label)

    if not noteworthy_y_val is None:
        plt.axhline(y=noteworthy_y_val,ls='--',lw=0.8,color='k')

    if not p_values_info is None:
        stat_sig_info_df = plot_input_df.drop_duplicates(subset=[x_label,p_vals_label_col],ignore_index=True)
        ##################
        #Only one statistical significance label should be available per x-axis label!
        assert stat_sig_info_df.shape[0] == plot_input_df.drop_duplicates(subset=[x_label],ignore_index=True).shape[0]
        #################

        if y_max > 0:
            y_pos_for_stat_sig_labels = y_max*0.8
        elif y_max < 0:
            y_pos_for_stat_sig_labels = y_max*1.2
        else:
            y_pos_for_stat_sig_labels = -0.05

        list_of_x_pos_vals_for_stat_sig_labels = [i for i in range(stat_sig_info_df.shape[0])]

        stat_sig_labels = stat_sig_info_df[p_vals_label_col].tolist()

        for i in range(len(list(stat_sig_labels))):
            matplot_lib_axes_obj.text(list_of_x_pos_vals_for_stat_sig_labels[i],y_pos_for_stat_sig_labels,stat_sig_labels[i], size='x-large',horizontalalignment='center')


    plt.tight_layout()

    plt.savefig(plot_name, transparent=True)
    plt.close('all')
    plt.clf()

def plot_metric_average_and_variability_summary_inside_vs_outside_domain_across_endpoints(plot_name,endpoints_to_all_fold_seed_in_and_out_ad_vals,endpoint_col,ad_subset_col,val_col='Value',y_min=0.0,y_max=1.0,p_values_info=None,p_vals_label_col=p_vals_label_col,should_filter_x_with_no_ys_for_at_least_one_subset=True,alt_x_label=None,alt_y_label=None):
    
    plot_input_df = get_plot_input_df(endpoints_to_all_fold_seed_in_and_out_ad_vals,endpoint_col,val_col,ad_subset_col,p_values_info=p_values_info,p_vals_label_col=p_vals_label_col)


    plot_distribution_of_y_vals_per_subset_across_different_x_vals(plot_name,plot_input_df,x_label=endpoint_col,y_label=val_col,subset_label=ad_subset_col,y_min=y_min,y_max=y_max,p_values_info=p_values_info,p_vals_label_col=p_vals_label_col,should_filter_x_with_no_ys_for_at_least_one_subset=should_filter_x_with_no_ys_for_at_least_one_subset,alt_x_label=alt_x_label,alt_y_label=alt_y_label)


def get_empirically_observed_limit(metric,metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals,limit_func=min,shift_metrics=False,expected_values_of_no_folds_times_no_seeds=[5,25]):
    #========================
    if shift_metrics:
        relevant_subsets = ['Inside - Outside']
    else:
        relevant_subsets = ['Inside','Outside']
    #========================
    
    all_metric_vals = []
    
    for test_set_type in metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric].keys():
        for ep in metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric][test_set_type].keys():
            for subset in relevant_subsets:
                ###############
                current_metric_vals = []
                ###############


                for metric_val in metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals[metric][test_set_type][ep][subset]:
                    ###############
                    current_metric_vals.append(metric_val)
                    ###############
                    all_metric_vals.append(metric_val)
                
                #######################
                if not len(current_metric_vals) in expected_values_of_no_folds_times_no_seeds:
                    raise Exception(f'metric={metric},test_set_type={test_set_type},ep={ep},subset={subset},current_metric_vals={current_metric_vals}')
                #######################
    
    non_missing_metric_vals = [v for v in all_metric_vals if not pd.isna(v)]
    if len(non_missing_metric_vals) < len(all_metric_vals): print(f'WARNING: metric = {metric} has {len(all_metric_vals) - len(non_missing_metric_vals)} missing values where the metric was undefined!')

    return limit_func(non_missing_metric_vals)

def get_consistent_metric_limits(metric,metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals,shift_metrics=False,globally_observed_min_max_metric_and_shifts_dict=None):
    
    empirical_min = get_empirically_observed_limit(metric,metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals,limit_func=min,shift_metrics=shift_metrics)
    empirical_max = get_empirically_observed_limit(metric,metrics_matched_to_test_set_type_to_endpoints_to_all_fold_seed_in_and_out_ad_vals,limit_func=max,shift_metrics=shift_metrics)
    

    if shift_metrics:
        name_to_check_in_dict = f'Shift:{metric}'
    else:
        name_to_check_in_dict = metric
    
    get_hardcoded_globally_observed_min_max = False
    if not globally_observed_min_max_metric_and_shifts_dict is None:
        if name_to_check_in_dict in globally_observed_min_max_metric_and_shifts_dict.keys():
            get_hardcoded_globally_observed_min_max = True

    if not get_hardcoded_globally_observed_min_max:
        limit_min = empirical_min
        limit_max = empirical_max
    else:
        limit_min = globally_observed_min_max_metric_and_shifts_dict[name_to_check_in_dict]['min']
        limit_max = globally_observed_min_max_metric_and_shifts_dict[name_to_check_in_dict]['max']

    if metric in ['R2','R2 (cal)']:
        if not shift_metrics:
            limits = [limit_min]
            limits += [1.0]
        else:
            limits = [limit_min,limit_max]
        
    elif metric in  ['Pearson coefficient','Spearman coefficient','Spearman coefficient (PIs vs. residuals)','MCC','Pearson coefficient (cal)','Spearman coefficient (cal)','Kappa']:
        if not shift_metrics:
            limits = [-1.0,1.0]
        else:
            limits = [-2,2]
    elif metric in ['Validity','Balanced Accuracy','AUC']:
        if not shift_metrics:
            limits = [0.0,1.0]
        else:
            limits = [-1.0,1.0]
    elif metric in ['Efficiency','RMSE','ENCE','ECE','Stratified Brier Score','RMSE (cal)']:
        if not shift_metrics:
            limits = [0.0,limit_max]
        else:
            limits = [limit_min,limit_max]
    else:
        raise Exception(f'You need to specify limits for metric={metric}')
    
    assert limits[0] <= empirical_min,f'metric = {metric}. lower limit={limits[0]} vs. empirical minimum = {empirical_min}'
    assert limits[1] >= empirical_max,f'metric = {metric}. upper limit={limits[1]} vs. empirical maximum = {empirical_max}'
    
    metric_lower_limit,metric_upper_limit = limits

    #-------------------------
    assert metric_lower_limit < metric_upper_limit,f'metric={metric}. metric_lower_limit={metric_lower_limit} not < metric_upper_limit={metric_upper_limit}'
    #-------------------------

    return metric_lower_limit,metric_upper_limit

def p_val_is_fraction_of_one_or_missing(p_val):
    if pd.isna(p_val):
        return True
    elif isinstance(p_val,float):
        if (p_val <= 1.0 and p_val>=0.0):
            return True
        else:
            return False
    else:
        return False

def p_val_is_sig(p_val,sig_level_fraction):
    status = (round(p_val,2) <= sig_level_fraction)
    #The following handles cases where the p-value could not be computed or, for the purposes of plot annotation, is treated as if it could not be computed:
    if pd.isna(p_val): assert not status,f'p_val={p_val}'
    return status

def get_statistical_significance_label(one_tail_raw_p_val,one_tail_adjusted_p_val,two_tail_raw_p_val,two_tail_adjusted_p_val,sig_level_perc):
    ##################
    #As can be seen in def add_p_vals_to_p_values_info(...), statistically significant one-tail p-values, or two-tail p-values, where the average shift-metric did not, or did, have the expected sign are repsectively treated, for the purposes of plot annotation, as if they weren't statistically significant.
    ###############

    all_pvals = [one_tail_raw_p_val,one_tail_adjusted_p_val,two_tail_raw_p_val,two_tail_adjusted_p_val]

    assert all([p_val_is_fraction_of_one_or_missing(p_val) for p_val in all_pvals]),f'all_pvals={all_pvals}'

    sig_level_fraction = sig_level_perc/100

    if p_val_is_sig(one_tail_raw_p_val,sig_level_fraction):
        if p_val_is_sig(one_tail_adjusted_p_val,sig_level_fraction):
            label = '**'
        else:
            label = '*'
    elif p_val_is_sig(two_tail_raw_p_val,sig_level_fraction):
        if p_val_is_sig(two_tail_adjusted_p_val,sig_level_fraction):
            label = 'xx'
        else:
            label = 'x'
    else:
        label = ''
    
    return label

def we_are_dealing_with_a_scenario_for_which_the_average_shift_metric_could_not_be_computed(average_shift_metric_val_has_wrong_sign,row,p_val_col,scenario_for_err_msg):

    if pd.isna(average_shift_metric_val_has_wrong_sign):
        assert pd.isna(row[p_val_col]),f'scenario_for_err_msg={scenario_for_err_msg},p_val={row[p_val_col]}'

        return True

    assert isinstance(average_shift_metric_val_has_wrong_sign,bool),f'scenario_for_err_msg={scenario_for_err_msg},average_shift_metric_val_has_wrong_sign={average_shift_metric_val_has_wrong_sign},type(average_shift_metric_val_has_wrong_sign)={type(average_shift_metric_val_has_wrong_sign)}'

    return False



def add_p_vals_to_p_values_info(row,p_values_info,modelling_type,tail_type,raw_p_val_col,adjusted_p_val_col,scenario_col,metric,test_set_type,metric_col='Metric'):

    scenario = row[scenario_col]
    
    if 6 == len(scenario.split('_')):
        endpoint,current_test_set_type,the_word_test,the_word_FPs,ad_method_precursor,modelling_method_precursor = scenario.split('_')
        ad_method = ad_method_precursor
        modelling_method = modelling_method_precursor
    elif 4 == len(scenario.split('_')):
        endpoint,current_test_set_type,ad_method_precursor,modelling_method_precursor = scenario.split('_')
        ad_method = ad_method_precursor.split('AD=')[1]
        modelling_method = modelling_method_precursor.split('m=')[1]
    elif re.match('(T[0-9])',scenario):
        endpoint = scenario
        current_test_set_type = test_set_type
        ###################
        #These should just be placeholders, so should not matter?
        ad_method = 'default AD method'
        modelling_method = 'default modelling method'
        ####################


    if (current_test_set_type == test_set_type and row[metric_col] == metric):

        for p_val_col in [raw_p_val_col,adjusted_p_val_col]:
            p_values_info[modelling_type][tail_type][endpoint][test_set_type][ad_method][modelling_method][p_val_col] = row[p_val_col]

            scenario_for_err_msg = f'{modelling_type}-{tail_type}-{endpoint}-{test_set_type}-{ad_method}-{modelling_method}-{metric}'

            #########################
            #Statistically significant one-tail p-value plot annotations should only be applied if the average shift-metric has the expected sign - as we want to flag statistically significant evidence that the AD method is working as expected:
            #Also, we only want to highlight statistically significant two-tail p-values where the average shift-metric has the unexpected sign!

            if one_tail_key == tail_type:
                average_shift_metric_val_has_wrong_sign = row['average_shift_metric_val_has_wrong_sign']

                #The following is designed to ensure that, for the purposes of plot annotation, the two-tail p-values are not flagged as significant if the average shift-metric has the expected sign:
                p_values_info[modelling_type]['tail-type-placeholder'][endpoint][test_set_type][ad_method][modelling_method]['average_shift_metric_val_has_wrong_sign'] = average_shift_metric_val_has_wrong_sign

                if we_are_dealing_with_a_scenario_for_which_the_average_shift_metric_could_not_be_computed(average_shift_metric_val_has_wrong_sign,row,p_val_col,scenario_for_err_msg):
                    continue
                
                if average_shift_metric_val_has_wrong_sign:
                    print(f'Treat one-tail p-value as IF it was statistically insignificant for the purposes of plotting for modelling_type={modelling_type},ep={endpoint},test_set_type={test_set_type},ad_method={ad_method},modelling_method={modelling_method}')
                    p_values_info[modelling_type][tail_type][endpoint][test_set_type][ad_method][modelling_method][p_val_col] = np.nan
                
                
            elif two_tail_key == tail_type:
                #The following should work, as the one-tail p-value files were parsed first:
                average_shift_metric_val_has_wrong_sign =  p_values_info[modelling_type]['tail-type-placeholder'][endpoint][test_set_type][ad_method][modelling_method]['average_shift_metric_val_has_wrong_sign']
                
                if we_are_dealing_with_a_scenario_for_which_the_average_shift_metric_could_not_be_computed(average_shift_metric_val_has_wrong_sign,row,p_val_col,scenario_for_err_msg):
                    continue

                if not average_shift_metric_val_has_wrong_sign:
                    p_values_info[modelling_type][tail_type][endpoint][test_set_type][ad_method][modelling_method][p_val_col] = np.nan
            else:
                raise Exception(f'tail_type={tail_type}?')
            #########################  


def create_new_p_vals_info_with_stat_sig_labels(p_values_info,p_vals_label_col,sig_level_perc,raw_p_val_col,adjusted_p_val_col,one_tail_key,two_tail_key):

    new_p_vals_info = neverEndingDefaultDict()

    all_endpoints = []

    for modelling_type in p_values_info.keys():
        ############################
        #The choice here should not matter:
        tail_type_placeholder = one_tail_key
        ############################
        for endpoint in p_values_info[modelling_type][tail_type_placeholder].keys():
            all_endpoints.append(endpoint)
            all_test_set_types = [tst for tst in p_values_info[modelling_type][tail_type_placeholder][endpoint].keys()]
            assert 1 == len(all_test_set_types),f'all_test_set_types={all_test_set_types}'
            for test_set_type in p_values_info[modelling_type][tail_type_placeholder][endpoint].keys():
                all_ad_methods = [adm for adm in p_values_info[modelling_type][tail_type_placeholder][endpoint][test_set_type].keys()]
                assert 1 == len(all_ad_methods),f'all_ad_methods={all_ad_methods}'
                for ad_method in p_values_info[modelling_type][tail_type_placeholder][endpoint][test_set_type].keys():
                    all_modelling_methods = [mm for mm in p_values_info[modelling_type][tail_type_placeholder][endpoint][test_set_type][ad_method].keys()]
                    assert 1 == len(all_modelling_methods),f'all_modelling_methods={all_modelling_methods}'
                    for modelling_method in p_values_info[modelling_type][tail_type_placeholder][endpoint][test_set_type][ad_method].keys():
                        
                        for tail_type in [one_tail_key,two_tail_key]:

                            row = p_values_info[modelling_type][tail_type][endpoint][test_set_type][ad_method][modelling_method]

                            
                            if tail_type == one_tail_key:
                                one_tail_raw_p_val = row[raw_p_val_col]
                                one_tail_adjusted_p_val = row[adjusted_p_val_col]
                            elif tail_type == two_tail_key:
                                two_tail_raw_p_val = row[raw_p_val_col]
                                two_tail_adjusted_p_val = row[adjusted_p_val_col]
                            else:
                                raise Exception(f'tail_type={tail_type}')
                            
                        new_p_vals_info[endpoint][p_vals_label_col] = get_statistical_significance_label(one_tail_raw_p_val,one_tail_adjusted_p_val,two_tail_raw_p_val,two_tail_adjusted_p_val,sig_level_perc)

    assert len(all_endpoints) == len(set(all_endpoints)),f'Duplicate endpoints across different modelling_type values={findDups(all_endpoints)}'

    return new_p_vals_info

def get_p_values_info_to_support_plot_annotation(dict_of_one_tail_adjusted_p_vals_files,dict_of_two_tail_adjusted_p_vals_files,one_tail_key,two_tail_key,raw_p_val_col,adjusted_p_val_col,p_vals_label_col,modelling_type,metric,test_set_type,scenario_col='Scenario',sig_level_perc=5):

    p_values_info = neverEndingDefaultDict()

    
    for tail_type in [one_tail_key,two_tail_key]:
        #----------------------------
        if one_tail_key == tail_type:
            dict_of_files_to_parse = dict_of_one_tail_adjusted_p_vals_files
        elif two_tail_key == tail_type:
            dict_of_files_to_parse = dict_of_two_tail_adjusted_p_vals_files
        else:
            raise Exception(f'tail_type={tail_type}')
        #--------------------------

        for file in dict_of_files_to_parse[modelling_type]:
            df = pd.read_csv(file)

            df.apply(add_p_vals_to_p_values_info,axis=1,args=(p_values_info,modelling_type,tail_type,raw_p_val_col,adjusted_p_val_col,scenario_col,metric,test_set_type))
    
    p_values_info = create_new_p_vals_info_with_stat_sig_labels(p_values_info,p_vals_label_col,sig_level_perc,raw_p_val_col,adjusted_p_val_col,one_tail_key,two_tail_key)

    return p_values_info
