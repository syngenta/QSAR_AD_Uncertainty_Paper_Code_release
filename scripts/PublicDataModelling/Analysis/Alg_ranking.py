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
########################################################
#Copyright (c) 2022-2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#This script was primarily written by Zied Hosni (z.hosni [at] sheffield.ac.uk), whilst working on a Syngenta funded project
####################################################
import pandas as pd
import os, sys
import numpy as np
pd.options.mode.chained_assignment = None
from pandas.testing import assert_frame_equal
###########
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
sys.path.append(top_dir)
pkg_dir = os.path.dirname(top_dir)
sys.path.append(pkg_dir)
data_dir = os.path.dirname(os.path.dirname(pkg_dir))

from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from AD_ranking import load_raw_results, \
    filter_raw_metrics_by_dataset_specific_targets, \
    define_dataset_specific_variables, average_over_folds_and_seed, save_average_raw_metrics_df,rename_duplicate_columns, average_dataframes, \
    replace_missing_testset_counts_of_winners_for_some_targets_with_zeros,get_combined_winners_counts_ready_to_write_to_file
from AD_ranking import filter_to_only_keep_default_AD_params_results


def save_target_performance_output(target, out_label, Normalised_perf_score_by_test_df, tot_perf_score_by_test_df, diff_perf_ranking_df,path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner):
    path = target + "_perf_" + out_label + "Norm_wins_count"
    save_dataframe_to_csv(Normalised_perf_score_by_test_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)

    path = target + "_perf_Raw_wins_count_thres_"
    save_dataframe_to_csv(tot_perf_score_by_test_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)

    path = target + "_metrics_perf_ranking_df_"
    save_dataframe_to_csv(diff_perf_ranking_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)
    return ()

def save_target_uncertainty_output(target, out_label, Normalised_uncert_score_by_test_df, tot_uncert_score_by_test_df,
                                  diff_uncer_ranking_df,path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner):
    # save the performance output of a target
    path = target + "_uncert_" + out_label + "Norm_wins_count"
    save_dataframe_to_csv(Normalised_uncert_score_by_test_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)

    path = target + "_uncert_Raw_wins_count_thres_"
    save_dataframe_to_csv(tot_uncert_score_by_test_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)

    path = target + "_metrics_uncer_ranking_df_"
    save_dataframe_to_csv(diff_uncer_ranking_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)
    return ()

def save_dataframe_to_csv(df_to_save, path, path_folder, generalisation, threshold, AD_statSigThresh, dataset, alg_winner):
    reference_id = "_threshold" + str(threshold) + "_" + dataset + ".csv"

    path = path + reference_id
    dataframe_path = os.path.sep.join([path_folder, path])
    df_to_save.to_csv(dataframe_path, index=True)
    return ()

def find_difference_in_average_metric_to_top_ranked(df, pos_metrics, neg_metrics,pos_metrics_total,neg_metrics_total):
    df = df.copy(deep=True)
    df = df.reset_index(drop=True)

    # initialize new dataframe with same size and column names as original dataframe
    new_df = pd.DataFrame(columns=df.columns, index=df.index)

    # iterate over each column
    for col in df.columns:
        # check if column in pos_metrics or col in neg_metrics
        if col in pos_metrics or col in neg_metrics:
            # iterate over each row
            for i in range(df.shape[0]):
                # get subset of rows with same values of "test", "seed", and "alg" columns
                subset = df[(df["Test Set Name (ignoring fold if applicable)"] == df.loc[i, "Test Set Name (ignoring fold if applicable)"])]

                if col in pos_metrics:
                    # calculate maximum value for current column in subset
                    max_val = subset[col].max()
                    if -np.inf == max_val:
                        assert -np.inf == df.loc[i, col], f"df.loc[i, col]={df.loc[i, col]}"
                        diff = np.nan
                    else:
                        diff = max_val - df.loc[i, col]
                elif col in neg_metrics:
                    # calculate minimum value for current column in subset
                    min_val = subset[col].min()
                    # calculate difference between minimum value and current value for current row
                    if np.inf == min_val:
                        assert np.inf == df.loc[i, col], f"df.loc[i, col]={df.loc[i, col]}"
                        diff = np.nan
                    else:
                        diff = df.loc[i, col] - min_val
                else:
                    raise Exception(f"col={col}")
                # assign difference to corresponding column in new dataframe for current row
                new_df.loc[i, col] = diff
        else:
            # if column does not in pos_metrics or col in neg_metrics, copy column from original dataframe to new dataframe
            #-unless the column corresponds to one of the metrics not currently being considered, e.g. uncertainty metrics when differences in shift-metrics for performance metrics are being computed
            if not col in pos_metrics_total+neg_metrics_total:
                new_df[col] = df[col]
            else:
                print(f'Not copying this metric = {col} into the data-frame containing differences to top ranked metrics for these metrics = {pos_metrics+neg_metrics}')
    return new_df

def computeNormalizedWins(new_df, algorithms, total_no_algorithms, pos_metrics, neg_metrics, threshold):

    # loop over each unique value in the "split" column
    out_label = "_"
    tot_score_by_test_df = pd.DataFrame(columns=new_df["Test Set Name (ignoring fold if applicable)"].unique(), index=new_df["Modelling Algorithm"].unique())
    PercWinsPerTest_df = pd.DataFrame(columns=new_df["Test Set Name (ignoring fold if applicable)"].unique(), index=new_df["Modelling Algorithm"].unique())
    for Alg_val in new_df["Modelling Algorithm"].unique():
        # get subset of rows with current value in "split" column
        subset = new_df[new_df["Modelling Algorithm"] == Alg_val]
        for test_val in subset["Test Set Name (ignoring fold if applicable)"].unique():
            subset_test = subset[subset["Test Set Name (ignoring fold if applicable)"] == test_val]
            subset_test = subset_test[subset_test["Modelling Algorithm"] == Alg_val]
            # initialize score variables to 0
            mp_score = 0
            mn_score = 0
            nan_count = 0
            len_subset = 0
            # loop over each column
            for col in new_df.columns:
                # check if column is among pos_metrics or neg_metrics
                if col in pos_metrics or col in neg_metrics:
                    # calculate number of values below threshold of 0.01 for current column in subset
                    num_below_thresh = (subset_test[col] <= threshold).sum(skipna=True)
                    nan_count += subset_test[col].isna().sum()
                    len_subset += len(subset_test[col])
                    # add to score variable for corresponding pos_metrics or neg_metrics columns
                    if col in pos_metrics:
                        mp_score += num_below_thresh
                    else:
                        mn_score += num_below_thresh
            #total score, no normalisation, no rescaling
            tot_score_by_test = mp_score + mn_score
            tot_score_by_test_df.loc[Alg_val, test_val] = tot_score_by_test
            

            #Normalised score
            PercWinsPerTest = 100 * (tot_score_by_test / ((len(pos_metrics)+ len(neg_metrics)))) #- nan_count
            PercWinsPerTest_df.loc[Alg_val, test_val] = PercWinsPerTest
    return tot_score_by_test_df, PercWinsPerTest_df, out_label

def remove_ad_splitting_results(averaged_raw_df):
    ###################
    #Debugging:
    print(f"averaged_raw_df['AD Method'].values.tolist()={averaged_raw_df['AD Method'].values.tolist()}")
    ###################
    averaged_raw_df = averaged_raw_df[averaged_raw_df['AD Method']=='No splitting!']
    assert averaged_raw_df.shape[0] > 0
    return(averaged_raw_df)

def save_all_targets_raw_and_normalised_winners_counts(all_targets_ByTest_diff_table_df,
                                                       list_of_raw_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs,
                                                       list_of_normalised_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs,targets,
                                                       path_folder,out_label,generalisation,threshold,AD_statSigThresh,dataset,alg_winner):
    path = "All_target_metrics_diff_table_thres_"
    save_dataframe_to_csv(all_targets_ByTest_diff_table_df, path, path_folder,"NA","NA",AD_statSigThresh,dataset,alg_winner)
    for i in ["raw","norm"]:
        if i=="norm":
            list_of_winners = list_of_normalised_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs
            path = "norm_ALG_wins_per_ts_averaged_over_eps_thresh_"
        else:
            list_of_winners = list_of_raw_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs
            path = "raw_ALG_wins_per_ts_averaged_over_eps_thresh_"
        # rename_duplicate_columns
        dataframes = rename_duplicate_columns(list_of_winners)
        #replace_missing_testset_counts_of_winners_for_some_targets_with_zeros
        dataframes = replace_missing_testset_counts_of_winners_for_some_targets_with_zeros(dataframes)
        #average_dataframes
        average_df = average_dataframes(dataframes)
        average_df.columns = [col if col.endswith('_uncert') else f"{col}_perf" for col in average_df.columns]
        # Sort the columns by those ending with '_perf' and then '_uncer'
        sorted_columns = sorted(average_df.columns, key=lambda x: (not x.endswith('_perf'), x))
        # Reindex the DataFrame with the sorted column names
        df_sorted = average_df[sorted_columns]
        # Insert the column names as the first row in the DataFrame
        new_row = pd.DataFrame([sorted_columns], columns=sorted_columns)
        average_df = pd.concat([new_row, df_sorted]).reset_index(drop=False)

        # Divide by the number of targets to get the average
        if len(targets) > 0:
            all_targets_ByTest_ranking_table = average_df
        else:
            all_targets_ByTest_ranking_table = pd.DataFrame()

        all_targets_ByTest_ranking_table.loc[0] = all_targets_ByTest_ranking_table.columns
        print(all_targets_ByTest_ranking_table.shape[0])
        print(all_targets_ByTest_ranking_table.shape[1])

        if 'ivit_perf' in all_targets_ByTest_ranking_table.columns:
            all_targets_ByTest_ranking_table.columns = ['Algorithm', 'Prediction performance', 'Prediction performance',
                                                        'Uncertainty estimation', 'Uncertainty estimation']
        else:
            all_targets_ByTest_ranking_table.columns = ['Algorithm','Prediction performance', 'Prediction performance',
                                                        'Prediction performance', 'Uncertainty estimation',
                                                        'Uncertainty estimation', 'Uncertainty estimation']

        # save the updated dataframe to a local path
        save_dataframe_to_csv(all_targets_ByTest_ranking_table, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)
    return()

def main():
    os.chdir('../..')
    my_path = os.getcwd()
    root_path = os.path.sep.join([data_dir, 'PublicData', "Algorithm_ranking_results"])
    createOrReplaceDir(root_path)
    datasets = ['CHEMBL', 'Morger', 'Wang']

    for dataset in datasets:
        metrics_csv, targets, algs, pos_performance_metrics, neg_performance_metrics, pos_uncertainty_metrics, neg_uncertainty_metrics, dataset_name, path_folder = define_dataset_specific_variables(
            dataset, root_path)
        
        pos_metrics_total = pos_performance_metrics[:] + pos_uncertainty_metrics[:]
        neg_metrics_total = neg_performance_metrics[:] + neg_uncertainty_metrics[:]

        createOrReplaceDir(path_folder)

        raw_df = load_raw_results(metrics_csv)

        raw_df = filter_to_only_keep_default_AD_params_results(raw_df)

        filt_raw_df = filter_raw_metrics_by_dataset_specific_targets(raw_df, targets)

        averaged_raw_df = average_over_folds_and_seed(filt_raw_df, is_shift_metrics_df=False)

        averaged_raw_df = remove_ad_splitting_results(averaged_raw_df)

        save_average_raw_metrics_df(path_folder, average_raw_metrics_df=averaged_raw_df)#to do check 1)are there any unexpected target and are all
        # expected targets present. 2) Are only the NoSplitting results presents. 3) Are the averages over all folds and seeds consistent with the
        # raw file (Exemplar_Endpoints_Regression_Stats.csv)
        #Each of these datasets coresponds to a collection of targets. But in general, a dataset may refer to a single target or an endpoint.

        #define the threshold to consider a runner-up as a winner
        for threshold in [0,0.01]:
            all_targets_ByTest_diff_table_list = []
            list_of_raw_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs = []
            list_of_normalised_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs = []

            for target in targets:


                sub_df = averaged_raw_df[averaged_raw_df['Endpoint'] == target]

                for evaluation_type in ["Performance", "Uncertainty"]:
                    if evaluation_type == "Performance":
                        pos_metrics = pos_performance_metrics
                        neg_metrics = neg_performance_metrics
                        #save_target_winners_analysis_function = save_target_performance_output
                    elif evaluation_type == "Uncertainty":
                        pos_metrics = pos_uncertainty_metrics
                        neg_metrics = neg_uncertainty_metrics
                        #save_target_winners_analysis_function = save_target_uncertainty_output

                    else:
                        raise Exception(f"Evaluation type={evaluation_type}.")
                    diff_ranking_df = find_difference_in_average_metric_to_top_ranked(sub_df,
                                                                                   pos_metrics, neg_metrics,pos_metrics_total,neg_metrics_total)

                    #save the new_df to check these points
                    raw_wins_by_test_df, Normalised_wins_by_test_df, out_label = computeNormalizedWins(diff_ranking_df, algs,
                                                                                                                        len(algs), pos_metrics,
                                                                                                                        neg_metrics, threshold
                                                                                                                        )
                    
                    ##################################################
                    if evaluation_type == "Performance":
                        raw_wins_perf_by_test_df = raw_wins_by_test_df
                        Normalised_perf_score_by_test_df = Normalised_wins_by_test_df
                    elif evaluation_type == "Uncertainty":
                        Normalised_uncert_score_by_test_df = Normalised_wins_by_test_df
                        raw_wins_uncert_by_test_df = raw_wins_by_test_df

                    else:
                        raise Exception(f"Evaluation type={evaluation_type}.")
                
                raw_wins_by_test_df,raw_wins_by_test_df_ready_to_write_to_file = get_combined_winners_counts_ready_to_write_to_file(perf_winners_count_df=raw_wins_perf_by_test_df,uncert_winners_count_df=raw_wins_uncert_by_test_df)
                            
                path = target + "_all_Raw_wins_count_"
                save_dataframe_to_csv(raw_wins_by_test_df_ready_to_write_to_file, path, path_folder,"NA",threshold,"NA",dataset,"NA")

                # ===================================##########################################################
                list_of_raw_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs.append(raw_wins_by_test_df)

                Normalised_wins_ByTest_result,Normalised_wins_ByTest_result_ready_to_write_to_file = get_combined_winners_counts_ready_to_write_to_file(perf_winners_count_df=Normalised_perf_score_by_test_df,uncert_winners_count_df=Normalised_uncert_score_by_test_df)
                            
                path = target + "_all_Norm_wins_count_"
                save_dataframe_to_csv(Normalised_wins_ByTest_result_ready_to_write_to_file, path, path_folder,"NA",threshold,"NA",dataset,"NA")

                list_of_normalised_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs.append(
                    Normalised_wins_ByTest_result)

                ##########################################################################
                # Append the dataframe containing the ranking of the AD methods
                all_targets_ByTest_diff_table_list.append(diff_ranking_df)

            # Full diff table over the different targets
            all_targets_ByTest_diff_table_df = pd.concat(all_targets_ByTest_diff_table_list, axis=0, ignore_index=False)

            save_all_targets_raw_and_normalised_winners_counts(all_targets_ByTest_diff_table_df,
                                                               list_of_raw_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs,
                                                               list_of_normalised_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs,
                                                               targets,
                                                               path_folder, out_label=out_label, generalisation=False, threshold=threshold,
                                                               AD_statSigThresh=None,
                                                               dataset=dataset,
                                                               alg_winner=algs)

    print("Finish everything!")

    

if __name__ == '__main__':
    sys.exit(main())
