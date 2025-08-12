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

from consistent_parameters_for_all_modelling_runs import regression_metric_to_expected_sub_1_minus_sub_2_sign,\
    classifcation_metric_to_expected_sub_1_minus_sub_2_sign,classification_performance_metrics_for_winners_analysis,\
    classification_uncertainty_metrics_for_winners_analysis,regression_performance_metrics_for_winners_analysis,regression_uncertainty_metrics_for_winners_analysis
from consistent_parameters_for_all_modelling_runs import ADMethod2Param,ad_params_col
pkg_dir = os.path.dirname(top_dir)

sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
data_dir = os.path.dirname(os.path.dirname(pkg_dir))
regression_exemplar_metrics_csv = os.path.sep.join([data_dir,'PublicData','ModellingResSummary','Exemplar_Endpoints_Regression_Stats.csv'])
classification_exemplar_metrics_csv = os.path.sep.join([data_dir,'PublicData','ModellingResSummary','Exemplar_Endpoints_Classification_Stats.csv'])

#targets_where_not_all_AD_methods_consistently_show_a_majority_of_nominal_in_domain_molecules_inside_AD = ['COX-2']
#raise Exception('We still need to populate this list!')


###########
#import the metrics tables
# Create an empty list to store the dataframes
def load_raw_results(metrics_csv):

    raw_df = pd.read_csv(metrics_csv)

    raw_df.fillna({'AD Method':'No splitting!'},inplace=True)

    return raw_df

def compute_shift_metrics(df,metrics):
    # Define the metrics for which we want to calculate the differences
    # Columns to group by
    group_columns = [
        "Endpoint", "Test Set Name (ignoring fold if applicable)",
        "Fold (if applicable)", "Random seed", "Modelling Algorithm", "AD Method"
    ]

    # Splitting the dataframe into 'Inside' and 'Outside' subsets
    df_inside = df[df['AD Subset'] == 'Inside'].copy()
    df_outside = df[df['AD Subset'] == 'Outside'].copy()

    # Dropping the 'AD Subset' column as it's no longer needed
    df_inside.drop('AD Subset', axis=1, inplace=True)
    df_outside.drop('AD Subset', axis=1, inplace=True)

    # Merging 'Inside' and 'Outside' dataframes on the group columns
    df_merged = df_inside.merge(df_outside, on=group_columns, suffixes=('_Inside', '_Outside'))

    # Calculating differences for each metric and storing in a new dataframe
    shift_metrics_df = df_merged[group_columns].copy()
    
    for metric in metrics:
        shift_metrics_df[f"{metric} Difference"] = df_merged[f"{metric}_Inside"] - df_merged[f"{metric}_Outside"]

    
    return shift_metrics_df





def average_over_folds_and_seed(df,is_shift_metrics_df=True):

    if is_shift_metrics_df:
        groupby_cols = ['Endpoint', 'Test Set Name (ignoring fold if applicable)', 'Modelling Algorithm', 'AD Method']
    else:
        groupby_cols = ['Endpoint', 'Test Set Name (ignoring fold if applicable)', 'Modelling Algorithm', 'AD Method','AD Subset']

    grouped = df.groupby(groupby_cols, as_index=False)

    averaged_df = grouped.mean()
    return  averaged_df

def filter_raw_metrics_by_dataset_specific_targets(raw_df,targets,ep_col='Endpoint'):
    #Raw metrics for different classification datasets are merged together, but we only want the results for the dataset-specific targets!
    
    filt_raw_df = raw_df[raw_df[ep_col].isin(targets)]

    return filt_raw_df

def compute_shift_metrics_and_then_average_over_folds_and_seeds(filt_raw_df, metrics, path_folder):

    just_shift_metrics_df = compute_shift_metrics(filt_raw_df,metrics)

    just_shift_metrics_averaged_df = average_over_folds_and_seed(just_shift_metrics_df,is_shift_metrics_df=True)

    save_raw_shift_metrics_df(path_folder, just_shift_metrics_df)

    save_average_shift_metrics_df(path_folder, just_shift_metrics_averaged_df)

    

    return(just_shift_metrics_averaged_df)

def find_difference_in_average_shift_metric_to_top_ranked(df, pos_metrics, neg_metrics,pos_metrics_total,neg_metrics_total):
    df = df.copy(deep=True)
    df = df.reset_index(drop=True)
    df.columns = df.columns.str.replace(" Difference", "", regex=False)

    # initialize new dataframe with same size and column names as original dataframe
    new_df = pd.DataFrame(columns=df.columns, index=df.index)
    unique_targets = new_df["Endpoint"].unique().tolist()
    assert len(unique_targets) == 1, f"unique_targets={unique_targets}"

    # iterate over each column
    for col in df.columns:
        # check if column in pos_metrics or col in neg_metrics
        if col in pos_metrics or col in neg_metrics:
            # iterate over each row
            for i in range(df.shape[0]):
                # get subset of rows with same values of "test", "seed", and "alg" columns
                subset = df[(df["Test Set Name (ignoring fold if applicable)"] == df.loc[i, "Test Set Name (ignoring fold if applicable)"]) &
                            (df["Modelling Algorithm"] == df.loc[i, "Modelling Algorithm"])]
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
                print(f'Not copying this shift-metric = {col} into the data-frame containing differences to top ranked shift-metrics for these metrics = {pos_metrics+neg_metrics}')
    return new_df

def onlyKeepBestAlgResults(new_df, algorithms):
    #select the algorithm that won in the competition between the algorithms
    assert 1 == len(algorithms),f'We are meant to be only specifying one algorithm here - not {algorithms}'
    subset_df_alg = new_df[new_df["Modelling Algorithm"] == algorithms[0]]
    assert not 0 == subset_df_alg.shape[0],f'No results for algorithm={algorithms[0]}'
    return subset_df_alg

def df_split(df):

    # Get the name of the first column
    first_column = df.columns[0]

    # Split the first column into separate columns at each underscore
    # The result is a DataFrame where each row contains the split values
    split_columns = df[first_column].str.split('_', expand=True)

    # Ensure that there are exactly six parts as expected
    if split_columns.shape[1] == 6:
        # Assign new column names to the split columns DataFrame
        split_columns.columns = ['target', 'testSet', 'test', 'FPs', 'AD method', 'Algorithm']

        # Drop the original first column from the original DataFrame
        df.drop(first_column, axis=1, inplace=True)

        # Concatenate the original DataFrame with the new split columns DataFrame
        df = pd.concat([split_columns, df], axis=1)
    else:
        raise ValueError("The first column does not contain exactly six parts separated by '_'")

    return df

def computeNormalizedWins(new_df, algorithms, total_no_algorithms, pos_metrics, neg_metrics, threshold, AD_statSigThresh=None):
    ###################################################
    #This function is for counting the percentage of times that an AD method is a winner.
    #we can either do this by looking at the shift-metrics (previously called delta-metrics) for all AD methods and all relevant test sets for the
    # best classification or regression algorithms or we can count the number of wins across all algorithms:
    # choose either normalised_wins_PerTest (consider all algorithms) or normalised_wins_PerTest_BestAlg (consider best classification or regression algorithm).
    # Note the shift-metric is defined as metric inside - metric outside the domain.
    #Note that if we generate results over multiple seeds and folds, the shift-metrics we use for ranking will actually be the average shift-metric
    # over all folds and seeds for a given test set, algorithm, and AD method.
    #
    # To compute the winners, we first consider all relevant shift-metrics for a single test set and algorithm, corresponding to all AD methods,
    # then rank them from best to worst and consider the top ranked winner as well as the AD methods associated with shift metrics which is within
    # the threshold of the top ranked.
    # By "Relevant shift-metrics", we either mean all shift-metrics or, if the argument AD_statSigThresh is not None, we mean the shift-metrics
    # associated with p-value less than AD_statSigThresh.
    #This gives us, for each AD method for the current test set and algorithm, a count of the number of times it was a winner across all
    # considered shift-metrics. These are then added up across all considered algorithms. In order to turn these counts into normalized percentages, we need to divide by the total number of times
    # it could have been a winner. 
    ###############################################
    if 1 == len(algorithms):
        new_df = onlyKeepBestAlgResults(new_df, algorithms)
    elif len(algorithms) == total_no_algorithms:
        new_df= new_df.copy(deep=True)
    else:
        raise Exception('algorithms = {}'.format(algorithms))
    out_label = "" #this variable was recognised to be redundant
    # loop over each unique value in the "split" column
    raw_wins_by_test_df = pd.DataFrame(columns=new_df["Test Set Name (ignoring fold if applicable)"].unique(), index=new_df["AD Method"].unique())
    normalised_wins_PerTest_df = pd.DataFrame(columns=new_df["Test Set Name (ignoring fold if applicable)"].unique(), index=new_df["AD Method"].unique())

    for split_val in new_df["AD Method"].unique():
        # get subset of rows with current value in "split" column
        subset = new_df[new_df["AD Method"] == split_val]

        for test_val in subset["Test Set Name (ignoring fold if applicable)"].unique():
            subset_test = subset[subset["Test Set Name (ignoring fold if applicable)"] == test_val]

            # initialize score variables to 0 (score=count of winners)
            raw_wins_by_test = 0

            overall_count_p_vals_missing_or_metric_vals_missing_testSet_AD_method = 0
            # loop over each column
            for col in new_df.columns:
                # check if column is among pos_metrics or neg_metrics
                if col in pos_metrics or col in neg_metrics:
                    # calculate number of values below threshold of 0.01 for current column in subset
                    num_below_thresh = ((subset_test[col] <= threshold) & subset_test[col].notna()).sum()

                    # add to score variable for corresponding pos_metrics or neg_metrics columns
                    raw_wins_by_test += num_below_thresh

            #total score, no normalisation, no rescaling
            raw_wins_by_test_df.loc[split_val, test_val] = raw_wins_by_test

            #Normalised score
            denominator = ((len(pos_metrics) + len(neg_metrics)) * len(subset_test["Modelling Algorithm"].unique()))
            normalised_wins_PerTest = 100 * (raw_wins_by_test / denominator)

            normalised_wins_PerTest_df.loc[split_val, test_val] = normalised_wins_PerTest

    return new_df, raw_wins_by_test_df, normalised_wins_PerTest_df, out_label

def check_sum_normalised_winners_equal_100_perc(new_df, normalised_wins_PerTest_df,some_shift_metric_differences_were_missing):
    for test_set_type in normalised_wins_PerTest_df.columns.values.tolist():
        perc_wins_for_each_ad_method = normalised_wins_PerTest_df[test_set_type].tolist()
        assert not any([pd.isna(v) for v in perc_wins_for_each_ad_method]), f"perc_wins_for_each_ad_method]={perc_wins_for_each_ad_method}"

        if not some_shift_metric_differences_were_missing:
            assert 100 <= round(sum(perc_wins_for_each_ad_method),2), f"sum(perc_wins_for_each_ad_method)={sum(perc_wins_for_each_ad_method)}"
            if not 100 == round(sum(perc_wins_for_each_ad_method),2):
                print("warning:the percentage wins exceed 100-is this because some shift metrics are to identical to the top ranked for 2 different AD method?")
        else:
            if 100 < round(sum(perc_wins_for_each_ad_method),2):
                print("warning:the percentage wins exceed 100-is this because some shift metrics are to identical to the top ranked for 2 different AD method?")


def import_p_vals(my_path, p_vals_csv):
    #import the csv file to df table
    path = os.path.sep.join([my_path, p_vals_csv])
    p_val_df = pd.read_csv(path)
    return p_val_df

def load_the_pValues_data(dataset_name, data_dir):
    if dataset_name in ["Tox21", "Morger_ChEMBL"]:
        folder_path = os.path.sep.join([data_dir, "PublicData", dataset_name, "Modelling"])
        type_of_modelling = 'binary_class'
        one_tail_PVals_file_name = "one_tail_binary_class_PVals.csv"
    elif dataset_name == "Wang_ChEMBL":
        folder_path = os.path.sep.join([data_dir,"PublicData",dataset_name,"data","dataset","Modelling"])
        type_of_modelling = "regression"
        one_tail_PVals_file_name = "one_tail_regression_PVals.csv"
    else:
        raise Exception(f"Unrecognised dataset name={dataset_name}")
    # Combine the folder path with the file name
    path_one_tail_PVals = os.path.join(folder_path, one_tail_PVals_file_name)
    # Load the pickle and extract dataframe
    info = pd.read_csv(path_one_tail_PVals)
    print(info.keys())

    return (info, folder_path, type_of_modelling)

#selecting the algorithms
def select_class_or_reg_algorithms_to_use_for_AD_ranking(generalisation, algs, alg_winner):
    if generalisation:
        filtered_algs = algs
    else:
        filtered_algs = [word for word in algs if word == alg_winner]
    return (filtered_algs)


def load_and_format_pValues(dataset_name, data_dir):
    info, folder_path, type_of_modelling = load_the_pValues_data(dataset_name, data_dir)
    info = df_split(info)
    return (info, folder_path, type_of_modelling)

def save_dataframe_to_csv(df_to_save, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner):
    
    if generalisation:
        if not 'NA' == alg_winner: raise Exception(f'If generalization is true, winners should be counted based on the results of all modelling/uncertainty algorithms! However, alg_winner={alg_winner} here!')

    reference_id = "_gen_" + str(generalisation) + f"alg={alg_winner}" + "threshold" + str(threshold) + "_AD_statSigThresh_" + str(AD_statSigThresh) + "_" + \
                   dataset + ".csv"
    
    path = path + reference_id
    dataframe_path = os.path.sep.join([path_folder, path])
    df_to_save.to_csv(dataframe_path, index=True)
    return ()

def save_target_performance_output(target, out_label, Normalised_perf_score_by_test_df, tot_perf_score_by_test_df, diff_perf_ranking_df,path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner):
    path = target + "_perf_" + out_label + "Norm_wins_count"
    save_dataframe_to_csv(Normalised_perf_score_by_test_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)

    path = target + "_perf_Raw_wins_count_thres_"
    save_dataframe_to_csv(tot_perf_score_by_test_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)

    path = target + "_metrics_shifts_perf_ranking_df_"
    save_dataframe_to_csv(diff_perf_ranking_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)
    return ()

def save_target_uncertainty_output(target, out_label, Normalised_uncert_score_by_test_df, tot_uncert_score_by_test_df,
                                  diff_uncer_ranking_df,path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner):
    # save the performance output of a target
    path = target + "_uncert_" + out_label + "Norm_wins_count"
    save_dataframe_to_csv(Normalised_uncert_score_by_test_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)

    path = target + "_uncert_Raw_wins_count_thres_"
    save_dataframe_to_csv(tot_uncert_score_by_test_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)

    path = target + "_metrics_shifts_uncer_ranking_df_"
    save_dataframe_to_csv(diff_uncer_ranking_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)
    return ()

def rename_duplicate_columns(dataframes):
    for df in dataframes:
        # Create a dictionary to track occurrences of column names
        col_tracker = {}
        # Use a list to store new column names
        new_cols = []
        for col in df.columns:
            if col in col_tracker:
                # Increment the count for this column
                col_tracker[col] += 1
                # For the first duplicate, use '_uncert', for further duplicates use '_<count>'
                new_col_name = f"{col}_uncert" if col_tracker[col] == 2 else f"{col}_{col_tracker[col] - 1}"
            else:
                # If it's the first occurrence, just use the original column name
                col_tracker[col] = 1
                new_col_name = col  # No suffix for first occurrence
            new_cols.append(new_col_name)
        # Set the new column names for the dataframe
        df.columns = new_cols

    return dataframes

def replace_missing_testset_counts_of_winners_for_some_targets_with_zeros(dataframes):

    if not dataframes:
        return dataframes  # Return the original list if it's empty

    # Identify the dataframe with the most columns
    max_cols_df = max(dataframes, key=lambda x: len(x.columns))

    # Extract the set of columns from the dataframe with the most columns
    max_cols_set = set(max_cols_df.columns)

    # Update each dataframe in the list
    for df in dataframes:
        # Find missing columns by comparing with the max columns set
        missing_cols = max_cols_set - set(df.columns)

        # For each missing column, add it to the dataframe and fill with zeros
        for col in missing_cols:
            df[col] = 0

    # Return the updated list of dataframes
    return dataframes


def average_dataframes(df_list):
    # Align all DataFrames to the same index order by sorting the indices
    aligned_dfs = [df.sort_index() for df in df_list]

    # Ensure all DataFrames have the same columns in the same order
    # This step is crucial for accurately calculating the mean across DataFrames
    common_columns = list(set.intersection(*(set(df.columns) for df in aligned_dfs)))
    aligned_dfs = [df[common_columns].sort_index(axis=1) for df in aligned_dfs]

    # Create an empty DataFrame to hold the averaged values
    avg_df = pd.DataFrame(index=aligned_dfs[0].index, columns=common_columns)

    # Calculate the mean of each column across all aligned DataFrames
    for column in common_columns:
        avg_df[column] = pd.concat([df[column] for df in aligned_dfs], axis=1).mean(axis=1)

    return avg_df

def update_column_names_perf_uncert_joint_winners_df(all_wins_ByTest_result,perf_winners_count_df,uncert_winners_count_df):
    all_wins_ByTest_result = all_wins_ByTest_result.copy(deep=True)

    new_column_names = []
    new_column_names += [f'Performance_winners_{c}' for c in perf_winners_count_df.columns.values.tolist()]
    new_column_names += [f'Uncertainty_winners_{c}' for c in uncert_winners_count_df.columns.values.tolist()]

    
    all_wins_ByTest_result.columns = new_column_names

    return all_wins_ByTest_result


def get_combined_winners_counts_ready_to_write_to_file(perf_winners_count_df,uncert_winners_count_df):
    all_wins_ByTest_result = pd.concat(
        [perf_winners_count_df, uncert_winners_count_df],
        axis=1)
    
    all_wins_ByTest_result_ready_to_write_to_file = update_column_names_perf_uncert_joint_winners_df(all_wins_ByTest_result,perf_winners_count_df,uncert_winners_count_df)
    
    assert not all_wins_ByTest_result_ready_to_write_to_file.columns.values.tolist() == all_wins_ByTest_result.columns.values.tolist(),all_wins_ByTest_result.columns.values.tolist()

    return all_wins_ByTest_result,all_wins_ByTest_result_ready_to_write_to_file

def save_all_targets_raw_and_normalised_winners_counts(all_targets_ByTest_diff_table_df,
                                                       list_of_raw_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs,
                                                       list_of_normalised_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs,targets,
                                                       path_folder,out_label,generalisation,threshold,AD_statSigThresh,dataset,alg_winner):
    path = "All_target_metrics_diff_shifts_table_thres_"
    save_dataframe_to_csv(all_targets_ByTest_diff_table_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)
    for i in ["raw","norm"]:
        if i=="norm":
            list_of_winners = list_of_normalised_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs
            path = "norm_AD_wins_per_ts_averaged_over_eps_thresh_"
        else:
            list_of_winners = list_of_raw_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs
            path = "raw_AD_wins_per_ts_averaged_over_eps_thresh_"
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
            all_targets_ByTest_ranking_table.columns = ['AD', 'Prediction performance', 'Prediction performance',
                                                        'Uncertainty estimation', 'Uncertainty estimation']
        else:
            all_targets_ByTest_ranking_table.columns = ['AD','Prediction performance', 'Prediction performance',
                                                        'Prediction performance', 'Uncertainty estimation',
                                                        'Uncertainty estimation', 'Uncertainty estimation']

        # save the updated dataframe to a local path
        save_dataframe_to_csv(all_targets_ByTest_ranking_table, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)
    return()

def save_average_raw_metrics_df(path_folder,average_raw_metrics_df):
    path_file = "average_raw_metrics_df.csv"
    full_file_name = os.path.sep.join([path_folder,path_file])
    average_raw_metrics_df.to_csv(full_file_name,index=False)

def save_average_shift_metrics_df(path_folder, average_shift_metrics_df):
    path_file = "average_shift_metrics_df.csv"
    cleaned_final_df_path = os.path.sep.join([path_folder, path_file])
    average_shift_metrics_df.to_csv(cleaned_final_df_path, index=False)
    

def save_raw_shift_metrics_df(path_folder, raw_shift_metrics_df):

    path_file = "raw_shift_metrics_df.csv"
    cleaned_final_df_path = os.path.sep.join([path_folder, path_file])
    raw_shift_metrics_df.to_csv(cleaned_final_df_path, index=False)
    return()

def merge_average_shift_metrics_with_p_vals(average_shift_metrics_df, p_vals_df, path_folder, dataset_name, metrics,generalisation,threshold,AD_statSigThresh,dataset,alg_winner):
    scenarios_keys = ['Endpoint', 'Test Set Name (ignoring fold if applicable)', 'Modelling Algorithm', 'AD Method']
    how = 'inner'
    # Perform the merge
    # Rename columns in p_vals_df to match those in average_shift_metrics_df
    rename_columns = {
        'target': 'Endpoint',
        'testSet': 'Test Set Name (ignoring fold if applicable)',
        'AD method': 'AD Method',
        'Algorithm': 'Modelling Algorithm'
    }
    p_vals_df_renamed = p_vals_df.rename(columns=rename_columns)

    # Adjust the keys list according to the renamed columns for a proper merge
    scenarios_keys_adjusted = [rename_columns.get(key, key) for key in scenarios_keys]
    #extract from p_vals_df_renamed the rows with
    #that exist in the average_shift_metrics_df

    # Perform the merge using the adjusted keys
    merged_df = pd.merge(average_shift_metrics_df, p_vals_df_renamed, on=scenarios_keys_adjusted, how=how)
    merged_df["Fold (if applicable)"] = 0
    path = "average_shift_metrics_merged_with_pvalues"
    save_dataframe_to_csv(merged_df, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)
    scenarios_with_missing_p_values = identify_scenarios_where_pvalues_are_missing(average_shift_metrics_df, p_vals_df_renamed,
                                                                                   scenarios_keys_adjusted,
                                                                                  path_folder,generalisation,threshold,AD_statSigThresh,dataset)
    ############################
    if not 0 == len(scenarios_with_missing_p_values):
        cols_used_for_merge = scenarios_keys_adjusted 
        assert cols_used_for_merge == ['Endpoint' ,'Test Set Name (ignoring fold if applicable)', 'Modelling Algorithm','AD Method'],f'cols_used_for_merge={cols_used_for_merge}'
        print(f'scenarios_with_missing_p_values={scenarios_with_missing_p_values}')
        print('average_shift_metrics_df[cols_used_for_merge].head()=')
        print(average_shift_metrics_df[cols_used_for_merge].head())
        print('p_vals_df_renamed.head()=')
        print(p_vals_df_renamed.head())
        print(f'merged_df.columns.values.tolist()={merged_df.columns.values.tolist()}')
        raise Exception('Fix this!')
    else:
        print('This bug appears to have been fixed!')
    ###########################

    scenarios_in_merged_df = get_list_of_scenarios(df=merged_df,keys_adjusted=scenarios_keys_adjusted)
    assert 0 == len(set(scenarios_in_merged_df).intersection(set(scenarios_with_missing_p_values)))
    return merged_df

def identify_scenarios_where_pvals_were_computed_but_missing_metrics_were_found(merged_df, average_shift_metrics_df, keys_adjusted,
                                                                                dataset_name, metrics_columns):
    # Check for scenarios where p-values were computed but metrics might be missing

    merged_df.columns = [col.replace(" Difference", "") for col in merged_df.columns]

    incomplete_metrics = merged_df[metrics_columns].isna().any().any()

    if incomplete_metrics:
        raise Exception(f"There are scenarios with computed p-values but missing metrics were found for dataset = {dataset_name}.")
    else:
        print("No scenarios with computed p-values but missing metrics were found.")
    return(merged_df)

def get_list_of_scenarios(df,keys_adjusted,scenario_col="scenario"):
    df = df.copy(deep=True)
    df = df.apply(add_scenarios_column,axis=1,args=(keys_adjusted,scenario_col))
    scenarios = df[scenario_col].unique().tolist()
    return(scenarios)

def add_scenarios_column(row, keys_adjusted, scenario_col):
    row[scenario_col] = "_".join([row[key] for key in keys_adjusted])
    return(row)

def identify_scenarios_where_pvalues_are_missing(average_shift_metrics_df,p_vals_df_renamed,keys_adjusted,path_folder,generalisation,threshold,AD_statSigThresh,dataset):
    # Pre-check to identify records in average_shift_metrics_df without corresponding p-value entries in p_vals_df
    merged_df_precheck = pd.merge(average_shift_metrics_df, p_vals_df_renamed, on=keys_adjusted, how='left', indicator=True)
    missing_p_values = merged_df_precheck[merged_df_precheck['_merge'] == 'left_only']
    
    if not missing_p_values.empty:
        print(f"Records removed due to missing p-values: {missing_p_values.shape[0]}")
        scenarios = get_list_of_scenarios(df=missing_p_values, keys_adjusted=keys_adjusted)
    else:
        print("No records were removed due to missing p-values.")
        scenarios = []
    return(scenarios)


def filter_average_shift_metrics_by_p_vals(df, p_value_threshold):
    p_value_column = 'Shift-Metric P-value'

    # Check if the p-value column exists
    if p_value_column not in df.columns:
        raise ValueError(f"{p_value_column} column not found in dataframe.")

    # Filter the dataframe based on the p-value threshold
    #22/03/25: syntax checked for simple example in Python interpreter
    filtered_df = df[round(df[p_value_column],2) <= p_value_threshold]

    return filtered_df

def assign_metrics_which_did_not_pass_p_val_or_sign_filter_as_plus_or_minus_inf_so_they_can_never_be_winners(df, metric_cols, positive_metrics, 
                                                                                                  negative_metrics):
    # Dynamically determine the metric column names, excluding specified columns

    # Create an empty DataFrame to store results
    df.columns = [col.replace(" Difference", "") for col in df.columns]
    #df.rename(columns={'Stratified Brier Score': 'Stratified Brier'}, inplace=False)
    adjusted_df = pd.DataFrame(columns=df.columns)
    # Process each group
    for _, group in df.groupby(['Endpoint','Test Set Name (ignoring fold if applicable)', 'Modelling Algorithm', 'AD Method']):
        # Iterate through each row in the group
        for idx, row in group.iterrows():
            # For each metric column, check if it matches the 'metric' value
            for col in metric_cols:
                # Determine the replacement value based on whether the metric is positive or negative
                if col in negative_metrics:
                    replacement_value = np.inf
                elif col in positive_metrics:
                    replacement_value = -np.inf
                else:
                    raise Exception(f'col={col} negative_metrics={negative_metrics} positive_metrics={positive_metrics}')
                

                # If the column does not match any of the 'metric' values, set it to the appropriate infinity
                #This corresponds to a situation where the one-tail p-value was not statistically significant, so the corresponding row was filtered in a previously applied function
                if not col in group['Metric'].values:
                    df.at[idx, col] = replacement_value
                
                else:
                    #Trying to ensure average shift-metrics without the expected sign are also not considered winners, in case these are sometimes considered statistically significant - which can happen!
                    

                    if col in positive_metrics:
                        if not df.at[idx, col] > 0:
                            df.at[idx, col] = replacement_value
                    elif col in negative_metrics:
                        if not df.at[idx, col] < 0:
                            df.at[idx, col] = replacement_value
                    else:
                        raise Exception(f'col={col} negative_metrics={negative_metrics} positive_metrics={positive_metrics}')
                

            # Append the modified row to the adjusted DataFrame
            adjusted_df = adjusted_df.append(df.loc[idx])

    return adjusted_df

def remove_rows_with_duplicate_scenarios(df):
    # Remove duplicate rows based on scenarios
    df_cleaned = df.drop_duplicates(subset=['Endpoint', 'Test Set Name (ignoring fold if applicable)', 'Modelling Algorithm', 'AD Method'])

    # Remove the 'pval' and 'metric' and other irrelevant columns
    df_cleaned = df_cleaned.drop(columns=['Metric', 'test', 'FPs', 'Shift-Metric P-value', 'all_p_vals_str',
       'all_shift_metrics_str', 'average_shift_metric_val',
       'average_shift_metric_val_has_wrong_sign'])

    return df_cleaned

def define_dataset_specific_variables(dataset,out_dir_top):
    if dataset == 'Wang':
        targets = ['Dopamine', 'COX-1', 'COX-2']
        

        metrics_csv = regression_exemplar_metrics_csv
        algs = ['SCP', 'ACP', 'ICP', 'Native']
        
        ##################################
        pos_performance_metrics = [m for m in regression_performance_metrics_for_winners_analysis if
                                   regression_metric_to_expected_sub_1_minus_sub_2_sign[m] > 0]

        neg_performance_metrics = [m for m in regression_performance_metrics_for_winners_analysis if
                                   regression_metric_to_expected_sub_1_minus_sub_2_sign[m] < 0]

        pos_uncertainty_metrics = [m for m in regression_uncertainty_metrics_for_winners_analysis if
                                   regression_metric_to_expected_sub_1_minus_sub_2_sign[m] > 0]

        neg_uncertainty_metrics = [m for m in regression_uncertainty_metrics_for_winners_analysis if
                                   regression_metric_to_expected_sub_1_minus_sub_2_sign[m] < 0]
        
        dataset_name = "Wang_ChEMBL"

    elif dataset == 'Morger':
        metrics_csv = classification_exemplar_metrics_csv
        targets = ['NR-Aromatase', 'NR-AR', 'SR-HSE', 'SR-ARE']
        
        algs = ['IVAP', 'CVAP', 'Native']
        
        ####################
        pos_performance_metrics = [m for m in classification_performance_metrics_for_winners_analysis if
                                   classifcation_metric_to_expected_sub_1_minus_sub_2_sign[m] > 0]

        neg_performance_metrics = [m for m in classification_performance_metrics_for_winners_analysis if
                                   classifcation_metric_to_expected_sub_1_minus_sub_2_sign[m] < 0]

        pos_uncertainty_metrics = [m for m in classification_uncertainty_metrics_for_winners_analysis if
                                   classifcation_metric_to_expected_sub_1_minus_sub_2_sign[m] > 0]

        neg_uncertainty_metrics = [m for m in classification_uncertainty_metrics_for_winners_analysis if
                                   classifcation_metric_to_expected_sub_1_minus_sub_2_sign[m] < 0]

        dataset_name = "Tox21"

    elif dataset == 'CHEMBL':

        metrics_csv = classification_exemplar_metrics_csv
        targets = ["CHEMBL228", "CHEMBL240", "CHEMBL206", "CHEMBL4078"]
        
        algs = ['IVAP', 'CVAP', 'Native']
        

        pos_performance_metrics = [m for m in classification_performance_metrics_for_winners_analysis if
                                   classifcation_metric_to_expected_sub_1_minus_sub_2_sign[m] > 0]

        neg_performance_metrics = [m for m in classification_performance_metrics_for_winners_analysis if
                                   classifcation_metric_to_expected_sub_1_minus_sub_2_sign[m] < 0]

        pos_uncertainty_metrics = [m for m in classification_uncertainty_metrics_for_winners_analysis if
                                   classifcation_metric_to_expected_sub_1_minus_sub_2_sign[m] > 0]

        neg_uncertainty_metrics = [m for m in classification_uncertainty_metrics_for_winners_analysis if
                                   classifcation_metric_to_expected_sub_1_minus_sub_2_sign[m] < 0]
        dataset_name = "Morger_ChEMBL"

    else:
        raise Exception(f"Unrecognised dataset={dataset}")

    

    path_folder = os.path.sep.join([out_dir_top,dataset_name])
    return(metrics_csv, targets, algs, pos_performance_metrics,neg_performance_metrics,pos_uncertainty_metrics,
           neg_uncertainty_metrics,dataset_name,path_folder)

def check_AD_method_is_present_for_each_testset_type_for_each_algorithm(new_df):
    assert 1== len(new_df["Endpoint"].unique().tolist()), f'new_df["Endpoint"].unique().tolist()={new_df["Endpoint"].unique().tolist()}'
    for testset in new_df["Test Set Name (ignoring fold if applicable)"].unique().tolist():
        first_subset_df = new_df[new_df["Test Set Name (ignoring fold if applicable)"].isin([testset])]
        for algorithm in first_subset_df["Modelling Algorithm"].unique().tolist():
            subset_df = first_subset_df[first_subset_df["Modelling Algorithm"].isin([algorithm])]

            assert 4 == len(subset_df["AD Method"].tolist()), f'subset_df["AD Method"].tolist()={subset_df["AD Method"].tolist()}'

def report_if_all_AD_methods_present_for_all_targets(average_shift_metrics_ready_for_ranking):
    for target in average_shift_metrics_ready_for_ranking['Endpoint'].unique().tolist():
        target_subset = average_shift_metrics_ready_for_ranking[average_shift_metrics_ready_for_ranking['Endpoint'] == target]
        for testset in target_subset["Test Set Name (ignoring fold if applicable)"].unique().tolist():
            subset_df = target_subset[target_subset["Test Set Name (ignoring fold if applicable)"].isin([testset])]
            if not 4 == len(subset_df["AD Method"].unique().tolist()):
                return False
    return(True)

def add_a_dummy_row_to_avoid_dropping_AD_method_if_all_pvalues_are_insignificant(p_vals_df):
    # Add a new column 'scenario' which combines the relevant columns
    p_vals_df['scenario'] = p_vals_df.apply(
        lambda row: ';'.join([str(row['target']), str(row['Algorithm']), str(row['AD method']), str(row['testSet'])]),
        axis=1
    )

    # Get unique scenarios
    unique_scenarios = p_vals_df['scenario'].unique()

    # Define a DataFrame to hold the dummy rows
    dummy_df = pd.DataFrame()

    # Iterate over each unique scenario
    for scenario in unique_scenarios:
        # Split the scenario back into its components
        target, algorithm, ad_method, testset = scenario.split(';')

        # Create a dictionary for the dummy row
        dummy_row = {
            'target': target,
            'Algorithm': algorithm,
            'AD method': ad_method,
            'testSet': testset,
            'Shift-Metric P-value': 0
        }

        # Assign 'dummy' to all other columns
        other_cols = p_vals_df.columns.difference(['target', 'AD method', 'Algorithm', 'testSet', 'Shift-Metric P-value', 'scenario'])
        for col in other_cols:
            dummy_row[col] = 'dummy'

        # Append the dummy row to the dummy DataFrame
        dummy_df = dummy_df.append(dummy_row, ignore_index=True)

    # Remove the scenario column from the original data frame before returning
    p_vals_df.drop('scenario', axis=1, inplace=True)

    # Concatenate the original DataFrame with the dummy DataFrame
    p_vals_df = pd.concat([p_vals_df, dummy_df], ignore_index=True)
    return p_vals_df

def ensure_average_shift_metrics_corresponding_to_statistically_insignificant_p_values_or_the_wrong_sign_cannot_be_considered_winners(average_shift_metrics_df,AD_statSigThresh,dataset_name,data_dir,path_folder,metrics,pos_metrics,neg_metrics):
    #=====================================
    assert not AD_statSigThresh is None and isinstance(AD_statSigThresh,float),f'No statistical significance considerations should be being applied when AD_statSigThresh={AD_statSigThresh}!'
    assert AD_statSigThresh >=0 and AD_statSigThresh < 1,AD_statSigThresh
    #======================================
    #######################################
    #dummy values, as we should not have specified that these were included in the output file names here, as these are irrelevant inside this function:
    generalisation,threshold,dataset,alg_winner=['NA']*4
    #######################################

    # Load the table of p-values for different scenarios
    p_vals_df, folder_path, type_of_modelling = load_and_format_pValues(dataset_name, data_dir)
    p_vals_df= add_a_dummy_row_to_avoid_dropping_AD_method_if_all_pvalues_are_insignificant(p_vals_df)
    
    #step1
    merged_df_of_average_shift_metrics_and_p_vals = merge_average_shift_metrics_with_p_vals(average_shift_metrics_df, p_vals_df,path_folder,dataset_name, metrics,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)
    
    #step2

    average_shift_metrics_filtered_by_p_value = filter_average_shift_metrics_by_p_vals(merged_df_of_average_shift_metrics_and_p_vals,AD_statSigThresh)

    #step3
    average_shift_metrics_set_to_plus_or_minus_inf_if_did_not_pass_p_value_or_sign_filter = assign_metrics_which_did_not_pass_p_val_or_sign_filter_as_plus_or_minus_inf_so_they_can_never_be_winners(average_shift_metrics_filtered_by_p_value, metrics,pos_metrics,neg_metrics)

    assert not average_shift_metrics_set_to_plus_or_minus_inf_if_did_not_pass_p_value_or_sign_filter[metrics].isna().any().any(),f"average_shift_metrics_set_to_plus_or_minus_inf_if_did_not_pass_p_value_or_sign_filter={average_shift_metrics_set_to_plus_or_minus_inf_if_did_not_pass_p_value_or_sign_filter}"

    #step4
    average_shift_metrics_ready_for_ranking = remove_rows_with_duplicate_scenarios(average_shift_metrics_set_to_plus_or_minus_inf_if_did_not_pass_p_value_or_sign_filter)

    assert not average_shift_metrics_ready_for_ranking[metrics].isna().any().any(), f"average_shift_metrics_ready_for_ranking={average_shift_metrics_ready_for_ranking}"

    return average_shift_metrics_ready_for_ranking

def replace_shift_metrics_with_difference_to_top_ranked_prior_to_writing_to_file_for_both_kinds_of_metrics(for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df,shift_metrics_differences_to_top_ranked_for_current_algs_df,pos_metrics,neg_metrics):
    
    #--------------------
    #Without this, copying whole column contents would not work:
    for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df = for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df.reset_index(inplace=False)


    shift_metrics_differences_to_top_ranked_for_current_algs_df = shift_metrics_differences_to_top_ranked_for_current_algs_df.reset_index(inplace=False)

    assert 1 == len(for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df["Endpoint"].unique().tolist())
    assert 1 == len(shift_metrics_differences_to_top_ranked_for_current_algs_df["Endpoint"].unique().tolist())
    assert for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df["Endpoint"].tolist()==shift_metrics_differences_to_top_ranked_for_current_algs_df["Endpoint"].tolist()
    assert for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df["Test Set Name (ignoring fold if applicable)"].tolist()==shift_metrics_differences_to_top_ranked_for_current_algs_df["Test Set Name (ignoring fold if applicable)"].tolist()
    assert for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df["Modelling Algorithm"].tolist()==shift_metrics_differences_to_top_ranked_for_current_algs_df["Modelling Algorithm"].tolist()
    #-------------------

    #Iterate over columns for the currently considered kind of metrics (performance or uncertainty):
    for currently_considered_metric_col in pos_metrics+neg_metrics:
        for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df[currently_considered_metric_col] = shift_metrics_differences_to_top_ranked_for_current_algs_df[currently_considered_metric_col]

    return for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df

def remove_trailing_zeros_from_number_for_string_formatting(v):
    #based on this documentation: https://docs.python.org/3/library/string.html#format-specification-mini-language
    assert isinstance(v,float) or isinstance(v,int),type(v)
    return f'{v:g}'

def filter_to_only_keep_default_AD_params_results(raw_df,ad_params_col=ad_params_col,ADMethod2Param=ADMethod2Param):

    print(f'Prior to dropping non-default AD parameters results, there are {raw_df.shape[0]} results!')

    default_ad_params_str = f"k={remove_trailing_zeros_from_number_for_string_formatting(ADMethod2Param['UNC'])}_t={remove_trailing_zeros_from_number_for_string_formatting(ADMethod2Param['Tanimoto'])}"

    print(f'default_ad_params_str={default_ad_params_str}')

    filtered_df = raw_df[raw_df[ad_params_col].isin([default_ad_params_str])]
    
    filtered_df = filtered_df.drop(ad_params_col,axis=1)

    print(f'After dropping non-default AD parameters results, there are {filtered_df.shape[0]} results!')
    print(f'After dropping non-default AD parameters results, these are the columns = {filtered_df.columns.values.tolist()}')

    return filtered_df

def main():
    os.chdir('../..')
    my_path = os.getcwd()
    
    out_dir_top = os.path.sep.join([data_dir,'PublicData',"ADWA"])
    createOrReplaceDir(out_dir_top)
    
    datasets = ['Wang','CHEMBL','Morger']
    generalisations = [True, False] 
    thresholds = [0, 0.01] 

    AD_statSigThreshs = [None,0.05] #0.05,
    for dataset in datasets:
        metrics_csv, targets, algs, pos_performance_metrics,neg_performance_metrics,pos_uncertainty_metrics,neg_uncertainty_metrics,dataset_name,path_folder = define_dataset_specific_variables(dataset,out_dir_top)

        createOrReplaceDir(path_folder)

        #prepare the relevant metrics to calculate the AD methods scores
        metrics = pos_performance_metrics[:] + neg_performance_metrics [:] + pos_uncertainty_metrics[:] + neg_uncertainty_metrics[:]
        pos_metrics_total = pos_performance_metrics[:] + pos_uncertainty_metrics[:]
        neg_metrics_total = neg_performance_metrics[:] + neg_uncertainty_metrics[:]
        

        raw_df = load_raw_results(metrics_csv)

        raw_df = filter_to_only_keep_default_AD_params_results(raw_df)

        filt_raw_df = filter_raw_metrics_by_dataset_specific_targets(raw_df,targets)

        averaged_raw_df = average_over_folds_and_seed(filt_raw_df,is_shift_metrics_df=False)

        save_average_raw_metrics_df(path_folder,average_raw_metrics_df=averaged_raw_df)
                    
        average_shift_metrics_df = compute_shift_metrics_and_then_average_over_folds_and_seeds(filt_raw_df, metrics, path_folder)

        for AD_statSigThresh in AD_statSigThreshs:

            if not AD_statSigThresh is None:
                average_shift_metrics_ready_for_ranking = ensure_average_shift_metrics_corresponding_to_statistically_insignificant_p_values_or_the_wrong_sign_cannot_be_considered_winners(average_shift_metrics_df,AD_statSigThresh,dataset_name,data_dir,path_folder,metrics,pos_metrics_total,neg_metrics_total)

            else:
                average_shift_metrics_ready_for_ranking = average_shift_metrics_df

                    
            average_shift_metrics_ready_for_ranking.to_csv(os.path.sep.join([path_folder,f'average_shift_metrics_ready_for_ranking_p={AD_statSigThresh}.csv']))
                    
            Status = report_if_all_AD_methods_present_for_all_targets(average_shift_metrics_ready_for_ranking)
            assert Status

            for threshold in thresholds:
                for generalisation in generalisations:
                    if generalisation:
                        alg_winner_possibilities = ['NA']
                    else:
                        alg_winner_possibilities = algs
                    
                    
                    
                    for alg_winner in alg_winner_possibilities:

                        #generalise the AD ranking for all algorithms
                        filtered_algs = select_class_or_reg_algorithms_to_use_for_AD_ranking(generalisation, algs, alg_winner)
                    

                        #initialise the tables of AD ranking
                        list_of_normalised_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs = []
                        all_targets_ByTest_diff_table_list = []
                        list_of_raw_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs = []
                        
                          
                        for target in targets:

                            #----------------------------
                            #if target in targets_where_not_all_AD_methods_consistently_show_a_majority_of_nominal_in_domain_molecules_inside_AD:
                            #    print(f'Skipping target = {target}')
                            #    continue
                            #----------------------------
                            

                            subset_df = average_shift_metrics_ready_for_ranking[average_shift_metrics_ready_for_ranking['Endpoint'] == target]

                            ####################
                            #Placeholder, until shift-metrics can be replaced with differences to top-ranked for both kinds of metrics (performance or uncertainty):
                            if not generalisation:
                                for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df = onlyKeepBestAlgResults(subset_df,filtered_algs)
                            else:
                                for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df = subset_df.copy(deep=True)
                            for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df.columns = for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df.columns.str.replace(" Difference", "", regex=False)
                            ####################

                            check_AD_method_is_present_for_each_testset_type_for_each_algorithm(subset_df)

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

                                
                                
                                shift_metrics_differences_to_top_ranked_df = find_difference_in_average_shift_metric_to_top_ranked(subset_df,
                                                                                           pos_metrics, neg_metrics,pos_metrics_total,neg_metrics_total)

                                check_AD_method_is_present_for_each_testset_type_for_each_algorithm(shift_metrics_differences_to_top_ranked_df)

                                shift_metrics_differences_to_top_ranked_for_current_algs_df, raw_wins_by_test_df, Normalised_wins_by_test_df, out_label = computeNormalizedWins(shift_metrics_differences_to_top_ranked_df, filtered_algs,
                                                                                                                                             len(algs), pos_metrics, neg_metrics, threshold,
                                                                                                                                             AD_statSigThresh)
                                
                                for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df = replace_shift_metrics_with_difference_to_top_ranked_prior_to_writing_to_file_for_both_kinds_of_metrics(for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df,shift_metrics_differences_to_top_ranked_for_current_algs_df,pos_metrics,neg_metrics)                                                                                                           

                                print(f'alg_winner={alg_winner},generalisation={generalisation}')
                                if generalisation:
                                    assert_frame_equal(shift_metrics_differences_to_top_ranked_for_current_algs_df,shift_metrics_differences_to_top_ranked_df)  #These will not be aliases if only the results for one algorithm are considered!                                  
                                del shift_metrics_differences_to_top_ranked_df

                                if 0 ==threshold:
                                    some_shift_metric_differences_were_missing = shift_metrics_differences_to_top_ranked_for_current_algs_df[pos_metrics+neg_metrics].isnull().values.any()
                                    check_sum_normalised_winners_equal_100_perc(shift_metrics_differences_to_top_ranked_for_current_algs_df, Normalised_wins_by_test_df,
                                                                            some_shift_metric_differences_were_missing)

                                print("path_folder",path_folder)
                                #save_target_winners_analysis_function(target, out_label, Normalised_wins_by_test_df, raw_wins_by_test_df,
                                #                           shift_metrics_differences_to_top_ranked_for_current_algs_df, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)

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
                            save_dataframe_to_csv(raw_wins_by_test_df_ready_to_write_to_file, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)

                            list_of_raw_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs.append(raw_wins_by_test_df)

                            Normalised_wins_ByTest_result,Normalised_wins_ByTest_result_ready_to_write_to_file = get_combined_winners_counts_ready_to_write_to_file(perf_winners_count_df=Normalised_perf_score_by_test_df,uncert_winners_count_df=Normalised_uncert_score_by_test_df)
                            
                            path = target + "_all_Norm_wins_count_"
                            save_dataframe_to_csv(Normalised_wins_ByTest_result_ready_to_write_to_file, path, path_folder,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)

                            list_of_normalised_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs.append(Normalised_wins_ByTest_result)

                            ##########################################################################
                            #Append the dataframe containing the ranking of the AD methods
                            all_targets_ByTest_diff_table_list.append(for_both_kinds_of_metrics_shift_metrics_differences_to_top_ranked_for_current_algs_df)

                        #Full diff table over the different targets
                        all_targets_ByTest_diff_table_df = pd.concat(all_targets_ByTest_diff_table_list, axis=0, ignore_index=False)

                        save_all_targets_raw_and_normalised_winners_counts(all_targets_ByTest_diff_table_df,
                                                                       list_of_raw_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs,
                                                                       list_of_normalised_count_of_winners_for_all_targets_per_testSets_and_aggregated_across_metrics_and_filtered_algs,targets,
                                                                       path_folder,out_label,generalisation,threshold,AD_statSigThresh,dataset,alg_winner)
    print("Finish everything!")

if __name__ == '__main__':

    sys.exit(main())
