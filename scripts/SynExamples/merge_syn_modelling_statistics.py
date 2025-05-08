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
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
pkg_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict,reportSubDirs,createOrReplaceDir,load_from_pkl_file,convertDefaultDictDictIntoDataFrame
#----------------------------------------------------------------------------
top_scripts_dir = os.path.dirname(dir_of_this_script)
sys.path.append(top_scripts_dir)
from consistent_parameters_for_all_modelling_runs import raw2prettyMetricNames,regression_stats_in_desired_order,classification_stats_in_desired_order,stats_metadata_cols_in_desired_order
from consistent_parameters_for_all_modelling_runs import endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col
from consistent_parameters_for_all_modelling_runs import ad_params_col
from PublicDataModelling.general_purpose.merge_exemplar_modelling_stats import get_all_raw_stats_csvs,prettify_dataset_specific_merged_df,remove_duplicate_rows,get_final_merged_stats_dfs
#------------
top_res_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'SyngentaData'])
out_dir = os.path.sep.join([top_res_dir,'Merged_Stats'])
regression_dataset_names = ['logP']
classification_dataset_names = ['DT50']

#################
#Syngenta data specific adaptations of merge statistics functions:

def get_df_with_metadata(csv_file,stats_metadata_cols_in_desired_order,dataset,regression_dataset_names):
    
    df = pd.read_csv(csv_file)

    ad_params_str = 'Default'

    base_name_csv = os.path.basename(csv_file)
        
    endpoint = base_name_csv.split('e=')[1].split('_')[0]

    test_name_ignoring_fold = os.path.basename(os.path.dirname(csv_file)).split('_')[0]

    fold = 'N/A'

    seed = os.path.basename(os.path.dirname(csv_file)).split('_s=')[1]

    algorithm = base_name_csv.split('m=')[1].split('_')[0]


    ad_method = base_name_csv.split('ad=')[1].split('_')[0]


    

    #---------------------------
    try: #Debug
        print(f'Parsing {csv_file}')
        assert (2 == df.shape[0]) or (3 == df.shape[0]),f'csv_file={csv_file} has {df.shape[0]} rows!' #2 means statistics inside vs. outside the domain. 3 means we also include the results without AD splitting, i.e. the AD method name is merely a placeholder!
        if 2 == df.shape[0]:
            assert ['Inside','Outside'] == df[ad_subset_col].tolist(),df[ad_subset_col].tolist()
        else:
            assert ['All','Inside','Outside'] == df[ad_subset_col].tolist(),df[ad_subset_col].tolist()
    except AssertionError as err:
        print(f'Problem={err}')
    #--------------------------
    
    ##############################
    metadata_dict = {}
    metadata_dict[ad_params_col] = ad_params_str
    metadata_dict[endpoint_col] = endpoint
    metadata_dict[test_set_type_col] = test_name_ignoring_fold
    metadata_dict[fold_col] = fold
    metadata_dict[rnd_seed_col] = seed
    metadata_dict[alg_col] = algorithm
    metadata_dict[ad_col] = ad_method
    ##############################
    #-----------------------------
    assert all([key in stats_metadata_cols_in_desired_order for key in metadata_dict.keys()])
    #-----------------------------

    for metadata_name in metadata_dict.keys():
        if not (ad_col == metadata_name and ['All','Inside','Outside'] == df[ad_subset_col].tolist()):

            no_repeats = df.shape[0]

            df.insert(0,metadata_name,[metadata_dict[metadata_name]]*no_repeats,allow_duplicates=False)
        else:
            df.insert(0,metadata_name,['N/A']+[metadata_dict[metadata_name]]*2,allow_duplicates=False)

    return df


def merge_all_raw_stats_with_metadata(all_raw_stats_csvs,stats_metadata_cols_in_desired_order,dataset,regression_dataset_names):
    all_dfs_ready_to_merge = [get_df_with_metadata(csv_file,stats_metadata_cols_in_desired_order,dataset,regression_dataset_names) for csv_file in all_raw_stats_csvs]

    dataset_specific_merged_df = pd.concat(all_dfs_ready_to_merge,axis=0,ignore_index=True)

    return dataset_specific_merged_df

def counts_rows_per_ds(ds_to_merged_df):
    ds_to_row_count = {}
    for ds in ds_to_merged_df.keys():
        ds_to_row_count[ds] = ds_to_merged_df[ds].shape[0]
    
    return ds_to_row_count

def check_no_duplicate_rows(ds_to_merged_df):

    prior_to_remove_duplicate_rows_ds_to_row_count = counts_rows_per_ds(ds_to_merged_df)

    ds_to_merged_df = remove_duplicate_rows(ds_to_merged_df)

    after_remove_duplicate_rows_ds_to_row_count = counts_rows_per_ds(ds_to_merged_df)

    assert prior_to_remove_duplicate_rows_ds_to_row_count == after_remove_duplicate_rows_ds_to_row_count

####################

def main():
    print('THE START')

    createOrReplaceDir(out_dir)

    ds_to_merged_df = {}

    for dataset in regression_dataset_names+classification_dataset_names:
        #--------------------------------------
        if dataset in classification_dataset_names:
            top_dir_with_stats = os.path.sep.join([top_res_dir,f'{dataset}_Updates','Calc'])
        elif dataset in regression_dataset_names:
            top_dir_with_stats = os.path.sep.join([top_res_dir,f'{dataset}_updates','Calc'])
        else:
            raise Exception(f'dataset={dataset}')
        #---------------------------------------

        all_raw_stats_csvs = get_all_raw_stats_csvs(top_dir_with_stats,dataset,regression_dataset_names+classification_dataset_names)

        dataset_specific_merged_df = merge_all_raw_stats_with_metadata(all_raw_stats_csvs,stats_metadata_cols_in_desired_order,dataset,regression_dataset_names)

        prettified_dataset_specific_merged_df = prettify_dataset_specific_merged_df(dataset_specific_merged_df,raw2prettyMetricNames,stats_metadata_cols_in_desired_order,regression_stats_in_desired_order,classification_stats_in_desired_order,dataset,regression_dataset_names)

        ds_to_merged_df[dataset] = prettified_dataset_specific_merged_df
        
    check_no_duplicate_rows(ds_to_merged_df)

    classification_merged_df, regression_merged_df = get_final_merged_stats_dfs(ds_to_merged_df,regression_dataset_names)

    classification_merged_df.to_csv(os.path.sep.join([out_dir,f'SYN_{classification_dataset_names[0]}_Classification_Stats.csv']),index=False)

    regression_merged_df.to_csv(os.path.sep.join([out_dir,f'SYN_{regression_dataset_names[0]}_Regression_Stats.csv']),index=False)
    
    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())
