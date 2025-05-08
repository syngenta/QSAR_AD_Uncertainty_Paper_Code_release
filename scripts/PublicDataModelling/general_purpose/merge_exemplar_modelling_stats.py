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
#Copyright (c) 2023-2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#############################################################
import os,sys,glob,re
import pandas as pd
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
#---------------------------------------------------------------------------
#Need to not fail on these imports if re-using functions in SYN merge statistics script!
sys.path.append(dir_of_this_script)
from common_globals import top_class_or_reg_ds_dirs,regression_dataset_names
from extract_ad_params_from_abs_file_name import get_ad_params
#----------------------------------------------------------------------------
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict,reportSubDirs,createOrReplaceDir,load_from_pkl_file,convertDefaultDictDictIntoDataFrame
#----------------------------------------------------------------------------
top_scripts_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
sys.path.append(top_scripts_dir)
from consistent_parameters_for_all_modelling_runs import raw2prettyMetricNames,regression_stats_in_desired_order,classification_stats_in_desired_order,stats_metadata_cols_in_desired_order
from consistent_parameters_for_all_modelling_runs import endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col
from consistent_parameters_for_all_modelling_runs import ad_params_col
#-------------
out_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData','ModellingResSummary'])

def get_raw_stats_csvs_in_this_dir(dir_):
    return glob.glob(os.path.sep.join([dir_,'*_Statistics.csv']))

def get_all_raw_stats_csvs(top_dir_with_stats,dataset,regression_dataset_names):

    if not dataset in regression_dataset_names:
        ad_params_specific_subdirs = reportSubDirs(top_dir_with_stats)
        modelling_subdirs = ad_params_specific_subdirs
    else:
        endpoint_and_ad_params_specific_subdirs = reportSubDirs(top_dir_with_stats)
        modelling_subdirs = endpoint_and_ad_params_specific_subdirs
    
    all_raw_stats_csvs = []
    for dir_ in modelling_subdirs:
        all_raw_stats_csvs += get_raw_stats_csvs_in_this_dir(dir_)


    return all_raw_stats_csvs

def get_endpoint(base_name_csv):

    endpoint = base_name_csv.split('t=')[1].split('_')[0]

    return endpoint

def get_test_name_ignoring_fold(base_name_csv,dataset,regression_dataset_names,endpoint):

    test_name = base_name_csv.split(f'{endpoint}_')[1].split('_FPs')[0]

    test_name_ignoring_fold = re.sub('(_f\=[0-9]_test)','',test_name)

    return test_name_ignoring_fold

def get_fold(base_name_csv,dataset,regression_dataset_names):
    
    if not dataset in regression_dataset_names:
        fold = 'N/A'
    else:
        fold = base_name_csv.split('f=')[1].split('_')[0]

    return fold

def get_seed(base_name_csv):

    seed = base_name_csv.split('s=')[1].split('_')[0]

    return seed

def get_algorithm(base_name_csv):

    algorithm = base_name_csv.split('m=')[1].split('_')[0]

    return algorithm 

def get_ad_method(base_name_csv):

    ad_method = base_name_csv.split('ad=')[1].split('_')[0]

    return ad_method

def get_ad_params_str(csv_file):
    k_val,tanimoto_distance_threshold = get_ad_params(csv_file)
    ad_params_str = f"k={k_val}_t={tanimoto_distance_threshold}"
    return ad_params_str

def get_df_with_metadata(csv_file,stats_metadata_cols_in_desired_order,dataset,regression_dataset_names):
    
    df = pd.read_csv(csv_file)

    ad_params_str = get_ad_params_str(csv_file)

    base_name_csv = os.path.basename(csv_file)
        
    endpoint = get_endpoint(base_name_csv)

    test_name_ignoring_fold = get_test_name_ignoring_fold(base_name_csv,dataset,regression_dataset_names,endpoint)

    fold = get_fold(base_name_csv,dataset,regression_dataset_names)

    seed = get_seed(base_name_csv)

    algorithm = get_algorithm(base_name_csv)

    ad_method = get_ad_method(base_name_csv)

    

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

def check_columns_are_all_consistent(all_dfs_ready_to_merge):
    all_cols_list_strs = [';'.join(df.columns.values.tolist()) for df in all_dfs_ready_to_merge]

    assert 1 == len(set(all_cols_list_strs)),list(set(all_cols_list_strs))

def merge_all_raw_stats_with_metadata(all_raw_stats_csvs,stats_metadata_cols_in_desired_order,dataset,regression_dataset_names):
    all_dfs_ready_to_merge = [get_df_with_metadata(csv_file,stats_metadata_cols_in_desired_order,dataset,regression_dataset_names) for csv_file in all_raw_stats_csvs]

    #check_columns_are_all_consistent(all_dfs_ready_to_merge) #Actually, observed for two dataframes that concatenation works even if the columns are in a different order

    dataset_specific_merged_df = pd.concat(all_dfs_ready_to_merge,axis=0,ignore_index=True)

    return dataset_specific_merged_df

def prettify_dataset_specific_merged_df(dataset_specific_merged_df,raw2prettyMetricNames,stats_metadata_cols_in_desired_order,regression_stats_in_desired_order,classification_stats_in_desired_order,dataset,regression_dataset_names):
    
    prettified_dataset_specific_merged_df = dataset_specific_merged_df.rename(mapper=raw2prettyMetricNames,axis=1)

    if dataset in regression_dataset_names:
        cols_in_desired_order = stats_metadata_cols_in_desired_order[:]+regression_stats_in_desired_order[:]
    else:
        cols_in_desired_order = stats_metadata_cols_in_desired_order[:]+classification_stats_in_desired_order[:]
    
    prettified_dataset_specific_merged_df = prettified_dataset_specific_merged_df[cols_in_desired_order]

    return prettified_dataset_specific_merged_df

def get_final_merged_stats_dfs(ds_to_merged_df,regression_dataset_names):

    assert 1 == len(regression_dataset_names)
    
    regression_merged_df = ds_to_merged_df[regression_dataset_names[0]]

    ds_specific_classification_merged_dfs = [ds_to_merged_df[ds] for ds in ds_to_merged_df.keys() if not ds in regression_dataset_names]

    classification_merged_df = pd.concat(ds_specific_classification_merged_dfs,axis=0,ignore_index=True)

    return classification_merged_df, regression_merged_df

def remove_duplicate_rows(ds_to_merged_df):

    for ds in ds_to_merged_df.keys():
        ds_to_merged_df[ds] = ds_to_merged_df[ds].drop_duplicates(inplace=False,ignore_index=True)
    
    return ds_to_merged_df


def main():
    print('THE START')

    createOrReplaceDir(out_dir)

    ds_to_merged_df = {}

    for dataset in top_class_or_reg_ds_dirs.keys():
        top_dir_with_stats = os.path.sep.join([top_class_or_reg_ds_dirs[dataset],'Modelling'])
        
        all_raw_stats_csvs = get_all_raw_stats_csvs(top_dir_with_stats,dataset,regression_dataset_names)

        dataset_specific_merged_df = merge_all_raw_stats_with_metadata(all_raw_stats_csvs,stats_metadata_cols_in_desired_order,dataset,regression_dataset_names)

        prettified_dataset_specific_merged_df = prettify_dataset_specific_merged_df(dataset_specific_merged_df,raw2prettyMetricNames,stats_metadata_cols_in_desired_order,regression_stats_in_desired_order,classification_stats_in_desired_order,dataset,regression_dataset_names)

        ds_to_merged_df[dataset] = prettified_dataset_specific_merged_df
    
    ds_to_merged_df = remove_duplicate_rows(ds_to_merged_df)

    classification_merged_df, regression_merged_df = get_final_merged_stats_dfs(ds_to_merged_df,regression_dataset_names)

    classification_merged_df.to_csv(os.path.sep.join([out_dir,'Exemplar_Endpoints_Classification_Stats.csv']),index=False)

    regression_merged_df.to_csv(os.path.sep.join([out_dir,'Exemplar_Endpoints_Regression_Stats.csv']),index=False)

    

    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())
