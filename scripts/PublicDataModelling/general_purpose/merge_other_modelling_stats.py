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
#----------------------------------------------------------------------------
from common_globals import top_class_or_reg_ds_dirs,regression_dataset_names,classification_dataset_names
#----------------------------------------------------------------------------
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict,reportSubDirs,createOrReplaceDir,load_from_pkl_file,convertDefaultDictDictIntoDataFrame
#----------------------------------------------------------------------------
top_scripts_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
sys.path.append(top_scripts_dir)
from consistent_parameters_for_all_modelling_runs import raw2prettyMetricNames,regression_stats_in_desired_order,classification_stats_in_desired_order,stats_metadata_cols_in_desired_order
#-------------
out_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData','OtherMergedStats'])
#--------------
from merge_exemplar_modelling_stats import get_all_raw_stats_csvs,merge_all_raw_stats_with_metadata,prettify_dataset_specific_merged_df,get_final_merged_stats_dfs,remove_duplicate_rows
#---------------


def main():
    print('THE START')

    createOrReplaceDir(out_dir)

    ds_to_merged_df = {}

    for dataset in top_class_or_reg_ds_dirs.keys():
        top_dir_with_stats = os.path.sep.join([top_class_or_reg_ds_dirs[dataset],'Modelling.2'])
        
        all_raw_stats_csvs = get_all_raw_stats_csvs(top_dir_with_stats,dataset,regression_dataset_names+classification_dataset_names)

        dataset_specific_merged_df = merge_all_raw_stats_with_metadata(all_raw_stats_csvs,stats_metadata_cols_in_desired_order,dataset,regression_dataset_names)

        prettified_dataset_specific_merged_df = prettify_dataset_specific_merged_df(dataset_specific_merged_df,raw2prettyMetricNames,stats_metadata_cols_in_desired_order,regression_stats_in_desired_order,classification_stats_in_desired_order,dataset,regression_dataset_names)

        ds_to_merged_df[dataset] = prettified_dataset_specific_merged_df
    
    ds_to_merged_df = remove_duplicate_rows(ds_to_merged_df)

    classification_merged_df, regression_merged_df = get_final_merged_stats_dfs(ds_to_merged_df,regression_dataset_names)

    classification_merged_df.to_csv(os.path.sep.join([out_dir,'Other_Endpoints_Classification_Stats.csv']),index=False)

    regression_merged_df.to_csv(os.path.sep.join([out_dir,'Other_Endpoints_Regression_Stats.csv']),index=False)

    

    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())
