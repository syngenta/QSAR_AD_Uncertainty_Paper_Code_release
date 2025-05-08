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
#Copyright (c) 2022-2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#The lines of code incorporated into the function prepare_ChEMBL_train_plus_temporal_splits(...) were taken and adapted by Zied Hosni,followed by Richard Marchese Robinson, as part of a Syngenta fundeded collaboration with the University of Sheffield
#These lines of code were adapted from: https://github.com/volkamerlab/CPrecalibration_manuscript_SI/blob/main/code/1_continuous_calibration_example.ipynb
#Provenance, copyright and license information for the original lines of code is provided below
#############################################################
# MIT License

# Copyright (c) 2021 Volkamer Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################
import os
import pandas as pd

from common_globals_classification_scripts import chembl_ids_col,chembl_smiles_col,chembl_act_class_col,chembl_subsets_of_interest

def merge_smiles_and_activities(X_np_array_with_ids_smiles_year,Y_contains_activities,x_column_names_lost_when_converted_to_np_array):
    df = pd.DataFrame(X_np_array_with_ids_smiles_year,columns=x_column_names_lost_when_converted_to_np_array)
    
    df.insert(1,chembl_act_class_col,Y_contains_activities,allow_duplicates=True)
    
    df=df.astype({'year':float}) #Why was this sometimes float and sometimes int previously?
    
    return df

def get_dataframes_for_subsets_of_interest(subsets_of_interest,X_train,y_train,X_update1,y_update1,X_update2,y_update2,X_holdout,y_holdout,x_column_names_lost_when_converted_to_np_array):
    subset2df = {}
    
    for subset in subsets_of_interest:
        if 'train' == subset:
            X_np_array_with_ids_smiles_year = X_train
            Y_contains_activities = y_train
        elif 'update1' == subset:
            X_np_array_with_ids_smiles_year = X_update1
            Y_contains_activities = y_update1
        elif 'update2' == subset:
            X_np_array_with_ids_smiles_year = X_update2
            Y_contains_activities = y_update2
        elif 'holdout' == subset:
            X_np_array_with_ids_smiles_year = X_holdout
            Y_contains_activities = y_holdout
        else:
            raise Exception(f'Unrecognised subset={subset}')
        
        subset2df[subset] = merge_smiles_and_activities(X_np_array_with_ids_smiles_year,Y_contains_activities,x_column_names_lost_when_converted_to_np_array)
    
    return subset2df

def prepare_ChEMBL_train_plus_temporal_splits(endpoint,data_chembl_path,subsets_of_interest=chembl_subsets_of_interest,chembl_ids_col=chembl_ids_col,chembl_smiles_col=chembl_smiles_col):
    data_path = os.path.join(data_chembl_path, f"{endpoint}_chembio_normalizedDesc.csv")
    time_split_threshold_path = os.path.join(data_chembl_path, "data_size_chembio_chembl.csv")
    data = pd.read_csv(data_path)
    
    ########################
    #RMR copied from 1_continuous_calibration_example.ipynb:
    # The publication date is required for temporal data splitting.
    data.dropna(subset=["year"], inplace=True)
    #########################
    
    y = data[f"{endpoint}_bioactivity"].values
    
    columns = [chembl_ids_col,chembl_smiles_col,'year'] #RMR: We know the limited number of columns we want to keep, so can specify explicitly.

    X = data[columns].values
    
    years = data["year"].values

    splits_df = pd.read_csv(time_split_threshold_path, index_col=0,
                            usecols=["chembl_id", "train_thresh", "update1_thresh", "update2_thresh"])

    # splits_df.index
    thresholds = splits_df["train_thresh"][endpoint], splits_df["update1_thresh"][endpoint], \
                 splits_df["update2_thresh"][endpoint]

    # Collect the respective indices based on the threshold
    mask_train = years <= thresholds[0]
    mask_update1 = (years > thresholds[0]) & (years <= thresholds[1])
    mask_update2 = (years > thresholds[1]) & (years <= thresholds[2])
    mask_holdout = years > thresholds[2]

    # Split the data accordingly
    X_train, y_train = X[mask_train], y[mask_train]
    X_update1, y_update1 = X[mask_update1], y[mask_update1]
    X_update2, y_update2 = X[mask_update2], y[mask_update2]
    X_holdout, y_holdout = X[mask_holdout], y[mask_holdout]
    
    #RMR:
    subset2df = get_dataframes_for_subsets_of_interest(subsets_of_interest,X_train,y_train,X_update1,y_update1,X_update2,y_update2,X_holdout,y_holdout,x_column_names_lost_when_converted_to_np_array=columns)
    
    return subset2df
