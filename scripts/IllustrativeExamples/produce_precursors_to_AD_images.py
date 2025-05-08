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
##############################
#Copright (c) 2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
###############################
import os,  sys,  time,  pickle,shutil,itertools,re,glob
from collections import defaultdict
import pandas as pd
import numpy as np
from textwrap import wrap
#from define_plot_settings import *
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.qsar_AD import dk_NN_thresholds as dk_NN_AD
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.utils.basic_utils import load_from_pkl_file,  create_pkl_file
#----------------------------------------------------------------------------
##################################
ADMethod2ParameterValues = {}

tanimoto_distance_thresholds = [(i/10) for i in range(1,10)]+[0.35]
k_values_for_multiple_k_kNN_based_methods = range(1,7)


ADMethod2ParameterValues['dkNN'] = 2
ADMethod2ParameterValues['UNC'] = 2
ADMethod2ParameterValues['RDN'] = 2
#################################
distance_metric = 'manhattan'
id_col = 'ID'
scale = False
descs_no = 2
#only_consider_k_neighbours_for_final_threshold_calculations = False #This is the default!
##################################
rand_seed = 42 #only RND would depend upon this
#################################


def get_training_set_thresholds(X_train_with_ids,ad_method,distance_metric,id_col,scale,descs_no):
    

    if 'dkNN' == ad_method:
        k_val = ADMethod2ParameterValues[ad_method]

        dk_NN_thresholds_instance = dk_NN_AD.dk_NN_thresholds(train_df=X_train_with_ids, id_col=id_col, k=k_val,distance_metric=distance_metric, scale=scale,debug=False)
        
        #dk_NN_thresholds_instance.debug=True

        dk_NN_thresholds_instance.getInitialTrainingSetThresholds()
        
        dk_NN_thresholds_instance.updateZeroValuedTrainingSetThresholds()

        train_ids_matched_to_thresholds = dk_NN_thresholds_instance.TrainIdsMatchedToThresholds
    else:
        raise Exception(f'This has not been implemented for AD method = {ad_method}')
    
    return train_ids_matched_to_thresholds

def main():
    print('THE START')
    
    for dataset_name in glob.glob('*_input.csv'):
        X_train_with_ids = pd.read_csv(dataset_name)
        
        for ad_method in ['dkNN']:
            print(f'Parsing {dataset_name} with AD method {ad_method}')
            print(X_train_with_ids.columns.values.tolist()) #debugging
            
            train_ids_matched_to_thresholds = get_training_set_thresholds(X_train_with_ids,ad_method,distance_metric,id_col,scale,descs_no)

            out_file = re.sub('(\.csv$)',f'_{ad_method}_train_thresholds.csv',dataset_name)
            assert not out_file == dataset_name,dataset_name

            thresholds_in_correct_order = [train_ids_matched_to_thresholds[id_] for id_ in X_train_with_ids[id_col].tolist()]

            out_df = X_train_with_ids.copy(deep=True)

            out_df.insert(out_df.shape[1],'Threshold',thresholds_in_correct_order,allow_duplicates=False)

            out_df.to_csv(out_file,index=False)
            
            
            
        
        
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
