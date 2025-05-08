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
#Copright (c) 2022-2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#Contact zied.hosni [at] syngenta.com
###############################
import os,  sys,  time,  pickle,shutil,itertools
from collections import defaultdict
import pandas as pd
import numpy as np
from textwrap import wrap
##################################
from common_globals import consistent_id_col,  consistent_act_col,  consistent_smiles_col,  consistent_inchi_col
from common_globals import top_dir_of_public_data_scripts,  pkg_dir,  top_class_or_reg_ds_dirs,  regression_dataset_names,  classification_dataset_names,  ds_matched_to_ep_list,  get_endpoint_type
from common_globals import all_Wang_endpoints,  wang_raw_smiles_col,  wang_raw_act_col,  wang_ids_col,  wang_test_set_names,  no_wang_outer_folds
from common_globals import all_Tox21_endpoints,  all_ChEMBL_endpoints,  chembl_ids_col,  chembl_smiles_col,  chembl_act_class_col,  chembl_subsets_of_interest,  tox21_subsets_of_interest
from common_globals import class_ds_name_to_train_subset_name
from common_globals import stand_mol_col,  fp_col,  stand_mol_inchi_col,  stand_mol_smiles_col,  subset_name_col
from common_globals import get_rand_subset_train_test_suffix,  rand_split_seed,  no_train_test_splits
from common_globals import ds_matched_to_ep_to_all_train_test_pairs
from common_globals import ds_matched_to_exemplar_ep_list
from common_globals import load_ds_from_pkl_files_ready_for_modelling_and_ad_calculations
from common_globals import all_global_random_seed_opts
from common_globals import class_1,  class_0
from common_globals import rnd_test_suffix, wang_test_set_names
from common_globals import class_ds_name_to_test_subset_names
##################################
from define_plot_settings import *
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.ML_utils import getTestYFromTestIds as getSubsetYFromSubsetIds
from modelsADuncertaintyPkg.utils.basic_utils import check_findDups,  convertDefaultDictDictIntoDataFrame,  neverEndingDefaultDict,  findDups
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import findInsideOutsideADTestIds,  getInsideOutsideADSubsets
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.utils.ML_utils import checkTrainTestHaveUniqueDistinctIDs,  makeIDsNumeric,  prepareInputsForModellingAndUncertainty
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import getInputRequiredForModellingAndAD,  getADSubsetTestIDsInOrder
from modelsADuncertaintyPkg.utils.basic_utils import load_from_pkl_file,  create_pkl_file
#----------------------------------------------------------------------------
##################################
ADMethod2ParameterValues = {}

tanimoto_distance_thresholds = [(i/10) for i in range(1,10)]+[0.35]
k_values_for_multiple_k_kNN_based_methods = range(1,7)


ADMethod2ParameterValues['Tanimoto'] = tanimoto_distance_thresholds #N.B. 'Tanimoto' = '1-kNN'
ADMethod2ParameterValues['dkNN'] = k_values_for_multiple_k_kNN_based_methods #N.B. 'dk-NN' was originally referred to as 'dkNN'
ADMethod2ParameterValues['UNC'] = k_values_for_multiple_k_kNN_based_methods #N.B. 'nUNC' was originally referred to as the 'UNC' approach
ADMethod2ParameterValues['RDN'] = k_values_for_multiple_k_kNN_based_methods
##################################
random_seed_opts_to_consider_here = all_global_random_seed_opts[:1] #Only RDN should depend upon this.
#################################

def get_inside_AD_perc(test_id_status_dict,  test_ids,  dataset_name,  endpoint,  test_set_label):
    
    inside_test_ids = getADSubsetTestIDsInOrder('Inside',  test_id_status_dict,  test_ids)
    
    #---------------------
    all_test_ids = getADSubsetTestIDsInOrder('All',  test_id_status_dict,  test_ids)
    assert len(test_ids)==len(all_test_ids),  f"Are some IDs not assigned an AD status? - dataset_name={dataset_name},  endpoint={endpoint},  test_set_label={test_set_label},   len(test_ids)={len(test_ids)},  len(all_test_ids)={len(all_test_ids)}"
    #---------------------
    
    inside_ad_perc = round(100*len(inside_test_ids)/len(test_ids),  0)
    
    
    return inside_ad_perc

def get_all_inside_ad_perc_vals(dataset_name, dir_with_files_to_parse,  out_dir,  ds_matched_to_exemplar_ep_list,  ds_matched_to_ep_to_all_train_test_pairs,  ADMethod2ParameterValues,  random_seed_opts_to_consider_here):
    inside_ad_perc_vals_pkl_file = os.path.sep.join([out_dir,  'InsideADPercVals.pkl'])
    
    dict_of_all_ad_perc_vals = defaultdict(list)
        
    for endpoint in ds_matched_to_exemplar_ep_list[dataset_name]:
        for train_test_label_pair in ds_matched_to_ep_to_all_train_test_pairs[dataset_name][endpoint]:
            
            train_set_label = train_test_label_pair[0]
            test_set_label = train_test_label_pair[1]
            
            fps_train,  fps_test,  test_ids,  X_train_and_ids_df,  X_test_and_ids_df,  train_y,  test_y,  train_ids =load_ds_from_pkl_files_ready_for_modelling_and_ad_calculations(dir_with_files_to_parse,  train_set_label,  test_set_label)
            
            #--------------------
            orig_test_ids = X_test_and_ids_df[consistent_id_col].tolist()
            orig_test_ids.sort()
            test_ids.sort()
            assert test_ids==orig_test_ids
            del test_ids
            #--------------------
            
            X_train_and_ids_df = makeIDsNumeric(df=X_train_and_ids_df,  id_col=consistent_id_col)
            
            X_test_and_ids_df = makeIDsNumeric(df=X_test_and_ids_df,  id_col=consistent_id_col)
            
            test_ids = X_test_and_ids_df[consistent_id_col].tolist() #This is needed for (at least) d1NN ("Tanimoto") AD method.
            
            for AD_method_name in ADMethod2ParameterValues.keys():
                for ad_parameter in ADMethod2ParameterValues[AD_method_name]:
                    for rand_seed in random_seed_opts_to_consider_here:
                        print('='*20)
                        print('Running the following inside AD percentage calculation:')
                        print(f'Dataset={dataset_name}')
                        print(f'Endpoint={endpoint}')
                        print(f'Splitting approach={AD_method_name}')
                        print(f'Threshold"={ad_parameter}')
                        print(f'test set={test_set_label}')
                        print(f'rand_seed={rand_seed}')
                        print('='*20)
                        
                        start = time.time()
                        
                        if 'regression' == get_endpoint_type(dataset_name,  regression_dataset_names,  classification_dataset_names):
                            
                            test_id_status_dict = findInsideOutsideADTestIds(X_train=X_train_and_ids_df,   X_test=X_test_and_ids_df,   fps_train=fps_train,   fps_test=fps_test,   threshold=ad_parameter,   id_col=consistent_id_col,  rand_seed=rand_seed,   endpoint_col=consistent_act_col,   AD_method_name=AD_method_name,  test_ids=test_ids,   y_train=train_y,   y_test=None,   regression=True,  class_1_label=None,  class_0_label=None)
                        elif 'classification' == get_endpoint_type(dataset_name,  regression_dataset_names,  classification_dataset_names):
                            
                            test_id_status_dict = findInsideOutsideADTestIds(X_train=X_train_and_ids_df,   X_test=X_test_and_ids_df,   fps_train=fps_train,   fps_test=fps_test,   threshold=ad_parameter,   id_col=consistent_id_col,  rand_seed=rand_seed,   endpoint_col=consistent_act_col,   AD_method_name=AD_method_name,  test_ids=test_ids,   y_train=train_y,   y_test=None,   regression=False,  class_1_label=class_1,  class_0_label=class_0)
                        else:
                            raise Exception(f'Unexpected endpoint type with dataset_name={dataset_name}: {get_endpoint_type(dataset_name,  regression_dataset_names,  classification_dataset_names)}')
                        
                        inside_ad_perc = get_inside_AD_perc(test_id_status_dict,  test_ids,  dataset_name,  endpoint,  test_set_label)
                        
                        #####################
                        #Copying key names directly from Zied's old Venn_Abers_class_main.py:
                        dict_of_all_ad_perc_vals['Dataset'].append(dataset_name)
                        dict_of_all_ad_perc_vals['Endpoint'].append(endpoint)
                        dict_of_all_ad_perc_vals['Splitting approach'].append(AD_method_name)
                        dict_of_all_ad_perc_vals["Threshold"].append(ad_parameter)
                        dict_of_all_ad_perc_vals["Value"].append(inside_ad_perc)
                        dict_of_all_ad_perc_vals['test set'].append(test_set_label)
                        #####################
                        
                        print('='*20)
                        print('Adding the following results to the pickle file:')
                        print(f'Dataset={dataset_name}')
                        print(f'Endpoint={endpoint}')
                        print(f'Splitting approach={AD_method_name}')
                        print(f'Threshold"={ad_parameter}')
                        print(f'Value={inside_ad_perc}')
                        print(f'test set={test_set_label}')
                        print(f'rand_seed={rand_seed}')
                        print('='*20)
                        
                        create_pkl_file(pkl_file=inside_ad_perc_vals_pkl_file,  obj=dict_of_all_ad_perc_vals) #Keep saving this to help with debugging in case of an intermediate crash
                        
                        end = time.time()
                        
                        print(f'Time taken = {(end-start)/60} minutes')
    
    return dict_of_all_ad_perc_vals

def get_all_relevant_generic_test_set_names(dataset_name,rnd_test_suffix, wang_test_set_names, class_ds_name_to_test_subset_names):
    if not dataset_name in class_ds_name_to_test_subset_names.keys():
        return [n for n in wang_test_set_names]
    else:
        return [rnd_test_suffix]+[test_label for test_label in class_ds_name_to_test_subset_names[dataset_name]]

def get_generic_test_set_name_from_test_set_label(test_set_name,ds_specific_generic_test_set_names):
    
    
    all_generic_test_set_names = ds_specific_generic_test_set_names
    
    all_matching_generic_test_set_names = []
    
    for generic_test_name in all_generic_test_set_names:
        if generic_test_name in test_set_name:
            all_matching_generic_test_set_names.append(generic_test_name)
    
    assert 1 == len(all_matching_generic_test_set_names),f'test_set_name={test_set_name},all_matching_generic_test_set_names={all_matching_generic_test_set_names}'
    
    return all_matching_generic_test_set_names[0]

def make_generic_test_set_name_look_nice(generic_test_set_name,rnd_test_suffix,wang_test_set_names):
    #These names must be consistent with def map_test_names_onto_plot_args(...)

    if rnd_test_suffix == generic_test_set_name:
        return 'Random'
    elif generic_test_set_name in wang_test_set_names:
        return generic_test_set_name.upper()
    else:
        return generic_test_set_name

def map_test_names_onto_plot_args(test_set_name):
    x = test_set_name
    
    dict_of_maps = {}
    
    dict_of_maps['Random'] = 'g'
    dict_of_maps['Tox21Score'] = 'r'
    dict_of_maps['Tox21Test'] = 'b'
    #--------------------
    dict_of_maps['Random'] = 'g'
    dict_of_maps['update1'] = 'r'
    dict_of_maps['holdout'] = 'b'
    #--------------------
    dict_of_maps['IVIT'] = 'g'
    dict_of_maps['IVOT'] = 'r'
    #---------------------
    
    return dict_of_maps[x]
    


def replace_test_set_name_with_a_nice_looking_generic_test_set_name(row,rnd_test_suffix,wang_test_set_names,ds_specific_generic_test_set_names):

    test_set_name = row['test set']

    generic_test_set_name = get_generic_test_set_name_from_test_set_label(test_set_name,ds_specific_generic_test_set_names)

    nice_looking_generic_test_set_name = make_generic_test_set_name_look_nice(generic_test_set_name,rnd_test_suffix,wang_test_set_names)


    row['test set'] = nice_looking_generic_test_set_name

    return row


def record_inside_ad_perc_vals_as_csv(out_dir,inAD_perc_df):
    inside_ad_perc_csv = os.path.sep.join([out_dir,  'InsideADPercVals.DataFrame.csv'])
    inAD_perc_df.to_csv(inside_ad_perc_csv,index=False)

def plot_inside_ad_perc_vs_parameter_vals(inAD_perc_df, out_dir, dataset_name,rnd_test_suffix, wang_test_set_names, class_ds_name_to_test_subset_names,not_multiple_k_kNN_based_methods=['Tanimoto']):
    
    ds_specific_generic_test_set_names = get_all_relevant_generic_test_set_names(dataset_name,rnd_test_suffix, wang_test_set_names, class_ds_name_to_test_subset_names)
    
    inAD_perc_df = inAD_perc_df.apply(replace_test_set_name_with_a_nice_looking_generic_test_set_name,axis=1,args=(rnd_test_suffix,wang_test_set_names,ds_specific_generic_test_set_names))

    ds_specific_generic_test_set_names = [make_generic_test_set_name_look_nice(generic_test_set_name,rnd_test_suffix,wang_test_set_names) for generic_test_set_name in ds_specific_generic_test_set_names]

    ################################
    #The following was copied and adapted from Zied's Venn_Abers_class_main.py:
    ################################
    
    # save the inAD_df to a multiple lines plot.
    Endpoints = inAD_perc_df['Endpoint'].unique()
    inAd_perc_dfs = {p: inAD_perc_df[inAD_perc_df['Endpoint'] == p] for p in Endpoints}
    
    for endpoint in inAd_perc_dfs.keys():
        
        subset_inAD_perc_df = inAd_perc_dfs[endpoint]
        
        Apps = subset_inAD_perc_df['Splitting approach'].unique()
        inAD_splt_dfs = {App: subset_inAD_perc_df[subset_inAD_perc_df['Splitting approach'] == App] for App in Apps}
        
        for ad_method in inAD_splt_dfs.keys():
            plot_file_name = os.path.sep.join([out_dir,  f'ep={endpoint}_ad={ad_method}.png'])
            
            series_to_define_colours_of_points = inAD_splt_dfs[ad_method]['test set'].map(lambda x: map_test_names_onto_plot_args(x))
            
            #---------------------
            assert isinstance(series_to_define_colours_of_points,pd.Series)
            
            inAD_splt_dfs[ad_method].insert(0,'test set int',series_to_define_colours_of_points,allow_duplicates=True)
            
            #==========================
            if ad_method in not_multiple_k_kNN_based_methods:
                x_values = inAD_splt_dfs[ad_method]["Threshold"]
            else:
                x_values = inAD_splt_dfs[ad_method]["Threshold"].astype(int)
            #==========================
            
            #---------------------
            assert isinstance(x_values,pd.Series)
            #----------------------
            
            ax = plt.scatter(x_values,inAD_splt_dfs[ad_method]["Value"],c=inAD_splt_dfs[ad_method]['test set int'])
            #########################
            #Defining legend
            #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
            #https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
            #https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html
            xdata=[0]
            ydata=[0]
            handles = [matplotlib.lines.Line2D(xdata,  ydata,  marker="o",  color=c,  linestyle="none") for c in ["g",  "r",  "b"]]
            plt.legend(handles=handles,  labels=ds_specific_generic_test_set_names,  prop={'size': 10},  ncol=1,numpoints=1)
            #########################
            
            plt.subplots_adjust(right=0.8)
            plt.ylim(0,  100)
            
            #########################
            #https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.scatter.html
            #https://matplotlib.org/3.5.3/api/collections_api.html#matplotlib.collections.PathCollection
            fig = ax.get_figure()
            ########################
            
            title = f'Percentage inside AD [target={endpoint},AD method={ad_method}'
            plt.title('\n'.join(wrap(title,  30)))
            
            #==========================
            if ad_method in not_multiple_k_kNN_based_methods:
                plt.xlabel("Tanimoto distance threshold")
            else:
                plt.xlabel("k value")
                
                #https://matplotlib.org/3.5.3/api/figure_api.html#matplotlib.figure.Figure -> gca() method gets the current axes of the figure object from above
                #https://matplotlib.org/stable/api/axis_api.html#matplotlib.axis.XAxis
                #https://matplotlib.org/stable/api/_as_gen/matplotlib.axis.Axis.set_major_locator.html
                #https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.Locator
                #https://matplotlib.org/3.5.3/api/ticker_api.html#matplotlib.ticker.MaxNLocator
                #We want 'nice' tick locations and default nbins = 10 should be fine if we have 6 k thresholds:
                if len(x_values.unique().tolist()) <= 10:
                    n_bins = 10
                else:
                    n_bins = len(x_values.unique().tolist())
                XAxis=fig.gca().xaxis
                XAxis.set_major_locator(MaxNLocator(nbins=n_bins,integer=True))
            #==========================
            
            plt.ylabel("Percentage of compounds inside AD")
            plt.rcParams.update({'font.size': 18})
            
            
            fig.savefig(plot_file_name,  bbox_inches='tight',  dpi=300,  transparent=True)
            
            plt.clf()
            plt.close()
            
        

def main():
    print('THE START')
    
    for dataset_name in top_class_or_reg_ds_dirs.keys():
        
        
        dir_with_files_to_parse = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],  'Model_Ready'])
        
        out_dir = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],  'AD_Sweeps'])
        
        assert not out_dir == dir_with_files_to_parse,  dir_with_files_to_parse
        
        createOrReplaceDir(dir_=out_dir)
        
        dict_of_all_ad_perc_vals = get_all_inside_ad_perc_vals(dataset_name, dir_with_files_to_parse, out_dir,  ds_matched_to_exemplar_ep_list,  ds_matched_to_ep_to_all_train_test_pairs,  ADMethod2ParameterValues,  random_seed_opts_to_consider_here)
        
        #dict_of_all_ad_perc_vals = load_from_pkl_file(os.path.sep.join([out_dir,'InsideADPercVals.pkl']))
        
        inAD_perc_df = pd.DataFrame(dict_of_all_ad_perc_vals)
        
        record_inside_ad_perc_vals_as_csv(out_dir,inAD_perc_df)
        
        plot_inside_ad_perc_vs_parameter_vals(inAD_perc_df, out_dir, dataset_name,rnd_test_suffix, wang_test_set_names, class_ds_name_to_test_subset_names)
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
