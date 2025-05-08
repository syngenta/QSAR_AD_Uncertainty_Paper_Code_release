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
import os,sys
#==========================================
dir_of_this_file = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_file)))
#==========================================
sys.path.append(pkg_dir)

from modelsADuncertaintyPkg.qsar_eval import all_key_class_stats_and_plots as ClassEval
from modelsADuncertaintyPkg.utils.basic_utils import load_from_pkl_file,createOrReplaceDir

top_ds_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData'])





def main():
    print('THE START')

    top_ds_dir_of_interest = os.path.sep.join([top_ds_dir,'Morger_ChEMBL'])

    res_dir = os.path.sep.join([top_ds_dir_of_interest,'Modelling','k=3_t=0.4'])

    output_dir = os.path.sep.join([res_dir,'DeltaEffect'])

    createOrReplaceDir(output_dir)

    raw_res = load_from_pkl_file(os.path.sep.join([res_dir,'RawResAndStats.pkl']))['dict_of_raw_results']

    exemplar_endpoint_to_consider = 'CHEMBL206'

    method = 'CVAP'

    ################
    #AD_method is redundant if All compounds are considered:
    AD_method = 'Tanimoto' 
    subset_name = 'All'
    ################

    test_set_types_to_consider = ['holdout_FPs','train_FPs_RndTest0.2']

    delta_vals_to_consider = [0.05,0.2]

    for test_type in test_set_types_to_consider:
        
        raw_res_subset = raw_res[exemplar_endpoint_to_consider][f'{exemplar_endpoint_to_consider}_{test_type}'][1][AD_method][method][subset_name]

        test_y = raw_res_subset['subset_test_y']

        predicted_y = raw_res_subset['subset_predicted_y']

        probs_for_class_1 = raw_res_subset['subset_probs_for_class_1']

        for delta in delta_vals_to_consider:

            precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal,no_cmpds_1,no_cmpds_0,no_cmpds = ClassEval.computeAllClassMetrics(test_y,predicted_y,probs_for_class_1,method,subset_name,output_dir,delta_for_calib_plot=delta)

            print(f'e={exemplar_endpoint_to_consider},m={method},t={test_type},delta={delta}: R2(cal)={coeffOfDeterminationCal:.2f}, Pearson coefficient (cal)={PearsonCoeffCal:.2f}, Spearman coefficient (cal)={SpearmanCoeffCal:.2f}, RMSE (cal)={rmseCal:.2f}')



    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())

                        