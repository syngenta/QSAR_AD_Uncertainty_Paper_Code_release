# Overview

The code provided here was written to generate the results reported in the following publication:

Zied Hosni, Valerie J. Gillet, Richard L. Marchese Robinson, "How do uncertainty estimation methods perform for out-of-domain compounds according to different applicability domain methods ?" (Article In Preparation)

The code which is presented here has been organized into folders described under "Code Folders".

In order to make use of this code, please see "Installing dependencies". This describes how to install the Python and module versions used to produce the results reported in the publication.

In order to reproduce the results reported in the publication, please see the step-by-step instructions under "How to reproduce our results".

Please note that any references to variables in square brackets, e.g. [Your top level directory name], means that this should be replaced in its entirety, including the square brackets, with the appropriate value. 

********
Reproducing these results requires access to the modelled datasets. For the public domain datasets for which most results were generated and conclusions drawn, we provide step-by-step instructions how to obtain the relevant files.

Please note that references to calculations performed on the Syngenta datasets below (from step 20 onwards) are included for completeness and to understand how these calculations were performed. The Syngenta datasets themselves have not been made publicly available.

If you do not have access to the Syngenta datasets, the contents of the raw merged statistics files have been provided in the supporting information of Hosni et al., in SI_Tables_of_Results.xlsx (Tables ES15, ES16). These should be saved under the following names and should then be placed in the following location, where [Your top level directory name] is defined below, prior to running the summary plots scripts:

[Your top level directory name]\SyngentaData\Merged_Stats\SYN_logP_Regression_Stats.csv  
[Your top level directory name]\SyngentaData\Merged_Stats\SYN_DT50_Classification_Stats.csv

Likewise, the Syngenta summary plots can only be reproduced if the applicability domain (AD) p-values for the Syngenta results are taken from SI_Tables_of_Results.xlsx (Tables ES17a-ES18b) and saved as follows:

[Your top level directory name]\SyngentaData\DT50_Updates\Calc\AD.stat.sig\one_tail_Classification_PVals_GlobalAdjusted.csv

[Your top level directory name]\SyngentaData\DT50_Updates\Calc\AD.stat.sig\Classification_PVals_GlobalAdjusted.csv

[Your top level directory name]\SyngentaData\logP_updates\Calc\AD.stat.sig\one_tail_Regression_PVals_GlobalAdjusted.csv

[Your top level directory name]\SyngentaData\logP_updates\Calc\AD.stat.sig\Regression_PVals_GlobalAdjusted.csv
*********

# Code Folders

scripts\ : this folder contains the scripts used to generate results for the publication

modelsADuncertaintyPkg\ : this folder is structured like a Python package and contains implementations of methods and utilities which were used within the scripts for generating results for the publication

test_suite\ : this folder contains definitions of unit-tests, other tests and associated files for testing the code presented in the folders scripts\ or modelsADuncertaintyPkg\

.github/workflows\ : this folder contains a file defining a CI/CD flow, which triggers running the tests found under test_suite\ using pytest upon pushing commits to the GitHub repo

# Installing dependencies

First, clone this repo from GitHub and navigate to the top-level directory containing requirements.txt

Install Python 3.8.17

Create and activate a Python environment as follows [see https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/ if you get stuck]

Create a Python environment:
python -m venv [name of folder into which the environment should be installed]

[Here, we used python -m venv ./fcc1shef]

Activate that Python environment:  
[Here, we created an environment in folder fcc1shef on a Linux box.]
source fcc1shef/bin/activate

Install the dependencies into this environment:

python -m pip install -r requirements.txt (or pip install -r requirements.txt)

N.B. Some dependencies of these modules will still need to be resolved by pip. 

For the purposes of the publication, the full details of the environment dependencies installed by pip are described in full_env_pip_install_desc.txt

# How to reproduce our results

Note (1): In some cases in the code and the README text below, the terms 'dataset' or 'dataset_name' etc. may be used to refer to 'dataset groups', as defined in the publication, i.e. Morger-ChEMBL, Morger-Tox21, Wang-ChEMBL may be referred to as a 'dataset', even though they are 'dataset groups', comprising endpoint-specific (= target-specific) datasets.

Note (2): The scripts referred to can also be ran from [Your top level directory name], as long as their full name is specified following the python command. The one exception is python get_model_and_AD_ready_public_datasets.py, for which you do need to navigate to this directory. 

## Step 1: Clone this repository and ensure the correct local directory structure

You should clone this repository, or otherwise download the ZIP file containing the code and extract the folder \QSAR_AD_Uncertainty_Paper_Code\.  

[N.B. Here, for convenience, all directory delimiters are denoted as if you were working on Windows - using a backwards slash, even though the final runs for our paper were performed under Linux.]

The folder \QSAR_AD_Uncertainty_Paper_Code\ containing, at the top level, the sub-folders \scripts\ and \modelsADuncertaintyPkg\ should be located inside a directory [Your sub-directory name] which should, in turn, be located inside [Your top level directory name]

Here, [Your top level directory name] and [Your sub-directory name] can be names of your choosing. However, at least on Windows, take care that the absolute path names are not too long!

This should give you the following directory structure:  
[Your top level directory name]\ [Your sub-directory name]\QSAR_AD_Uncertainty_Paper_Code  

[Example: C:\Calculations\OurCode\QSAR_AD_Uncertainty_Paper_Code]

## Step 2: Download the public dataset files into the correct local directories

Create the following sub-directories of the top-level directory referred to above:

[Your top level directory name]\PublicData  

[Your top level directory name]\PublicData\Morger_ChEMBL  

[Your top level directory name]\PublicData\Tox21  

[Your top level directory name]\PublicData\Tox21\Tox21Train  

[Your top level directory name]\PublicData\Tox21\Tox21Test  

[Your top level directory name]\PublicData\Tox21\Tox21Score  

[Your top level directory name]\PublicData\Wang_ChEMBL  

### Step 2(a): Downloading the Morger-ChEMBL dataset

Manually download and extract all of the following CSV files into the folder ..\Morger_ChEMBL defined above.

All links referred to here were last accessed on 14th July 2023.

The file "chembl_chembio_descriptors.tar.bz2" should be downloaded from Zenodo [https://doi.org/10.5281/zenodo.5167636]

Subsequently, the files [CHEMBLID]_chembio_normalizedDesc.csv, e.g. "CHEMBL203_chembio_normalizedDesc.csv", should be manually extracted.

The file "data_size_chembio_chembl.csv" should be downloaded from GitHub [https://github.com/volkamerlab/CPrecalibration_manuscript_SI/tree/main/data]

### Step 2(b): Downloading the Morger-Tox21 dataset

Manually download and extract all of the following SDF and text files into the indicated sub-directories of the folder ..\Tox21 defined above.

All of these files were downloaded from https://tripod.nih.gov/tox21/challenge/data.jsp  (last accessed 8th July 2023).

Download and extract all of the "Training Datasets" SDF files into ..\Tox21\Tox21Train

Download and extract tox21_10k_challenge_test.sdf into ..\Tox21\Tox21Test

Download and extract tox21_10k_challenge_score.sdf into ..\Tox21\Tox21Score

Download ("Results for the final evaluation dataset") tox21_10k_challenge_score.txt into ..\Tox21\Tox21Score

### Step 2(c): Downloading the Wang-ChEMBL dataset

Clone the following repo / download as a ZIP: https://github.com/wangdingyan/HybridUQ (last accessed 14th July 2023)

Extract the \data\ sub-folder into ..\Wang_ChEMBL defined above.

## Step 3: Prepare initial SMILES files based upon the literature defined train/test splits - for the public classification datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\classification

Run the following Python scripts:

generate_Tox21_SMILES_files.py

generate_ChEMBL_SMILES_files.py

## Step 4: Prepare initial SMILES files based upon the literature defined train/test splits - for the public regression datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\regression

Run the following Python script:

generate_Wang_ChEMBL_SMILES_files.py

## Step 5: Prepare the public datasets for modelling and applicability domain calculations

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\general_purpose

Run the following Python script:

get_model_and_AD_ready_public_datasets.py

That script calls a sequence of scripts which prepare the dataset for modelling and applicability domain (AD) calculations, as described in detail in the publication, by firstly removing duplicates, both before and after standardizing molecular structures, filtering compounds based upon additional criteria, randomly splitting the original literature defined training sets for the classification datasets (80:20 train:test) and then computing fingerprints, dropping molecules where fingerprints cannot be computed.

## Step 6: Produce public dataset statistics

### Step 6 (a): Convert datasets, prior to any of the operations required to get them ready for modelling, into a format required by the statistics script in step 6 (b)

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\general_purpose

Run the following Python script:

consistently_format_dataset_files_without_filtering.py

### Step 6 (b): Produce a summary of dataset statistics before and after the operations (filtering duplicates etc.) required to get them ready for modelling and AD calculations


Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\general_purpose

Run the following Python script:

summarize_dataset_stats_before_and_after_parsing.py

### Step 6 (c): Assess the overlaps between endpoint-specific datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\general_purpose

Run the following Python script:

python consider_dataset_overlaps.py

Note: this was considered when selecting the exemplar endpoints

## Step 7: Plot dataset distributions for the modelling-ready public datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\general_purpose

Run the following Python script:

plot_dataset_pairwise_distance_distributions.py

Run the following Python script:

plot_TSNE_distributions.py

## Step 8: Plot activity distributions for the modelling-ready regression public datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\regression

Run the following Python script:

plot_Wang_activities_train_vs_test.py

## Step 9: Plot the percentage of test set compounds inside the applicability domain (AD) against different parameter values for all AD methods for all train:test splits of the exemplar public targets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\general_purpose

Run the following Python script:

compute_percentage_in_AD_for_different_parameter_options.py

Note: The names for the AD methods in the publication corresponded to the labels of the AD methods in this and subsequent scripts as follows: (1) nUNC = "UNC"; (2) RDN = "RDN"; (3) dk-NN = "dkNN"; (4) d1-NN = "Tanimoto".

## Step 10: Perform modelling and compute statistics for all test set compounds, as well as compounds inside and outside the AD, for all uncertainty methods (= algorithms) and AD methods, on the exemplar endpoints (=targets) for the public classification datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\classification

Run the following Python script:

perform_classification_modelling_on_exemplar_endpoints.py

Note: This script also generates the delta-calibration plots corresponding to the analyzed statistics using the default delta value (lit_precedent_delta_for_calib_plot). In addition, delta-calibration plots are generated for a larger value of delta (larger_delta_for_calib_plot), as discussed in the manuscript.

**********************
To investigate non-default AD method parameters (no. nearest neighbours for dk-NN = dkNN, RDN, nUNC = UNC and distance threshold for d1-NN = Tanimoto), use the following command-line options:  

-k [no. nearest neighbours]

-t [distance threshold]

***********************

***********************
To report delta-calibration plot statistics for other values of delta (other than 0.05 which was chosen based on literature precedence), run this script:

compute_non_default_delta_calibration_plot_stats.py
***********************

## Step 11: Perform modelling and compute statistics for all test set compounds, as well as compounds inside and outside the AD, for all uncertainty methods (= algorithms) and AD methods, on the exemplar endpoints (=targets) for the public regression dataset

You will need to manually create the \Modelling\ sub-directory here first: [Your top level directory name]\PublicData\Wang_ChEMBL\data\dataset\Modelling\

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\regression

Run the following Python script, once per exemplar endpoint, as follows:

perform_regression_modelling_on_exemplar_endpoints.py -e Dopamine

perform_regression_modelling_on_exemplar_endpoints.py -e COX-1

perform_regression_modelling_on_exemplar_endpoints.py -e COX-2

**********************
To investigate non-default AD method parameters (no. nearest neighbours for dk-NN = dkNN, RDN, nUNC = UNC and distance threshold for d1-NN = Tanimoto), use the following command-line options: 

-k [no. nearest neighbours]

-t [distance threshold]

***********************

## Step 12: Merge all statistics obtained for the exemplar endpoints from the public classification and regression datasets into a single set of classification and regression statistics

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\general_purpose

Run the following Python script:

merge_exemplar_modelling_stats.py

*******************************
To select the most suitable Tanimoto distance threshold (d1-NN) and k for the k-nearest neighbour AD methods (dk-NN, RDN, nUNC), we considered, as well as the percentage inside the domain for different kinds of test sets, looking for values which would either restore validity inside the domain, in terms of the average coverage (denoted as 'validity' here), for our recommended default conformal regression method - or otherwise maximize this - for the nominal out-of-domain test set type (IVOT). For looking at the averages, the script  ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\Analysis\average_validity_inside_over_folds_and_seeds.py may be used.

After analysing these results, the default values for the AD methods were finalized and hardcoded in ..\QSAR_AD_Uncertainty_Paper_Code\scripts\consistent_parameters_for_all_modelling_runs.py: see ADMethod2Param
The winners analyses, comparing AD methods, were based upon the finalized default values for the AD methods.
******************************** 

## Step 13: Carry out the selection of algorithm winners on exemplar targets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\Analysis

Run the following Python script:

python Alg_ranking.py

Output files from this script are organized per dataset under the folder ..\PublicData\Algorithm_ranking_results\ [dataset_name]:

average_raw_metrics_df.csv - metrics and numbers of compounds for test sets, prior to applicability domain method splitting, averaged over random seeds and, where applicable, folds

All_target_metrics_diff_table_thres__thresholdNA_[dataset].csv - difference between averaged metrics and top-ranked values

[norm or raw]_ALG_wins_per_ts_averaged_over_eps_thresh__threshold[threshold]_[dataset].csv - normalized or raw counts of winners, where both the algorithms with the top-ranked metrics and those with metric values up to the value of [threshold] different from the top-ranked were considered winners, averaged over all targets (= endpoints)

[target]_all_[Norm or Raw]_wins_count__threshold[threshold]_[dataset].csv - normalized or raw counts of winners for a given [target] and [threshold]

## Step 14: Compute the applicability domain shift-metric p-values for the exemplar targets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\Analysis

Run the following Python script:

python assess_AD_stat_sig_for_exemplar_public_data.py

## Step 15: Carry out the selection of applicability domain (AD) method winners on exemplar targets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\Analysis

Run the following Python script:

python AD_ranking.py 

Output files from this script are organized per dataset under the folder ..\PublicData\ADWA\ [dataset_name]:

average_raw_metrics_df.csv - metrics and numbers of compounds for the subsets of the test sets, after applicability domain method splitting, averaged over random seeds and, where applicable, folds

raw_shift_metrics.csv - shift-metrics prior to averaging over seeds and, where applicable, folds

average_shift_metrics_df.csv - shift-metrics averaged over seeds and, where applicable, folds

average_shift_metrics_ready_for_ranking_p=[significance limit].csv - averaged shift-metrics replaced with dummy values (-Inf if a higher value is better or Inf if a lower value is better) when the corresponding p-value was not statistically significant (p-value [2dp] <= [significance limit]) or could not be computed, or the average shift-metric did not have the sign expected if the AD method was working, to ensure that the corresponding AD method can never be ranked as the best for this metric and test set

All_target_metrics_diff_shifts_table_thres__gen_[generalization]alg=[algorithm of interest]threshold[threshold]_AD_statSigThresh_[significance limit]_[dataset].csv - the difference in the averaged shift-metrics to the top-ranked values for a given test-set, across all relevant algorithms (either the single [algorithm of interest] or all modelling/uncertainty algorithms if [generalization]=True)

[target]_all_[Norm or Raw]_wins_count__gen_[generalization]alg=[algorithm of interest]threshold[threshold]_AD_statSigThresh_[significance limit]_[dataset].csv - the target-specific normalized (divided by all metrics and algorithms considered) or raw counts of winners, across all algorithms of interest, where both the AD method with the top-ranked averaged shift-metric (after ensuring results without a statistically significant one-tail p-value could not be top-ranked if [significance limit] is specified) and any other methods within [threshold] of this value are considered winners

[norm or raw]_AD_wins_per_ts_averaged_over_eps_thresh__gen_[generalization]alg=[algorithm of interest]threshold[threshold]_AD_statSigThresh_[significance limit]_[dataset].csv - normalized or raw counts of winners averaged across all targets for this [dataset]

*********
After examining the results of the AD winners analysis, focusing on the recommnded default algorithms (for classification and regression) from the algorithm winners analysis, the proposed default method can be sanity checked using the following script, where the proposed default was hardcoded using the variable default_ad_method and the recommended defaults for the modelling algorithsm were harcoded using the variable default_alg:

python default_ad_sanity_checking.py
********
********
Finally, the recommended default algorithms and AD methods for classification and regression were hardcoded in this file:

..\QSAR_AD_Uncertainty_Paper_Code\scripts\recommended_defaults.py
********

## Step 16: Perform modelling and compute statistics for all test set compounds, as well as compounds inside and outside the AD, for the default modelling and AD methods, on the other endpoints (=targets) for the public classification datasets

You will need to manually create the \Modelling.2\ sub-directories here first: [Your top level directory name]\PublicData\Morger_ChEMBL\Modelling.2\ and [Your top level directory name]\PublicData\Tox21\Modelling.2\

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\classification

Run the following Python script, once per endpoint, as follows:

perform_classification_modelling_on_other_endpoints.py -e CHEMBL220
perform_classification_modelling_on_other_endpoints.py -e CHEMBL5763
perform_classification_modelling_on_other_endpoints.py -e CHEMBL203
perform_classification_modelling_on_other_endpoints.py -e CHEMBL279
perform_classification_modelling_on_other_endpoints.py -e CHEMBL230
perform_classification_modelling_on_other_endpoints.py -e CHEMBL340
perform_classification_modelling_on_other_endpoints.py -e CHEMBL2039
perform_classification_modelling_on_other_endpoints.py -e CHEMBL222
perform_classification_modelling_on_other_endpoints.py -e NR-AR-LBD
perform_classification_modelling_on_other_endpoints.py -e NR-AhR
perform_classification_modelling_on_other_endpoints.py -e NR-ER
perform_classification_modelling_on_other_endpoints.py -e NR-ER-LBD
perform_classification_modelling_on_other_endpoints.py -e NR-PPAR-gamma
perform_classification_modelling_on_other_endpoints.py -e SR-ATAD5
perform_classification_modelling_on_other_endpoints.py -e SR-MMP
perform_classification_modelling_on_other_endpoints.py -e SR-p53

Note: This script also generates the delta-calibration plots corresponding to the analyzed statistics using the default delta value (lit_precedent_delta_for_calib_plot). In addition, delta-calibration plots are generated for a larger value of delta (larger_delta_for_calib_plot), as discussed in the manuscript.


## Step 17: Perform modelling and compute statistics for all test set compounds, as well as compounds inside and outside the AD, for the default modelling and AD methods, on the other endpoints (=targets) for the public regression dataset

You will need to manually create the \Modelling.2\ sub-directory here first: [Your top level directory name]\PublicData\Wang_ChEMBL\data\dataset\Modelling.2\

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\regression

Run the following Python script, once per endpoint, as follows:

perform_regression_modelling_on_other_endpoints.py -e Ephrin
perform_regression_modelling_on_other_endpoints.py -e Estrogen
perform_regression_modelling_on_other_endpoints.py -e Glucocorticoid
perform_regression_modelling_on_other_endpoints.py -e Glycogen
perform_regression_modelling_on_other_endpoints.py -e HERG
perform_regression_modelling_on_other_endpoints.py -e JAK2
perform_regression_modelling_on_other_endpoints.py -e LCK
perform_regression_modelling_on_other_endpoints.py -e Monoamine
perform_regression_modelling_on_other_endpoints.py -e opioid
perform_regression_modelling_on_other_endpoints.py -e A2a
perform_regression_modelling_on_other_endpoints.py -e Vanilloid
perform_regression_modelling_on_other_endpoints.py -e ABL1
perform_regression_modelling_on_other_endpoints.py -e Acetylcholinesterase
perform_regression_modelling_on_other_endpoints.py -e Cannabinoid
perform_regression_modelling_on_other_endpoints.py -e Carbonic
perform_regression_modelling_on_other_endpoints.py -e Caspase
perform_regression_modelling_on_other_endpoints.py -e Coagulation
perform_regression_modelling_on_other_endpoints.py -e Dihydrofolate

## Step 18: Merge all statistics obtained for the other endpoints from the public classification and regression datasets into a single set of classification and regression statistics

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\general_purpose

Run the following Python script:

merge_other_modelling_stats.py


## Step 19: Compute the applicability domain shift-metric p-values for the other public dataset endpoints

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\Analysis

Run the following Python script:

assess_AD_stat_sig_for_modelling_other_endpoints.py

## Step 20: Perform modelling and compute statistics for all test set compounds, as well as compounds inside and outside the AD, for the default modelling and AD methods, on the timesplits of the Syngenta classification dataset

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\SynExamples

Run the following Python script:

class_syn_updates.py

----
The curated training set statistics were computed using an adaptation of this script:
just_analyse_curated_ts_class_syn_updates.py


## Step 21: Perform modelling and compute statistics for all test set compounds, as well as compounds inside and outside the AD, for the default modelling and AD methods, on the timesplits of the Syngenta regression dataset

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\SynExamples

Run the following Python script:

reg_syn_updates.py

----
The curated training set statistics were computed using an adaptation of this script:
just_analyse_curated_ts_reg_syn_updates.py


## Step 22: Merge the statistics obtained from modelling on the Syngenta datasets into one file of classification and one file of regression statistics:

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\SynExamples

Run the following Python script:

merge_syn_modelling_statistics.py

## Step 23: Compute the applicability domain shift-metric p-values for the Syngenta datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\SynExamples

Run the following Python script:

assess_AD_stat_sig_for_syn_updates.py

## Step 24: Perform global adjustment of all p-values for multiple-testing, to control the false-discovery rate

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts

Run the following Python script:

adjust_all_p_values_as_one_family.py

*****
An analysis of the proportion of two-tail adjusted p-values which were statistically significant when the average shift-metric had the wrong sign, out of all statistically significant adjusted p-values, was performed using this script:

find_the_proportion_of_two_tail_adjusted_stat_sig_when_average_shift_metric_had_the_wrong_sign.py
*****

## Step 25: Produce the summary plots of modelling statistics for the other endpoints from the public datasets, where the axes are adjusted to be consistent with the range of values obtained for the Syngenta datasets:

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\PublicDataModelling\Analysis

Run the following Python script:

summary_plots_for_modelling_other_endpoints.py

***********
An analysis of the trends shown by these plots was also performed by running the other script here:

shift_metric_trends_analysis_other_eps.py
*************


## Step 26: Produce the summary plots of modelling statistics for the Syngenta datasets, where the axes are adjusted to be consistent with the range of values obtained for the public datasets:

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\SynExamples

Run the following Python script:

summary_plots_for_syn_updates.py

## Step 27: Produce the dataset distribution plots for the Syngenta datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code\scripts\SynExamples

Run the following Python scripts:

pairwise_distance_distributions_class_syn_updates.py

pairwise_distance_distributions_reg_syn_updates.py

tsne_distributions_class_syn_updates.py

tsne_distributions_reg_syn_updates.py





