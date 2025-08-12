# Overview

The code provided here was written to generate the results reported in the following publication:

Zied Hosni, Valerie J. Gillet, Richard L. Marchese Robinson, "Explicit applicability domain calculations can help determine when uncertainty estimates are less reliable" (Article being Revised)

The code which is presented here has been organized into folders described under "Code Folders".

In order to make use of this code, please see "Installing dependencies". This describes how to install the Python and module versions used to produce the results reported in the publication.

In order to reproduce the results reported in the publication, please see the step-by-step instructions under "How to reproduce our results" below.

Please note that any references to variables in square brackets, e.g. [Your top level directory name], mean that this should be replaced in its entirety, including the square brackets, with the appropriate value. 

********
Reproducing these results requires access to the modelled datasets. For the public domain datasets for which most results were generated and conclusions drawn, we provide step-by-step instructions how to obtain the relevant files.

Please note that references to calculations performed on the Syngenta datasets below (from step 17 onwards) are included for completeness and to understand how these calculations were performed. The Syngenta datasets themselves have not been made publicly available.

If you do not have access to the Syngenta datasets, the contents of the raw merged statistics files have been provided in the supporting information of Hosni et al., in the Excel file containing raw results (Tables ES7, ES8). These should be saved under the following names and should then be placed in the following location, where [Your top level directory name] is defined below, prior to running the summary plots scripts:

[Your top level directory name]\SyngentaData\Merged_Stats\SYN_logP_Regression_Stats.csv  
[Your top level directory name]\SyngentaData\Merged_Stats\SYN_DT50_Classification_Stats.csv

Likewise, the Syngenta summary plots can only be reproduced if the applicability domain (AD) p-values for the Syngenta results are taken from that Excel file (Tables ES9-ES12) and saved as follows:

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

You should clone this repository, or otherwise download the ZIP file containing the code and extract the folder \QSAR_AD_Uncertainty_Paper_Code_release\.  

[N.B. Here, for convenience, all directory delimiters are denoted as if you were working on Windows - using a backwards slash, even though the final runs for our paper were performed under Linux.]

The folder \QSAR_AD_Uncertainty_Paper_Code_release\ containing, at the top level, the sub-folders \scripts\ and \modelsADuncertaintyPkg\ should be located inside a directory [Your sub-directory name] which should, in turn, be located inside [Your top level directory name]

Here, [Your top level directory name] and [Your sub-directory name] can be names of your choosing. However, at least on Windows, take care that the absolute path names are not too long!

This should give you the following directory structure:  
[Your top level directory name]\ [Your sub-directory name]\QSAR_AD_Uncertainty_Paper_Code_release  

[Example: C:\Calculations\OurCode\QSAR_AD_Uncertainty_Paper_Code_release]

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

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\classification

Run the following Python scripts:

generate_Tox21_SMILES_files.py

generate_ChEMBL_SMILES_files.py

## Step 4: Prepare initial SMILES files based upon the literature defined train/test splits - for the public regression datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\regression

Run the following Python script:

generate_Wang_ChEMBL_SMILES_files.py

## Step 5: Prepare the public datasets for modelling and applicability domain calculations

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\general_purpose

Run the following Python script:

get_model_and_AD_ready_public_datasets.py

That script calls a sequence of scripts which prepare the dataset for modelling and applicability domain (AD) calculations, as described in detail in the publication, by firstly removing duplicates, both before and after standardizing molecular structures, filtering compounds based upon additional criteria, randomly splitting the original literature defined training sets for the classification datasets (80:20 train:test) and then computing fingerprints, dropping molecules where fingerprints cannot be computed.

## Step 6: Produce public dataset statistics

### Step 6 (a): Convert datasets, prior to any of the operations required to get them ready for modelling, into a format required by the statistics script in step 6 (b)

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\general_purpose

Run the following Python script:

consistently_format_dataset_files_without_filtering.py

### Step 6 (b): Produce a summary of dataset statistics before and after the operations (filtering duplicates etc.) required to get them ready for modelling and AD calculations


Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\general_purpose

Run the following Python script:

summarize_dataset_stats_before_and_after_parsing.py

## Step 7: Plot dataset distributions for the modelling-ready public datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\general_purpose

Run the following Python script:

plot_TSNE_distributions.py

## Step 8: Plot the percentage of test set compounds inside the applicability domain (AD) against different parameter values for all AD methods for all train:test splits of the exemplar public targets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\general_purpose

Run the following Python script:

compute_percentage_in_AD_for_different_parameter_options.py

Note (1): These plots, which are not shown in the manuscript for brevity, were analyzed to identify reasonable default AD method parameters based on our inituition that a majority of compounds from a random test set should lie inside the domain but that - across different test sets - we would generally identify a sizeable number inside and outside the domain which were expected to allow us compute robust shift-metric values to evaluate the different AD methods. 

Note (2): The AD methods considered here include the approach (nUNC - denoted "UNC" in the scripts) we focus on in the manuscript, as well as other approaches we initially explored on the exemplar datasets: "RDN", "dkNN" and "Tanimoto" (based on SIMILARITYNEAREST1 with a defined Tanimoto distance threshold for inside vs. outsid the domain).

## Step 9: Perform modelling and compute statistics for all test set compounds, as well as compounds inside and outside the AD, for all uncertainty methods (= algorithms) and AD methods, on the exemplar endpoints (=targets) for the public classification datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\classification

Run the following Python script:

perform_classification_modelling_on_exemplar_endpoints.py

Note: This script also generates the delta-calibration plots corresponding to the analyzed statistics using the default delta value (lit_precedent_delta_for_calib_plot). In addition, delta-calibration plots are generated for a larger value of delta (larger_delta_for_calib_plot), as discussed in the manuscript.

***********************
To report delta-calibration plot statistics for other values of delta (other than 0.05 which was chosen based on literature precedence), run this script:

compute_non_default_delta_calibration_plot_stats.py
***********************

## Step 10: Perform modelling and compute statistics for all test set compounds, as well as compounds inside and outside the AD, for all uncertainty methods (= algorithms) and AD methods, on the exemplar endpoints (=targets) for the public regression dataset

You will need to manually create the \Modelling\ sub-directory here first: [Your top level directory name]\PublicData\Wang_ChEMBL\data\dataset\Modelling\

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\regression

Run the following Python script, once per exemplar endpoint, as follows:

perform_regression_modelling_on_exemplar_endpoints.py -e Dopamine

perform_regression_modelling_on_exemplar_endpoints.py -e COX-1

perform_regression_modelling_on_exemplar_endpoints.py -e COX-2

## Step 11: Merge all statistics obtained for the exemplar endpoints from the public classification and regression datasets into a single set of classification and regression statistics

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\general_purpose

Run the following Python script:

merge_exemplar_modelling_stats.py

## Step 12: Compute the applicability domain shift-metric p-values, for all AD and uncertainty methods, for the exemplar targets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\Analysis

Run the following Python script:

python assess_AD_stat_sig_for_exemplar_public_data.py

## Step 13: Perform modelling and compute statistics for all test set compounds, as well as compounds inside and outside the AD, for the default uncertainty and AD methods, on the other endpoints (=targets) for the public classification datasets

You will need to manually create the \Modelling.2\ sub-directories here first: [Your top level directory name]\PublicData\Morger_ChEMBL\Modelling.2\ and [Your top level directory name]\PublicData\Tox21\Modelling.2\

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\classification

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


## Step 14: Perform modelling and compute statistics for all test set compounds, as well as compounds inside and outside the AD, for the default uncertainty and AD methods, on the other endpoints (=targets) for the public regression dataset

You will need to manually create the \Modelling.2\ sub-directory here first: [Your top level directory name]\PublicData\Wang_ChEMBL\data\dataset\Modelling.2\

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\regression

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

## Step 15: Merge all statistics obtained for the other endpoints from the public classification and regression datasets into a single set of classification and regression statistics

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\general_purpose

Run the following Python script:

merge_other_modelling_stats.py


## Step 16: Compute the applicability domain shift-metric p-values for the other public dataset endpoints

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\Analysis

Run the following Python script:

assess_AD_stat_sig_for_modelling_other_endpoints.py

## Step 17: Perform modelling and compute statistics for all test set compounds, as well as compounds inside and outside the AD, for the default uncertainty and AD methods, on the timesplits of the Syngenta classification dataset

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\SynExamples

Run the following Python script:

class_syn_updates.py

----
The curated training set statistics were computed using an adaptation of this script:
just_analyse_curated_ts_class_syn_updates.py


## Step 18: Perform modelling and compute statistics for all test set compounds, as well as compounds inside and outside the AD, for the default uncertainty and AD methods, on the timesplits of the Syngenta regression dataset

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\SynExamples

Run the following Python script:

reg_syn_updates.py

----
The curated training set statistics were computed using an adaptation of this script:
just_analyse_curated_ts_reg_syn_updates.py


## Step 19: Merge the statistics obtained from modelling on the Syngenta datasets into one file of classification and one file of regression statistics:

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\SynExamples

Run the following Python script:

merge_syn_modelling_statistics.py

## Step 20: Compute the applicability domain shift-metric p-values for the Syngenta datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\SynExamples

Run the following Python script:

assess_AD_stat_sig_for_syn_updates.py

## Step 21: Perform global adjustment of all p-values for multiple-testing, to control the false-discovery rate

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts

Run the following Python script:

adjust_all_p_values.py

*****
An analysis of the proportion of two-tail adjusted p-values which were statistically significant when the average shift-metric had the wrong sign, out of all statistically significant adjusted two-tail p-values, was performed using this script:

find_the_proportion_of_two_tail_adjusted_stat_sig_when_average_shift_metric_had_the_wrong_sign.py
*****

## Step 22: Merge the default uncertainty and AD method results for the public exemplar and other targets prior to plotting and summary analyses

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\Analysis

Run the following Python script:

python merge_default_method_res_for_all_eps.py

## Step 23: Produce the summary plots of modelling statistics, for the default uncertainty and AD methods, for all endpoints from the public datasets, where the axes are adjusted to be consistent with the range of values obtained for the Syngenta datasets:

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\PublicDataModelling\Analysis

Run the following Python script:

summary_plots_for_modelling_all_endpoints.py

***********
An analysis of the trends shown by these plots was also performed by running the other script here:

shift_metric_trends_analysis_all_eps.py
*************

************
In addition, an analysis of the trends shown for raw metrics was performed by running this other script here:

compute_fract_of_better_than_chance_or_valid_metrics_for_AD_subsets.py
*************


## Step 24: Produce the summary plots of modelling statistics for the Syngenta datasets, where the axes are adjusted to be consistent with the range of values obtained for the public datasets:

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\SynExamples

Run the following Python script:

summary_plots_for_syn_updates.py

## Step 25: Produce the dataset distribution plots for the Syngenta datasets

Navigate to ..\QSAR_AD_Uncertainty_Paper_Code_release\scripts\SynExamples

Run the following Python scripts:

tsne_distributions_class_syn_updates.py

tsne_distributions_reg_syn_updates.py





