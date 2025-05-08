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
###########################
#Copyright (c) 2022 Syngenta
###########################
#This code was adapted at Syngenta from the following Open Source project:
#https://doi.org/10.5281/zenodo.3477986
#get_input_for_FF_eval_v3.py
#Copyright (c) 2016-2019 University of Leeds
# Contact R.L.MarcheseRobinson@leeds.ac.uk
####################################
#https://opensource.org/licenses/BSD-3-Clause
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
####################################
import sys,re,os
import numpy as np
from collections import defaultdict

def getAveragePlusDeviationEstimateForBioactAfterRemovingOutliers(dict_of_bio_data_points_for_one_mol,n_sd_extreme=3,type_of_averaging='ArithmeticMean'):
	'''
	Explanation of function arguments:
	
	dict_of_bio_data_points_for_one_mol : A dictionary mapping each relevant bioactivity value for a single molecule (previously grouped by the same InChI) to a unqiue number, e.g. if a given InChI was found in 4 different rows associated with 4 bioactivity values [1.8, 2.9, 0.9,3.8] for the relevant target and organism, dict_of_bio_data_points_for_one_mol = {1:1.8,2:2.9,3:0.9,4:3.8}
	
	n_sd_extreme : The number of multiples of the standard deviation of the other values a value must lie from the mean of those other values before being considered an outlier. The identification of outliers only happens when there are more than 2 data points. If this is the case, each data point in turn is considered to determine whether it is an outlier that should be excluded from averaging. 
	
	type_of_averaging : Currently only arithmetic mean is supported. Could this be extended to the geometric mean etc., although this might be inconsistent with the algorithm to identify outliers?
	'''
	
	highest_Q_info_dict = defaultdict(dict)
	
	for data_point_id in dict_of_bio_data_points_for_one_mol.keys():
		highest_Q_info_dict[data_point_id]['BioActVal'] = float(dict_of_bio_data_points_for_one_mol[data_point_id])
	
	bioactVal_average,bioactVal_deviation = averageBioactExcludingOutliers(highest_Q_info_dict,n_sd_extreme,type_of_averaging)
	
	return bioactVal_average,bioactVal_deviation

def check_getAveragePlusDeviationEstimateForBioactAfterRemovingOutliers():
	examples = defaultdict(dict)
	
	#===========================
	examples[1]['dict_of_bio_data_points_for_one_mol'] = {1:0.9,2:1.1,3:500}
	examples[1]['expected_bioactVal_average'] = 1.0
	examples[1]['expected_bioactVal_deviation'] = 0.1
	#===========================
	
	#===========================
	examples[2]['dict_of_bio_data_points_for_one_mol'] = {1:0.9,2:500}
	examples[2]['expected_bioactVal_average'] = 250.45
	examples[2]['expected_bioactVal_deviation'] = 249.55
	#===========================
	
	for eg in examples.keys():
		print('Checking getAveragePlusDeviationEstimateForBioactAfterRemovingOutliers(..) for example {}'.format(eg))
		
		bioactVal_average,bioactVal_deviation = getAveragePlusDeviationEstimateForBioactAfterRemovingOutliers(dict_of_bio_data_points_for_one_mol=examples[eg]['dict_of_bio_data_points_for_one_mol'])
		
		assert round(examples[eg]['expected_bioactVal_average'],5) == round(bioactVal_average,5),"expected_bioactVal_average = {} vs. bioactVal_average = {}".format(examples[eg]['expected_bioactVal_average'],bioactVal_average)
		
		assert round(examples[eg]['expected_bioactVal_deviation'],5) == round(bioactVal_deviation,5),"expected_bioactVal_deviation = {} vs. bioactVal_deviation = {}".format(examples[eg]['expected_bioactVal_deviation'],bioactVal_deviation)
		
		print('CHECKED getAveragePlusDeviationEstimateForBioactAfterRemovingOutliers(..) for example {}'.format(eg))

def excludeOutliers(highest_Q_info_dict,n_sd_extreme=3,type_of_averaging='ArithmeticMean'):
	
	
	info_dict_excluding_outliers = defaultdict(dict)
	
	#all_bioact_vals = [highest_Q_info_dict[row_count]['BioActVal'] for row_count in highest_Q_info_dict.keys()]
	
	if len(highest_Q_info_dict.keys()) <= 2:
		for row_count in highest_Q_info_dict.keys():
			for key in highest_Q_info_dict[row_count].keys():
				info_dict_excluding_outliers[row_count][key] = highest_Q_info_dict[row_count][key]
	else:
		outliers_excluded = False
		
		for row_count in highest_Q_info_dict.keys():
			bioact_val = highest_Q_info_dict[row_count]['BioActVal']
			other_bioact_vals = [highest_Q_info_dict[other_row_count]['BioActVal'] for other_row_count in highest_Q_info_dict.keys() if not other_row_count==row_count]
			
			
			if 'ArithmeticMean' == type_of_averaging:
				bioactVal_average,bioactVal_deviation = getMeanSd(other_bioact_vals)
			else:
				raise Exception('{}is not supported!'.format(type_of_averaging))
			
			others_mean = bioactVal_average
			others_sd = bioactVal_deviation
			
			if abs(bioact_val - others_mean) > (n_sd_extreme*others_sd):
				outliers_excluded = True
				print('%f is an outlier vs. %s!' % (bioact_val,str(other_bioact_vals)))
				continue
			else:
				for key in highest_Q_info_dict[row_count].keys():
					info_dict_excluding_outliers[row_count][key] = highest_Q_info_dict[row_count][key]
				del key
		del row_count
		
		if (len(info_dict_excluding_outliers.keys()) > 2) and outliers_excluded:
			print ('Recursively applying outlier exclusion!')
			info_dict_excluding_outliers = excludeOutliers(info_dict_excluding_outliers)
	
	return info_dict_excluding_outliers

def averageBioactExcludingOutliers(highest_Q_info_dict,n_sd_extreme=3,type_of_averaging='ArithmeticMean'):#averageHsubExcludingOutliers(highest_Q_info_dict):
	
	info_dict_excluding_outliers = excludeOutliers(highest_Q_info_dict,n_sd_extreme,type_of_averaging)
	
	#final_all_errors = checkAllErrorsPresentOrAbsent(info_dict_excluding_outliers)
	
	final_all_bioactVal_vals = [info_dict_excluding_outliers[row_count]['BioActVal'] for row_count in info_dict_excluding_outliers.keys()]
	
	if 'ArithmeticMean' == type_of_averaging:
		bioactVal_average,bioactVal_deviation = getMeanSd(final_all_bioactVal_vals)
	else:
		raise Exception('{}is not supported!'.format(type_of_averaging))
	
	return bioactVal_average,bioactVal_deviation#,final_all_errors

def getMeanSd(list_of_floats):
	############
	assert type([]) == type(list_of_floats)
	assert 0 == len([v for v in list_of_floats if not type(0.5)==type(v)]),"{}".format(list_of_floats)
	############
	
	l_array = np.array(list_of_floats)
	
	return float(np.mean(l_array)),float(np.std(l_array))

def check_getMeanSd():
	m,s = getMeanSd([1.0,2.0,2.5])
	assert 1.83 == round(m,2)
	assert 0.62 == round(s,2)

if __name__ == '__main__':
	print('Peforming checks on the code ...')
	check_getMeanSd()
	
	check_getAveragePlusDeviationEstimateForBioactAfterRemovingOutliers()
	print('All checks passed!')

