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
########################################################
#Copyright (c) 2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
####################################################
import os,sys,re
import pandas as pd
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
data_dir = os.path.dirname(os.path.dirname(os.path.dirname(top_dir)))
regression_exemplar_metrics_csv = os.path.sep.join([data_dir,'PublicData','ModellingResSummary','Exemplar_Endpoints_Regression_Stats.csv'])


def average_validity_over_folds_and_seeds(df):

    groupby_cols = ['AD Method Parameters (k applicable to dkNN,RDN,UNC and t applicable to Tanimoto)','Endpoint', 'Test Set Name (ignoring fold if applicable)', 'Modelling Algorithm', 'AD Method','AD Subset']

    grouped = df.groupby(groupby_cols, as_index=False)

    averaged_df = grouped.mean()
    return  averaged_df[groupby_cols+['Validity']]


def main():
    print('THE START')

    df = pd.read_csv(regression_exemplar_metrics_csv)

    averaged_validity_df = average_validity_over_folds_and_seeds(df)

    averaged_validity_inside_df = averaged_validity_df[averaged_validity_df['AD Subset'].isin(['Inside'])]

    out_file = re.sub('(\.csv$)','_averaged_validity_inside.xlsx',regression_exemplar_metrics_csv)
    assert not out_file == regression_exemplar_metrics_csv

    averaged_validity_inside_df.to_excel(out_file,index=False,engine='openpyxl')

    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())

