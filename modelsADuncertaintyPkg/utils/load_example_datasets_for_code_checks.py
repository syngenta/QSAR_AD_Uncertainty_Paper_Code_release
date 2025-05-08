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
#######################
#Copyright (c) 2020-2022 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#######################
import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer,fetch_california_housing

def getXthenYDataFrame(data_bunch):
    #'bunch' - a 'dictionary-like' object [https://scikit-learn.org/stable/datasets/index.html]
    
    if not type(data_bunch['feature_names']) == type([]):
        fns=data_bunch['feature_names'].tolist()
    else:
        fns=data_bunch['feature_names']
    
    data_x = data_bunch['data']
    data_y = data_bunch['target']
    
    df = pd.DataFrame(data_x)
    df=df.assign(data_y=pd.Series(data_y).values)
    
    return df,fns

BCWD_x_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
BCWD_y_name = 'Class'

def generateExampleDatasetForChecking_BCWD(write_to_csv=False):
    data_bunch = load_breast_cancer()
    
    df,fns = getXthenYDataFrame(data_bunch)
    
    if not df.columns.values.tolist() == list(range(0,30))+['data_y']: raise Exception(df.columns.values.tolist())
    
    df.insert(0,'ID',pd.Series(range(0,df.shape[0])))
    
    if not df.columns.values.tolist() == ['ID']+list(range(0,30))+['data_y']: raise Exception(df.columns.values.tolist())
    
    df=df.rename(mapper=dict(zip(df.columns.values.tolist(),['ID']+fns[:]+[BCWD_y_name])),axis=1)
    
    if not df.columns.values.tolist() == ['ID']+BCWD_x_names+[BCWD_y_name]: raise Exception(df.columns.values.tolist())
    
    if not 569 == df.shape[0]: raise Exception(df.shape[0])
    
    if not 32 == df.shape[1]: raise Exception(df.shape[1])
    
    if write_to_csv:
        df.to_csv('Classification_BCWD.csv',index=False)
    
    return df

Housing_x_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
Housing_y_name = 'Endpoint'

def generateExampleDatasetForChecking_Housing(write_to_csv=False):
    data_bunch = fetch_california_housing()
    
    df,fns = getXthenYDataFrame(data_bunch)
    
    if not df.columns.values.tolist() == list(range(0,8))+['data_y']: raise Exception(df.columns.values.tolist())
    
    df.insert(0,'ID',pd.Series(range(0,df.shape[0])))
    
    if not df.columns.values.tolist() == ['ID']+list(range(0,8))+['data_y']: raise Exception(df.columns.values.tolist())
    
    df=df.rename(mapper=dict(zip(df.columns.values.tolist(),['ID']+fns[:]+[Housing_y_name])),axis=1)
    
    if not df.columns.values.tolist() == ['ID']+Housing_x_names+[Housing_y_name]: raise Exception(df.columns.values.tolist())
    
    if not 20640 == df.shape[0]: raise Exception(df.shape[0])
    
    if not 10 == df.shape[1]: raise Exception(df.shape[1])
    
    if write_to_csv:
        df.to_csv('Regression_Housing.csv',index=False)
    
    return df

if __name__ == '__main__':
    generateExampleDatasetForChecking_BCWD(write_to_csv=True)
    generateExampleDatasetForChecking_Housing(write_to_csv=True)
