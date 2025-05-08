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
#####################################
#This code was inspired by consulting the following sources, but the code was not directly adapted from here:
#https://practicalcheminformatics.blogspot.com/2019/11/visualizing-chemical-space.html
#https://github.com/PatWalters/workshop/blob/master/predictive_models/2_visualizing_chemical_space.ipynb
#https://github.com/rdkit/rdkit-tutorials/blob/master/notebooks/005_Chemical_space_analysis_and_visualization.ipynb
#https://scikit-learn.org/1.0/modules/generated/sklearn.decomposition.PCA.html
#https://scikit-learn.org/1.0/modules/generated/sklearn.manifold.TSNE.html
#https://seaborn.pydata.org/generated/seaborn.scatterplot.html
#https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html#matplotlib.axes.Axes.scatter
#https://seaborn.pydata.org/generated/seaborn.move_legend.html
#https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html#matplotlib.axes.Axes.legend
#####################################

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def get_tsne_coords_from_fp_bit_vector_array(fp_bit_vector_array,use_pca_first=True,pca_components=50,tsne_components=2,tsne_perplexity=50,tsne_n_iter=5000,tsne_method='exact',tsne_metric="jaccard",learning_rate=200.0,rand_state=42):
    

    #---------------------
    assert isinstance(fp_bit_vector_array,np.ndarray),type(fp_bit_vector_array)
    #--------------------

    X = fp_bit_vector_array

    if use_pca_first:
        pca_transformer = PCA(n_components=pca_components,random_state=rand_state)

        X_ready_for_tsne = pca_transformer.fit_transform(X)

        percentage_of_var_in_pca_components = round(100*sum(pca_transformer.explained_variance_ratio_.tolist()),2)
    else:

        X_ready_for_tsne = X

        percentage_of_var_in_pca_components = None
    
    tsne_transformer = TSNE(n_components=tsne_components,n_iter=tsne_n_iter,method=tsne_method,perplexity=tsne_perplexity,metric=tsne_metric,learning_rate=learning_rate,random_state=rand_state)

    X_tsne_coords = tsne_transformer.fit_transform(X_ready_for_tsne)

    return X_tsne_coords,percentage_of_var_in_pca_components

def add_tsne_coords_to_dataset_with_fp_bit_vector_array_and_subset_labels(dataset_df,subset_col,tsne_coords_column_names=['TSNE1','TSNE2'],use_pca_first=True,pca_components=50,tsne_components=2,tsne_perplexity=50,tsne_n_iter=5000,tsne_method='exact',tsne_metric="jaccard",learning_rate=200.0,rand_state=42):    
    #--------------------------
    assert isinstance(dataset_df,pd.DataFrame),type(dataset_df)
    assert dataset_df.index.tolist() == list(range(dataset_df.shape[0]))
    assert len(tsne_coords_column_names) == tsne_components
    assert len(tsne_coords_column_names) == 2
    #--------------------------

    fp_bit_vector_array = np.array(dataset_df.drop([subset_col],axis=1))

    assert fp_bit_vector_array.shape[0] == dataset_df.shape[0]
    assert fp_bit_vector_array.shape[1] == (dataset_df.shape[1]-1)


    X_tsne_coords,percentage_of_var_in_pca_components = get_tsne_coords_from_fp_bit_vector_array(fp_bit_vector_array,use_pca_first,pca_components,tsne_components,tsne_perplexity,tsne_n_iter,tsne_method,tsne_metric,learning_rate,rand_state)


    df_with_tsne_coords = pd.DataFrame(X_tsne_coords)

    #------------------------
    assert len(tsne_coords_column_names) == df_with_tsne_coords.shape[1]
    assert df_with_tsne_coords.shape[1] == X_tsne_coords.shape[1]
    #------------------------

    df_with_tsne_coords.columns = tsne_coords_column_names

    #--------------------------
    assert df_with_tsne_coords.index.tolist() == dataset_df.index.tolist()
    #--------------------------

    df_with_tsne_coords.insert(0,subset_col,dataset_df[subset_col])

    return df_with_tsne_coords,percentage_of_var_in_pca_components

def get_tsne_plot_of_dataset_with_fp_bit_vector_array_and_subset_labels(plot_name_prefix,title_prefix,dataset_df,subset_col,tsne_coords_column_names=['TSNE1','TSNE2'],use_pca_first=True,pca_components=100,tsne_components=2,tsne_perplexity=30.0,tsne_n_iter=1000,tsne_method='barnes_hut',tsne_metric="euclidean",learning_rate=200.0,rand_state=42,opaqueness=0.20,edgecolor=None):

    df_with_tsne_coords,percentage_of_var_in_pca_components = add_tsne_coords_to_dataset_with_fp_bit_vector_array_and_subset_labels(dataset_df,subset_col,tsne_coords_column_names,use_pca_first,pca_components,tsne_components,tsne_perplexity,tsne_n_iter,tsne_method,tsne_metric,learning_rate,rand_state)

    sns.set_style("white")

    matplot_lib_axes_obj = sns.scatterplot(data=df_with_tsne_coords,x=tsne_coords_column_names[0], y=tsne_coords_column_names[1], hue=subset_col,edgecolors=edgecolor,alpha=opaqueness)

    

    
    if percentage_of_var_in_pca_components is None:
        title = title_prefix
    else:
        title = f'{title_prefix} - PCA pre-processing captures {percentage_of_var_in_pca_components}% of structural variance'
    
    #plt.title(title)

    sns.move_legend(bbox_to_anchor=(0.5, 1),obj=matplot_lib_axes_obj,loc='lower center',ncol=len(df_with_tsne_coords[subset_col].unique().tolist()),frameon=False,title=title,fontsize='x-small')


    plt.tight_layout()
    
    plot_name = f'{plot_name_prefix}_pca={use_pca_first}_pcaN={pca_components}_tsneP={tsne_perplexity}_tsneITER={tsne_n_iter}_tsneMETH={tsne_method}_tsneM={tsne_metric}_tsneLR={learning_rate}_rs={rand_state}.tiff'

    plt.savefig(plot_name, transparent=True)
    plt.close('all')
    plt.clf()

def prepare_X_and_ids_df_for_TSNE(X_and_ids_df,subset_col,subset_name,id_col,no_fp_bit_vector_cols=1024):

    dataset_df_ready_for_tsne_precursor = X_and_ids_df.drop([id_col],axis=1)

    #-----------------------------
    assert dataset_df_ready_for_tsne_precursor.shape[1] == no_fp_bit_vector_cols
    #-----------------------------

    dataset_df_ready_for_tsne_precursor.insert(0,subset_col,[subset_name]*X_and_ids_df.shape[0])

    dataset_df_ready_for_tsne = dataset_df_ready_for_tsne_precursor

    return dataset_df_ready_for_tsne