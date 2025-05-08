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
#Copyright (c)  2022-2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#######################
#
import matplotlib.pyplot as pp
import seaborn as sb
from .define_plot_settings import *
import numpy as np
import statistics

def add_identity(ax,color,ls,style='darkgrid'):
    ##############################
    #https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.axes.Axes.axline.html
    #https://matplotlib.org/3.3.0/gallery/subplots_axes_and_figures/axhspan_demo.html#sphx-glr-gallery-subplots-axes-and-figures-axhspan-demo-py
    #May not work in older versions of matplotlib
    ###############################
    
    sb.set_style(style)
    
    xy1 = (0,0)
    xy2 = (1,1)
    
    ax.axline(xy1, xy2, color=color,ls=ls)
    
    return ax

def update_label_with_mean_and_stdev(prefix_of_label,distribution_list):
    #manually checked consistency of statistics calculation with summarize_dataset_stats_before_and_after_parsing.py:
    mean = np.mean(distribution_list)
    sd = np.std(distribution_list,ddof=0)
    label = f'{prefix_of_label}: mean = {mean:.2f}, sd = {sd:.2f}'
    return label

def create_plot_comparing_two_distributions(plot_file_prefix,title,prefix_label_1,prefix_label_2,dist_list_1,dist_list_2,x_label,y_label,density=True,colour_1='blue',colour_2='red',num_bins=50,add_median_line=True,add_mean_and_stdev=True,alpha=0.3,legend_size=5):
    #Will need to manually make sure y_label and density are consistent!

    if add_mean_and_stdev:
        label_1 = update_label_with_mean_and_stdev(prefix_of_label=prefix_label_1,distribution_list=dist_list_1)

        label_2 = update_label_with_mean_and_stdev(prefix_of_label=prefix_label_2,distribution_list=dist_list_2)

    plt.hist(dist_list_1,  num_bins,  facecolor=colour_1,  alpha=alpha,  label=label_1,  density=density)
    
    plt.hist(dist_list_2,  num_bins,  facecolor=colour_2,  alpha=alpha,  label=label_2,  density=density)
    
    plt.title(title)
    
    if add_median_line:
        median_1 = statistics.median(dist_list_1)
        median_2 = statistics.median(dist_list_2)
    
        plt.axvline(median_1,  color=colour_1,  linestyle='dashed',  linewidth=1)
        plt.axvline(median_2,  color=colour_2,  linestyle='dashed',  linewidth=1)
    
    plt.legend(prop={'size': legend_size})
    
    plt.xlabel(x_label)
    
    plt.ylabel(y_label)
    
    plot_file = f'{plot_file_prefix}_fract={density}.png'
      
    plt.tight_layout()
    plt.savefig(plot_file,  dpi=300,  bbox_inches="tight",  transparent=True)
    plt.close()
    plt.clf()
