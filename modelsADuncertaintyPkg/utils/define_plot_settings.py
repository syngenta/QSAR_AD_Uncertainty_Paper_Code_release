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
##############################
import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib.ticker import MaxNLocator
matplotlib.use('Agg')
import seaborn as sb
#https://seaborn.pydata.org/generated/seaborn.set_context.html
#https://seaborn.pydata.org/generated/seaborn.plotting_context.html#seaborn.plotting_context
#sb.plotting_context("paper",rc={"font.size":12,"axes.labelsize":16,"axes.titlesize":18})
#https://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme
#https://www.codecademy.com/article/seaborn-design-i
#https://seaborn.pydata.org/tutorial/color_palettes.html#palette-tutorial
over_ride_context_rc={'axes.labelcolor': 'dimgrey',"font.size":10,"axes.titlesize":15,"axes.labelsize":16}#,'axes.axisbelow': False}
sb.set_theme(context="notebook",style="white",palette="muted",font='sans-serif',font_scale=1, color_codes=True,rc=over_ride_context_rc)
