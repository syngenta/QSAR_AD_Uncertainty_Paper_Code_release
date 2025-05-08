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
#Copyright (c)  2020-2022 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#This code was written by Zied Hosni whilst working on a Syngenta funded post-doc project at the University of Shefffield.
#The code was adapted from this notebook: https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
#The repo containing that notebook is licensed under the terms of the Creative Commons Attribution 4.0 International license (https://creativecommons.org/licenses/by/4.0/)
#######################
#https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
#Copyright (c) 2021 Greg Landrum
#This notebook is licensed under the terms of the Creative Commons Attribution 4.0 International license (https://creativecommons.org/licenses/by/4.0/)
#######################
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

def standardize(smiles):
    # follows some of the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as err:
        print('Problem converting this smiles: {} - type of problem = {} - error message = {}'.format(smiles,type(err),err))
        mol = None
    if not mol is None:
        Chem.FastFindRings(mol)
        # Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ \
        #                                Chem.SANITIZE_SETAROMATICITY ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_CLEANUPCHIRALITY, updatePropertyCache= True)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol,
                         Chem.SANITIZE_SYMMRINGS | Chem.SANITIZE_SETCONJUGATION | Chem.SANITIZE_SETHYBRIDIZATION)

        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol)

        # if many fragments, get the "parent" (the actual mol we are interested in)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim to to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.
        te = rdMolStandardize.TautomerEnumerator()  # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
        ############################
        #RMR: RDKit documentation indicates the default tautomer canonicalization rules are 'inspired' by the following paper:
        #M. Sitzmann et al., “Tautomerism in Large Databases.”, JCAMD 24:521 (2010) https://doi.org/10.1007/s10822-010-9346-4
        ############################
    else:
        taut_uncharged_parent_clean_mol = None
    return taut_uncharged_parent_clean_mol
