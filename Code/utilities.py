import os
import pandas as pd

import matplotlib.pyplot as plt
import rdkit.Chem
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors
import rdkit.Chem.rdMolDescriptors
from rdkit import RDLogger
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

plt.rcParams['axes.linewidth'] = 2.0


def n_plot(xlab, ylab, xs=14, ys=14):
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def embed_mol(mol, f_3d=False):
    # Sanitise the molecule
    Chem.SanitizeMol(mol, catchErrors=True)
    # Update the properties
    mol.UpdatePropertyCache()
    # kekulize the molecule
    Chem.Kekulize(mol)
    # Add hydrogens
    mol = rdkit.Chem.AddHs(mol)
    # Embed the molecule in 3D
    if f_3d:
        Chem.EmbedMolecule(mol)
    return mol


def standardize_mol(mol):
    # Standardize the molecule
    nrm = rdMolStandardize.Normalizer()
    nrm.normalizeInPlace(mol)
    # Embed the molecule
    mol = embed_mol(mol)
    return mol


def list_folders(target):
    return [f for f in os.listdir(target) if os.path.isdir(os.path.join(target, f))]


def file_list_all(mypath=None):
    """
    This function generates a list of all files in a specified directory and its subdirectories.
    If no directory is specified, it defaults to the current working directory.

    Parameters:
    mypath (str, optional): The path to the directory. Defaults to None, which means the current working directory.

    Returns:
    list: A list of all files in the specified directory and its subdirectories.
    """
    mypath = mypath or os.getcwd()  # If no path is provided, use the current working directory
    files = []
    # os.walk generates the file names in a directory tree by walking the tree either top-down or bottom-up
    for dirpath, dirnames, filenames in os.walk(mypath):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files


def get_chirality(mol_obj_series):
    chirality_dict = {}
    for cpd, mol in mol_obj_series.items():
        try:
            chirality_dict[cpd] = len(
                rdkit.Chem.FindMolChiralCenters(mol,
                                                useLegacyImplementation=False,
                                                includeUnassigned=True,
                                                includeCIP=False))
        except:
            chirality_dict[cpd] = float('NaN')
    return chirality_dict


def get_heavy_atoms(mol_obj_series):
    heavy_dict = {}
    for cpd, mol in mol_obj_series.items():
        try:
            heavy_dict[cpd] = mol.GetNumHeavyAtoms()
        except:
            heavy_dict[cpd] = float('NaN')
    return heavy_dict

def getCpdCount(cpd_df, kegg_df, ischiral_flag):
    count = list()
    for i in range(len(cpd_df.columns)):
        sub_df = kegg_df[kegg_df["compound_id"].isin(cpd_df.index[cpd_df[i] == True].tolist())]
        if ischiral_flag:
            count.append(len(sub_df[sub_df["n_chiral_centers"] != 0]))
        else:
            count.append(len(sub_df[sub_df["n_chiral_centers"] == 0]))

    return pd.DataFrame(count)

def getHeavyAtoms_perGen_fromKegg(cpd_df, kegg_df, ischiral_flag, calc):
    heavy_atoms = list()
    print(cpd_df)
    for i in range(len(cpd_df.columns)):
        sub_df = kegg_df[kegg_df["compound_id"].isin(cpd_df.index[cpd_df[i] == True].tolist())]
        if ischiral_flag:
            sub_df = sub_df[sub_df["n_chiral_centers"] != 0]
        else:
            sub_df = sub_df[sub_df["n_chiral_centers"] == 0]

        if calc == "mean":
            if len(sub_df) > 0:
                heavy_atoms.append(sub_df["n_heavy_atoms"].mean())
            else:
                heavy_atoms.append(0)
        elif calc == "std":
            if len(sub_df) > 0:
                heavy_atoms.append(sub_df["n_heavy_atoms"].std())
            else:
                heavy_atoms.append(0)

    return pd.DataFrame(heavy_atoms)

