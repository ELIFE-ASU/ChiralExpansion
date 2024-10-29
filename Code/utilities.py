import os
import pandas as pd
import numpy as np

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
    """
    Given a dataframe of compounds, return the number of compounds that are chiral or achiral (depending on state of ischiral_flag)
    """
    count = list()
    for i in range(len(cpd_df.columns)):
        sub_df = kegg_df[kegg_df["compound_id"].isin(cpd_df.index[cpd_df[i] == True].tolist())]
        if ischiral_flag:
            count.append(len(sub_df[sub_df["n_chiral_centers"] != 0]))
        else:
            count.append(len(sub_df[sub_df["n_chiral_centers"] == 0]))

    return pd.DataFrame(count)

def getHeavyAtoms_perGen_fromKegg(cpd_df, kegg_df, ischiral_flag, calc):
    """
    Given a dataframe of compounds, return the either the average of standard deviation (dependant on calc value) of heavy atoms 
    from either chiral or achiral compounds (dependant on ischiral_flag)
    """
    heavy_atoms = list()
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


def getRandomSeedSet(kegg_df, size, chirality="both"):
    """
    Return a seed set of a given size that is ready to be thrown into the network expansion algorithm
    Chirality options: "both" (default), "chiral", "achiral"
    """

    if chirality == "chiral":
        sub_df = kegg_df[kegg_df["n_chiral_centers"] != 0]
    elif chirality == "achiral":
        sub_df = kegg_df[kegg_df["n_chiral_centers"] == 0]
    else:
        sub_df = kegg_df #To ensure remainder of code works

    return sub_df["compound_id"].sample(size).values

def get_ChiralPercentage(cum_chiral, cum_achiral):
    """ Get percentage chiral at a specific location

    Args:
        cum_chiral (list): cumulative sum of chiral compounds over generations
        cum_achiral (list): cumulative sum of achiral compounds over generations
    """
    chiral_percent = []
    for i in range(len(cum_chiral)):
        chiral_percent.append( float(cum_chiral[i]) / (cum_chiral[i] + cum_achiral[i]))
    
    return chiral_percent

def linkArrivalStepHeavyAtoms(arrival_df, kegg_df):
    kegg_df = pd.merge(kegg_df, arrival_df, how="left", left_on="compound_id", right_index=True).dropna()
    kegg_df = kegg_df.rename(columns={"level_0": "arrival"}) #Cleaning up merge notation

    heavy_atom_counts = np.array(kegg_df["n_heavy_atoms"])

    heavy_atom_percentiles = []
    for ha in kegg_df["n_heavy_atoms"].to_list():
        try: #Account for possible NaNs
            #Bins of 20 (10 bins) & 40 (5 bins) heavy atoms
            if ha <= np.percentile(heavy_atom_counts, 20):
                heavy_atom_percentiles.append("0 - 20th")
            elif ha <= np.percentile(heavy_atom_counts, 40):
                heavy_atom_percentiles.append("21 - 40th")
            elif ha <= np.percentile(heavy_atom_counts, 60):
                heavy_atom_percentiles.append("41 - 60th")
            elif ha <= np.percentile(heavy_atom_counts, 80):
                heavy_atom_percentiles.append("61 - 80th")
            else:
                heavy_atom_percentiles.append("81 - 100th")

        except:
            heavy_atom_percentiles.append("0 - 20th")

    kegg_df["heavy atom percentile"] = heavy_atom_percentiles

    return kegg_df

