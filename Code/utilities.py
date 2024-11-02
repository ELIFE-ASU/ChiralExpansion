import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
tqdm.pandas()

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

def degrees_PerRxn(rxnString, cpd_degrees):
    rxn = rxnString.split("<=>")
    reactants = rxn[0]
    products = rxn[1]

    reactant_cpds = []
    product_cpds = []

    for pCpd in reactants.split(" "):
        if pCpd.startswith("C"):
            reactant_cpds.append(pCpd)
    for pCpd in products.split(" "):
        if pCpd.startswith("C"):
            product_cpds.append(pCpd)

    for cpd in reactant_cpds:
        if cpd not in cpd_degrees:
            cpd_degrees[cpd] = 1
        else:
            cpd_degrees[cpd] += len(product_cpds)
    for cpd in product_cpds:
        if cpd not in cpd_degrees:
            cpd_degrees[cpd] = 1
        else:
            cpd_degrees[cpd] += len(reactant_cpds)

def get_CpdDegrees(cpd_degrees):
    #atlas_kegg_processed_merged is the current source of truth (right, @Camerian & @Louie?)
    rxn_df = pd.read_csv("../../lets-get-kegg/Data/atlas_kegg_processed_merged.csv.zip", index_col=0, compression="zip")
    rxn_df.drop_duplicates(subset="reaction", inplace=True) #TODO: fully check for duplicates (or wait for lets-get-kegg to be fixed)
    
    rxn_df["reaction"].apply(degrees_PerRxn, args=(cpd_degrees,))

    return cpd_degrees

def scatter_hist(x, y, ax, ax_histy, c):
    """ 
    Scatter plot analysis for Figure 3C
    """
    achiral_color = "#FFC012"
    chiral_color = "#34CCFF"
    max_size = 100 #Maximum size of scatter points (for readability)

    ax_histy.tick_params(axis="y", labelleft=False)

    #Plot by chiral/achiral color, and size by degree
    for i in range(len(x)):
        if y[i] == 0:
            ax.scatter(x[i], y[i], color=achiral_color, s=min(x[i]/10, max_size))
        else:
            ax.scatter(x[i], y[i], color=chiral_color, s=min(x[i]/10, max_size))

    #Set bins to middle of chiral center y-values to line up across axes
    bins = np.arange(-0.5,11.5,1)
    cnts, values, bars = ax_histy.hist(np.array(y)-0.5, orientation='horizontal', color=c, bins=bins, edgecolor="k")

    #Make sure labels are smaller on histogram than scatter plot
    ax_histy.set_xlabel("Count", fontsize=14)
    ax_histy.tick_params(axis='both', which='major', labelsize=14)

    #Change color of histogram bars
    for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
        if i == 0:
            bar.set_facecolor(achiral_color)
        else:
            bar.set_facecolor(chiral_color)

def find_fit(X,Y,color):
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    # print('Slope: {:.6f}'.format(slope),'Intercept: {:.6f}'.format(intercept),'R-squared: {:.6f}'.format(r_value**2),'Std_err: {:.6f}'.format(std_err))
    x_dummy = np.linspace(start=0, stop=8000, num=100)
    plt.plot(x_dummy,slope*x_dummy + intercept,color=color)
    plt.fill_between(x_dummy,slope*x_dummy+intercept-std_err,slope*x_dummy+intercept+std_err,color=color, alpha=0.2)

def make_AUC_dashes(u, label, c, ls, ax):
    yvals = list(u.values())
    xvals = np.linspace(0, 1, num=len(yvals))

    #Calculate AUC using composite travezoidal rule
    area = np.trapz(yvals, dx=xvals[1])

    ax.plot(xvals, yvals, label = label, color=c, 
    linestyle=ls, linewidth=4)

    print(label + " AUC = " + str(round(area, 4)))

    return yvals

def find_chiral(u, df):
    achiral = {}
    chiral = {}

    for _, row in df.iterrows():
        if row["compound_id"] in u:
            if row["n_chiral_centers"] > 0:
                chiral[row["compound_id"]] = u[row["compound_id"]]
            else:
                achiral[row["compound_id"]] = u[row["compound_id"]]

    return chiral, achiral

def get_chiral_difference(chiral_values, achiral_values, c, l, ax):

    #Step 1: get rank-order percentile values
    m_chiral_percentiles = []
    for i in np.linspace(0, len(chiral_values)-1, 100):
        m_chiral_percentiles.append(chiral_values[int(i)])

    m_achiral_percentiles = []
    for i in np.linspace(0, len(achiral_values)-1, 100):
        m_achiral_percentiles.append(achiral_values[int(i)])

    #Step 2: find difference
    m_diffs = list(np.array(m_chiral_percentiles) - np.array(m_achiral_percentiles))

    #Step 3: plot
    label = ax.plot(m_diffs, color=c, linewidth=3, label=l)

    return label

def check_rxn_chirality(rxnString, keggC_df, rxn_chiral_count):
    rxn = rxnString.split("<=>")
    reactants = rxn[0]
    products = rxn[1]

    reactant_cpds = []
    product_cpds = []

    for pCpd in reactants.split(" "):
        if pCpd.startswith("C"):
            reactant_cpds.append(pCpd)
    for pCpd in products.split(" "):
        if pCpd.startswith("C"):
            product_cpds.append(pCpd)

    reactant_chiral_count = 0
    product_chiral_count = 0

    for cpd in reactant_cpds:
        if keggC_df.loc[cpd]["n_chiral_centers"]:
            reactant_chiral_count += 1
    for cpd in product_cpds:
        if keggC_df.loc[cpd]["n_chiral_centers"]:
            product_chiral_count += 1

    chiral_str = f"{reactant_chiral_count}_{product_chiral_count}"

    if chiral_str not in rxn_chiral_count:
        rxn_chiral_count[chiral_str] = 1
    else:
        rxn_chiral_count[chiral_str] += 1

    return pd.Series([reactant_chiral_count, product_chiral_count], index=["Reactant Chiral Count", "Product Chiral Count"])

def get_heatmap_array(keggC_df):
    keggC_df = keggC_df.set_index("compound_id")
    rxn_chiral_count = {}
    heatmap_array = []

    #atlas_kegg_processed_merged is the current source of truth (right, @Camerian & @Louie?)    
    rxn_df = pd.read_csv("../../lets-get-kegg/Data/atlas_kegg_processed_merged.csv.zip", index_col=0, compression="zip")

    print("- Calculating Reaction Chirality -")
    rxn_df[["Reactant Chiral Count", "Product Chiral Count"]] = rxn_df["reaction"].progress_apply(check_rxn_chirality, args=(keggC_df, rxn_chiral_count))

    for i in range(5):
        row = []
        for j in range(5):
            chiral_str = f"{i}_{j}"
            if chiral_str in rxn_chiral_count:
                row.append(rxn_chiral_count[f"{i}_{j}"])
            else:
                row.append(0)
        heatmap_array.append(row)

    return heatmap_array, rxn_df

def get_chiral_modulations(stepRxn_df, rxn_df):
    chiral_modulating = []
    chiral_conserving = []
    achiral_conserving = []

    for _, row in stepRxn_df.iterrows():
        sub_rxn_df = rxn_df[rxn_df["id"].isin(row["reactions"])]
        chiral_modulating.append(len(sub_rxn_df[sub_rxn_df["Reactant Chiral Count"] != sub_rxn_df["Product Chiral Count"]]))
        chiral_conserving.append(len(sub_rxn_df[sub_rxn_df["Reactant Chiral Count"] == sub_rxn_df["Product Chiral Count"]]))

        achiral_conserving_df = sub_rxn_df[sub_rxn_df["Reactant Chiral Count"] == 0]
        achiral_conserving.append(len(achiral_conserving_df[achiral_conserving_df["Reactant Chiral Count"] == achiral_conserving_df["Product Chiral Count"]]))

    return chiral_modulating, chiral_conserving, achiral_conserving

