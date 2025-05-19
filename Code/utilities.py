import json
import math
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors
import scipy.stats as stats
from matplotlib.colors import Normalize, to_hex, BoundaryNorm
from rdkit import RDLogger
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from requests import Session
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry

tqdm.pandas()
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

import network_expansion

plt.rcParams['axes.linewidth'] = 2.0


def n_plot(xlab, ylab, xs=14, ys=14):
    """
    Configure the plot with specified x and y labels, and customize tick parameters.

    Parameters:
    xlab (str): The label for the x-axis.
    ylab (str): The label for the y-axis.
    xs (int, optional): Font size for the x-axis label. Defaults to 14.
    ys (int, optional): Font size for the y-axis label. Defaults to 14.

    Returns:
    None
    """
    plt.minorticks_on()
    plt.tick_params(axis='both',
                    which='major',
                    labelsize=ys - 2,
                    direction='in',
                    length=6,
                    width=2)
    plt.tick_params(axis='both',
                    which='minor',
                    labelsize=ys - 2,
                    direction='in',
                    length=4,
                    width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def embed_mol(mol, f_3d=False):
    """
    Embed a molecule with optional 3D embedding.

    Parameters:
    mol (rdkit.Chem.Mol): The molecule to be embedded.
    f_3d (bool, optional): If True, embed the molecule in 3D. Defaults to False.

    Returns:
    rdkit.Chem.Mol: The embedded molecule.
    """
    # Sanitise the molecule
    Chem.SanitizeMol(mol, catchErrors=True)
    # Update the properties
    mol.UpdatePropertyCache()
    # Kekulize the molecule
    Chem.Kekulize(mol)
    # Add hydrogens
    mol = rdkit.Chem.AddHs(mol)
    # Embed the molecule in 3D
    if f_3d:
        Chem.EmbedMolecule(mol)
    return mol


def standardize_mol(mol):
    """
    Standardize the given molecule and embed it.

    Parameters:
    mol (rdkit.Chem.Mol): The molecule to be standardized and embedded.

    Returns:
    rdkit.Chem.Mol: The standardized and embedded molecule.
    """
    # Standardize the molecule
    nrm = rdMolStandardize.Normalizer()
    nrm.normalizeInPlace(mol)
    # Embed the molecule
    mol = embed_mol(mol)
    return mol


def list_folders(target):
    """
    List all folders in the specified directory.

    Parameters:
    target (str): The path to the directory.

    Returns:
    list: A list of folder names in the specified directory.
    """
    return [
        f for f in os.listdir(target) if os.path.isdir(os.path.join(target, f))
    ]


def file_list_all(mypath=None):
    """
    This function generates a list of all files in a specified directory and its subdirectories.
    If no directory is specified, it defaults to the current working directory.

    Parameters:
    mypath (str, optional): The path to the directory. Defaults to None, which means the current working directory.

    Returns:
    list: A list of all files in the specified directory and its subdirectories.
    """
    mypath = mypath or os.getcwd(
    )  # If no path is provided, use the current working directory
    files = []
    # os.walk generates the file names in a directory tree by walking the tree either top-down or bottom-up
    for dirpath, dirnames, filenames in os.walk(mypath):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files


def get_chirality(mol_obj_series):
    """
    Determine the chirality of molecules in a given series.

    Parameters:
    mol_obj_series (pandas.Series): A series where the index is compound IDs and the values are RDKit molecule objects.

    Returns:
    dict: A dictionary where the keys are compound IDs and the values are the number of chiral centers in each molecule.
    """
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
    """
    Get the number of heavy atoms for each molecule in the given series.

    Parameters:
    mol_obj_series (pandas.Series): A series where the index is compound IDs and the values are RDKit molecule objects.

    Returns:
    dict: A dictionary where the keys are compound IDs and the values are the number of heavy atoms in each molecule.
    """
    heavy_dict = {}
    for cpd, mol in mol_obj_series.items():
        try:
            heavy_dict[cpd] = mol.GetNumHeavyAtoms()
        except:
            heavy_dict[cpd] = float('NaN')
    return heavy_dict


def get_compound_chiral_count(cpd_df, kegg_df, chiral_flag):
    """
    Given a dataframe of compounds, return the number of compounds that are chiral or achiral (depending on the state of chiral_flag).

    Parameters:
    cpd_df (pandas.DataFrame): DataFrame containing compound data.
    kegg_df (pandas.DataFrame): DataFrame containing KEGG compound data.
    chiral_flag (bool): Flag indicating whether to count chiral (True) or achiral (False) compounds.

    Returns:
    pandas.DataFrame: DataFrame containing the count of chiral or achiral compounds for each column in cpd_df.
    """
    if 'compound_id' in kegg_df.columns:
        kegg_df = kegg_df.set_index("compound_id")
    if 'is_chiral' not in kegg_df.columns:
        kegg_df["is_chiral"] = kegg_df["n_chiral_centers"].gt(0)
    if chiral_flag:
        count = cpd_df[kegg_df.loc[cpd_df.index, 'is_chiral']].count().to_frame()
    else:
        count = cpd_df[~kegg_df.loc[cpd_df.index, 'is_chiral']].count().to_frame()
    return count

def get_heavy_atoms_per_gen_from_kegg(cpd_df, kegg_df, chiral_flag, calc):
    """
    Given a dataframe of compounds, return either the average or standard deviation (depending on the calc value) of heavy atoms
    from either chiral or achiral compounds (depending on ischiral_flag).

    Parameters:
    cpd_df (pandas.DataFrame): DataFrame containing compound data.
    kegg_df (pandas.DataFrame): DataFrame containing KEGG compound data.
    ischiral_flag (bool): Flag indicating whether to consider chiral (True) or achiral (False) compounds.
    calc (str): Calculation type, either "mean" for average or "std" for standard deviation.

    Returns:
    pandas.DataFrame: DataFrame containing the calculated values of heavy atoms for each column in cpd_df.
    """
    heavy_atoms = list()
    for i in range(len(cpd_df.columns)):
        sub_df = kegg_df[kegg_df["compound_id"].isin(
            cpd_df.index[cpd_df[i] == True].tolist())]
        if chiral_flag:
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


def get_random_seed_set(kegg_df, size, random_state, chirality="both"):
    """
    Return a seed set of a given size that is ready to be thrown into the network expansion algorithm.

    Parameters:
    kegg_df (pandas.DataFrame): DataFrame containing KEGG compound data.
    size (int): The size of the seed set to be returned.
    random_state (int): Random state for reproducibility.
    chirality (str, optional): Chirality options, either "both" (default), "chiral", or "achiral".

    Returns:
    numpy.ndarray: An array of compound IDs representing the seed set.
    """

    if 'compound_id' not in kegg_df.columns:
        kegg_df = kegg_df.reset_index()
    if chirality == "chiral":
        sub_df = kegg_df[kegg_df["n_chiral_centers"] != 0]
    elif chirality == "achiral":
        sub_df = kegg_df[kegg_df["n_chiral_centers"] == 0]
    else:
        sub_df = kegg_df  # To ensure remainder of code works

    return sub_df["compound_id"].sample(size, random_state=random_state).values


def get_chiral_percentage(cum_chiral, cum_achiral):
    """
    Get percentage chiral at a specific location.

    Parameters:
    cum_chiral (list): Cumulative sum of chiral compounds over generations.
    cum_achiral (list): Cumulative sum of achiral compounds over generations.

    Returns:
    list: A list of chiral percentages at each location.
    """
    chiral_percent = []
    for i in range(len(cum_chiral)):
        chiral_percent.append(
            float(cum_chiral[i]) / (cum_chiral[i] + cum_achiral[i]))

    return chiral_percent


def get_luca_percentage(df):
    """
    Given a dataframe with sets of compounds per network expansion step, calculate the percentage of compounds
    that show up in LUCA compounds.

    Parameters:
    df (pandas.DataFrame): DataFrame containing sets of compounds per network expansion step.

    Returns:
    list: A list of percentages representing the proportion of LUCA compounds at each network expansion step.
    """
    fpath = "../Data/Luca_ec_compound_unique_list_clean.json"
    with open(fpath) as f:
        data = json.load(f)
        luca_cpds = data["Luca"]

    percent_luca = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        luca_count = 0
        for cpd in row["compounds"]:
            if cpd in luca_cpds:
                luca_count += 1
        percent_luca.append((float(luca_count) / len(luca_cpds)) * 100)  # Put into a percentage (0-100) scale

    return percent_luca


def link_arrival_step_heavy_atoms(arrival_df, kegg_df):
    """
    Link arrival steps with heavy atom counts and categorize them into percentiles.

    Parameters:
    arrival_df (pandas.DataFrame): DataFrame containing arrival step information.
    kegg_df (pandas.DataFrame): DataFrame containing KEGG compound data.

    Returns:
    pandas.DataFrame: Updated DataFrame with heavy atom percentiles.
    """
    kegg_df = pd.merge(kegg_df,
                       arrival_df,
                       how="left",
                       left_on="compound_id",
                       right_index=True)

    kegg_df = kegg_df.rename(columns={"level_0": "arrival"})  # Cleaning up merge notation
    df = kegg_df.dropna(subset=['arrival'])  # Removing all cpds not found in expansion
    heavy_atom_counts = np.array(df["n_heavy_atoms"])
    heavy_atom_percentiles = []

    for ha in df["n_heavy_atoms"].to_list():
        try:  # Account for possible NaNs
            # Bins of 20 (10 bins) & 40 (5 bins) heavy atoms
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

    df["heavy atom percentile"] = heavy_atom_percentiles

    return df


def degrees_per_reaction(reaction_string, cpd_degrees):
    """
    Calculate the degree of each compound in a reaction and update the compound degrees dictionary.

    Parameters:
    rxnString (str): A string representing the reaction in the format "reactants <=> products".
    cpd_degrees (dict): A dictionary where the keys are compound IDs and the values are their degrees.

    Returns:
    None
    """
    rxn = reaction_string.split("<=>")
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


def get_compound_degrees(rxn_df):
    """
    Calculate each compound's degree, as the number of unique compounds opposite it in a reaction.

    Parameters:
    rxn_df (pandas.DataFrame): DataFrame containing reaction data (either full DB or a subset).

    Returns:
    pd.Series: pandas.Series of compound degrees WRT the input rxn_df.
    """
    rxn_df.drop_duplicates(subset="reaction", inplace=True)
    # Get the compounds on each side of each reaction.
    df = rxn_df['reaction'].str.split('<=>', expand=True).apply(lambda x: x.str.findall(r'(C\d{5})'))
    # Connect each compound to its neighbors opposite it in a reaction.
    df = df.explode(0).explode(1)
    # Merge in- and out-edges.
    df = df.assign(a=df.max(axis=1), b=df.min(axis=1)).drop(columns=[0, 1])
    # Remove self-loops and duplicate edges.
    df = df.query("a.ne(b)").drop_duplicates()
    # Count the number of unique compounds opposite each compound.
    cpd_degrees = df.stack().value_counts()

    return cpd_degrees


def scatter_hist(x, y, ax, ax_histy, c):
    """
    Scatter plot analysis for Figure 3C.

    Parameters:
    x (list or numpy.ndarray): Data for the x-axis.
    y (list or numpy.ndarray): Data for the y-axis.
    ax (matplotlib.axes.Axes): The main axes for the scatter plot.
    ax_histy (matplotlib.axes.Axes): The axes for the histogram.
    c (str): Color for the histogram bars.

    Returns:
    None
    """
    achiral_color = "#FFC012"
    chiral_color = "#34CCFF"
    max_size = 100  # Maximum size of scatter points (for readability)

    ax_histy.tick_params(axis="y", labelleft=False)

    # Plot by chiral/achiral color, and size by degree
    for i in range(len(x)):
        if y[i] == 0:
            ax.scatter(x[i],
                       y[i],
                       color=achiral_color,
                       s=min(x[i] / 10, max_size))
        else:
            ax.scatter(x[i],
                       y[i],
                       color=chiral_color,
                       s=min(x[i] / 10, max_size))

    # Set bins to middle of chiral center y-values to line up across axes
    bins = np.arange(-0.5, 11.5, 1)
    cnts, values, bars = ax_histy.hist(np.array(y) - 0.5,
                                       orientation='horizontal',
                                       color=c,
                                       bins=bins,
                                       edgecolor="k")

    # Make sure labels are smaller on histogram than scatter plot
    ax_histy.set_xlabel("Count", fontsize=14)
    ax_histy.tick_params(axis='both', which='major', labelsize=14)

    # Change color of histogram bars
    for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
        if i == 0:
            bar.set_facecolor(achiral_color)
        else:
            bar.set_facecolor(chiral_color)


def find_fit(X, Y, color):
    """
    Perform linear regression on the given data and plot the fit line with confidence interval.

    Parameters:
    X (list or numpy.ndarray): Data for the x-axis.
    Y (list or numpy.ndarray): Data for the y-axis.
    color (str): Color for the fit line and confidence interval.

    Returns:
    None
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    # print('Slope: {:.6f}'.format(slope),'Intercept: {:.6f}'.format(intercept),'R-squared: {:.6f}'.format(r_value**2),'Std_err: {:.6f}'.format(std_err))
    x_dummy = np.linspace(start=0, stop=8000, num=100)
    plt.plot(x_dummy, slope * x_dummy + intercept, color=color)
    plt.fill_between(x_dummy,
                     slope * x_dummy + intercept - std_err,
                     slope * x_dummy + intercept + std_err,
                     color=color,
                     alpha=0.2)


def make_auc_dashes(u, label, c, ls, ax):
    """
    Plot the AUC (Area Under the Curve) dashes for the given data.

    Parameters:
    u (dict): A dictionary where the keys are x-values and the values are y-values.
    label (str): The label for the plot.
    c (str): The color for the plot line.
    ls (str): The linestyle for the plot line.
    ax (matplotlib.axes.Axes): The axes on which to plot.

    Returns:
    list: A list of y-values used in the plot.
    """
    yvals = list(u.values())
    xvals = np.linspace(0, 1, num=len(yvals))

    # Calculate AUC using composite trapezoidal rule
    area = np.trapz(yvals, dx=xvals[1])

    ax.plot(xvals, yvals, label=label, color=c, linestyle=ls, linewidth=4)

    print(label + " AUC = " + str(round(area, 4)))

    return yvals


def find_chiral(u, df):
    """
    Categorize compounds into chiral and achiral based on the number of chiral centers.

    Parameters:
    u (dict): A dictionary where the keys are compound IDs and the values are some associated data.
    df (pandas.DataFrame): DataFrame containing compound data, including 'compound_id' and 'n_chiral_centers' columns.

    Returns:
    tuple: Two dictionaries, one for chiral compounds and one for achiral compounds.
    """
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
    """
    Calculate the difference between chiral and achiral values and plot the result.

    Parameters:
    chiral_values (list or numpy.ndarray): List of chiral values.
    achiral_values (list or numpy.ndarray): List of achiral values.
    c (str): Color for the plot line.
    l (str): Label for the plot line.
    ax (matplotlib.axes.Axes): The axes on which to plot.

    Returns:
    list: The label of the plot line.
    """
    # Step 1: get rank-order percentile values
    m_chiral_percentiles = []
    for i in np.linspace(0, len(chiral_values) - 1, 100):
        m_chiral_percentiles.append(chiral_values[int(i)])

    m_achiral_percentiles = []
    for i in np.linspace(0, len(achiral_values) - 1, 100):
        m_achiral_percentiles.append(achiral_values[int(i)])

    # Step 2: find difference
    m_diffs = list(
        np.array(m_chiral_percentiles) - np.array(m_achiral_percentiles))

    # Step 3: plot
    label = ax.plot(m_diffs, color=c, linewidth=3, label=l)

    return label


def check_rxn_chirality(reactions_string, kegg_c_df, rxn_chiral_count):
    """
    Check the chirality of reactants and products in a reaction and update the chiral count.

    Parameters:
    rxnString (str): A string representing the reaction in the format "reactants <=> products".
    keggC_df (pandas.DataFrame): DataFrame containing KEGG compound data, including 'n_chiral_centers' column.
    rxn_chiral_count (dict): Dictionary to store the count of reactions based on their chiral counts.

    Returns:
    pandas.Series: A series with the counts of chiral reactants and products.
    """
    rxn = reactions_string.split("<=>")
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

    kegg_cpds = set(list(kegg_c_df.index))

    for cpd in reactant_cpds:
        if cpd in kegg_cpds:
            if kegg_c_df.loc[cpd]["n_chiral_centers"]:
                reactant_chiral_count += 1
    for cpd in product_cpds:
        if cpd in kegg_cpds:
            if kegg_c_df.loc[cpd]["n_chiral_centers"]:
                product_chiral_count += 1

    chiral_str = f"{reactant_chiral_count}_{product_chiral_count}"

    if chiral_str not in rxn_chiral_count:
        rxn_chiral_count[chiral_str] = 1
    else:
        rxn_chiral_count[chiral_str] += 1

    return pd.Series([reactant_chiral_count, product_chiral_count],
                     index=["Reactant Chiral Count", "Product Chiral Count"])


def get_heatmap_array(kegg_c_df, rxn_df):
    """
    Generate a heatmap array based on the chirality of reactions.

    Parameters:
    keggC_df (pandas.DataFrame): DataFrame containing KEGG compound data, including 'compound_id' and 'n_chiral_centers' columns.
    rxn_df (pandas.DataFrame): DataFrame containing reaction data, including 'left' and 'right' columns.
    
    Returns:
    tuple: A tuple containing the heatmap array and the updated reaction DataFrame.
    """
    # Ensure compound_id is the index
    if 'compound_id' in kegg_c_df.columns:
        kegg_c_df = kegg_c_df.set_index("compound_id")

    print("- Calculating Reaction Chirality -")
    # Count the chiral compounds on either side of each reaction
    chiral_compounds = set(kegg_c_df.query('n_chiral_centers > 0').index)
    chiral_counts = rxn_df[['left', 'right']].map(lambda x: len(x.intersection(chiral_compounds)))
    chiral_counts.columns = ['Reactant Chiral Count', 'Product Chiral Count']
    rxn_df = rxn_df.join(chiral_counts)
    print("- Done -")

    # Generate an array for the heatmap
    lim = 5
    df = rxn_df.query('`Reactant Chiral Count` < @lim and `Product Chiral Count` < @lim').copy(deep=True)
    df = df.groupby(['Reactant Chiral Count', 'Product Chiral Count'], as_index=False).size()
    df = df.pivot_table(values='size', index='Reactant Chiral Count', columns='Product Chiral Count')
    heatmap_array = df.fillna(0).astype(int).to_numpy().tolist()

    return heatmap_array, rxn_df


def get_chiral_modulations(step_reaction_df, rxn_df):
    """
    Calculate the number of chiral-modulating, chiral-conserving, and achiral-conserving reactions.

    Parameters:
    step_reaction_df (pandas.DataFrame): DataFrame containing step reaction data, with each row representing a step and a column 'reactions' containing reaction IDs.
    rxn_df (pandas.DataFrame): DataFrame containing reaction data, including 'id', 'Reactant Chiral Count', and 'Product Chiral Count' columns.

    Returns:
    tuple: Three lists containing the counts of chiral-modulating, chiral-conserving, and achiral-conserving reactions for each step.
    """
    chiral_modulating = []
    chiral_conserving = []
    achiral_conserving = []

    for _, row in step_reaction_df.iterrows():
        if 'id' in rxn_df.columns:
            rxn_df = rxn_df.set_index('id')
        sub_rxn_df = rxn_df.loc[rxn_df.index.intersection(row["reactions"])]
        chiral_modulating.append(
            len(sub_rxn_df[sub_rxn_df["Reactant Chiral Count"] !=
                           sub_rxn_df["Product Chiral Count"]]))
        chiral_conserving.append(
            len(sub_rxn_df[sub_rxn_df["Reactant Chiral Count"] ==
                           sub_rxn_df["Product Chiral Count"]]))

        achiral_conserving_df = sub_rxn_df[sub_rxn_df["Reactant Chiral Count"]
                                           == 0]
        achiral_conserving.append(
            len(achiral_conserving_df[
                    achiral_conserving_df["Reactant Chiral Count"] ==
                    achiral_conserving_df["Product Chiral Count"]]))

    return chiral_modulating, chiral_conserving, achiral_conserving


def prepare_session():
    """
    Set up KEGG RestAPI session (from lets-get-kegg).

    Returns:
    requests.Session: A configured session with retry logic for KEGG RestAPI.
    """
    # Make the session
    s = Session()
    # Add retries
    retries = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[502, 503, 504],
        allowed_methods={'POST'},
    )
    # Mount the session
    s.mount('https://', HTTPAdapter(max_retries=retries))
    return s


def get_pathway_data(session, request_sleep=0.4):
    """
    Retrieve KEGG pathway information, including pathway-compound links and pathway names.

    Parameters:
    session (requests.Session): A configured session for making requests to the KEGG RestAPI.
    request_sleep (float, optional): Time to sleep between requests to avoid overloading the server. Defaults to 0.4 seconds.

    Returns:
    tuple: A tuple containing two dictionaries:
        - pathway_cpd_link (dict): A dictionary where keys are pathway IDs and values are lists of compound IDs.
        - pathway_names (dict): A dictionary where keys are pathway IDs and values are pathway names.
    """
    ### Get compounds belonging to each pathway
    print("- Getting KEGG pathway-compound links -")
    response = session.get(f"https://rest.kegg.jp/link/pathway/compound",
                           timeout=10.0)
    time.sleep(request_sleep)
    pathway_cpd_link = dict()
    if response.ok:
        res = response.text.split("\n")
        for line in res:
            if line.startswith("cpd"):
                cpd_path = line.split(
                    "\t")  # ([0] = "cpd:CXXXXX" ; [1] = "path:mapXXXXX")
                pathway_id = int(cpd_path[1].split(":")[1].replace("map", ""))
                if pathway_id not in pathway_cpd_link:
                    pathway_cpd_link[pathway_id] = [cpd_path[0].split(":")[1]]
                else:
                    pathway_cpd_link[pathway_id].append(
                        cpd_path[0].split(":")[1])
    else:
        # Some error in the response
        print(
            f"Error in /link/pathway/compound, response {response.status_code}",
            flush=True)
        return False

    ### Get pathway names
    print("- Getting KEGG pathway names -")
    response = session.get(f"https://rest.kegg.jp/list/pathway", timeout=10.0)
    pathway_names = dict()
    if response.ok:
        res = response.text.split("\n")
        for line in res:
            if line.startswith("map03020"):
                break  # Stop after Metabolism pathways end ()
            if line.startswith("map"):
                pathway_name = line.split(
                    "\t")  # ([0] = "mapXXXXX" ; [1] = "<NAME>")
                pathway_id = int(pathway_name[0].replace("map", ""))
                if pathway_id not in pathway_names:
                    pathway_names[pathway_id] = pathway_name[1]
    else:
        # Some error in the response
        print(f"Error in /list/pathway, response {response.status_code}",
              flush=True)
        return False

    return pathway_cpd_link, pathway_names


def get_chiral_percentage_from_list(cpds, kegg_c_df):
    """
    Given a list of KEGG compound IDs, find the chiral percentage.

    Parameters:
    cpds (list): A list of KEGG compound IDs.
    keggC_df (pandas.DataFrame): DataFrame containing KEGG compound data, including 'compound_id' and 'n_chiral_centers' columns.

    Returns:
    tuple: A tuple containing:
        - float: The percentage of chiral compounds.
        - int: The total number of compounds in the subset.
    """
    sub_df = kegg_c_df[kegg_c_df["compound_id"].isin(cpds)]
    if len(sub_df):
        return np.count_nonzero(
            sub_df["n_chiral_centers"]) / len(sub_df), len(sub_df)
    else:
        return 0, 0


def get_pathway_colors(pathway_df):
    """
    Assign colors to metabolic pathways and categorize them into groups.

    Parameters:
    pathway_df (pandas.DataFrame): DataFrame containing pathway information, including 'ID' and 'Chirality' columns.

    Returns:
    tuple: A tuple containing:
        - pandas.DataFrame: Updated DataFrame with pathway groups, keys, and colors.
        - list: A 2D list of chiralities for each metabolic pathway group.
        - dict: A dictionary mapping pathway IDs to their group colors.
        - dict: A dictionary mapping pathway group names to their pathway IDs.
        - dict: A dictionary mapping pathway group names to their colors.
        - list: A list of colors for the pathway groups.
    """
    # Hardcoding metabolic pathway subgroup names & colors
    metabolism_groups = {
        "Carbohydrate Metabolism": [
            10, 20, 30, 40, 51, 52, 53, 500, 520, 620, 630, 640, 650, 660, 562
        ],
        "Glycan Biosynthesis": [
            510, 513, 512, 515, 514, 532, 534, 533, 531, 563, 601, 603, 604,
            540, 542, 541, 550, 511, 571, 572
        ],
        "Terpenoids and Polyketides": [
            900, 902, 909, 904, 906, 905, 981, 908, 903, 281, 1052, 522, 1051,
            1059, 1056, 1057, 253, 523, 1054, 1053, 1055
        ],
        "Lipid Metabolism": [
            61, 62, 71, 73, 100, 120, 121, 140, 561, 564, 565, 600, 590, 591,
            592, 1040
        ],
        "Secondary Metabolite Biosynthesis": [
            940, 945, 941, 944, 942, 943, 901, 403, 950, 960, 1058, 232, 965,
            966, 402, 311, 332, 261, 331, 521, 524, 525, 401, 404, 405, 333,
            254, 999, 998, 997
        ],
        "Nucleotide Metabolism": [230, 240],
        "Amino Acid Metabolism": [
            250, 260, 270, 280, 290, 300, 310, 220, 330, 340, 350, 360, 380, 400
        ],
        "Metabolism of Other Amino Acids": [410, 430, 440, 450, 460, 470, 480],
        "Cofactors and Vitamins": [
            730, 740, 750, 760, 770, 780, 785, 790, 670, 830, 860, 130
        ],
        "Energy Metabolism": [190, 195, 196, 710, 720, 680, 910, 920],
        "Xenobiotics": [
            362, 627, 364, 625, 361, 623, 622, 633, 642, 643, 791, 930, 363,
            621, 626, 624, 365, 984, 980, 982, 983
        ]
    }

    colors = {
        "Carbohydrate Metabolism": "#aa0f35",
        "Glycan Biosynthesis": "#487c7c",
        "Terpenoids and Polyketides": "#009ecb",
        "Lipid Metabolism": "#abdbfc",
        "Secondary Metabolite Biosynthesis": "#430057",
        "Nucleotide Metabolism": "#a0a75c",
        "Amino Acid Metabolism": "#603734",
        "Metabolism of Other Amino Acids": "#d97e43",
        "Cofactors and Vitamins": "#313a61",
        "Energy Metabolism": "#fdc799",
        "Xenobiotics": "#f5c500"
    }

    pathway_group_keys = {
        "Carbohydrate Metabolism": 1,
        "Glycan Biosynthesis": 2,
        "Terpenoids and Polyketides": 3,
        "Lipid Metabolism": 4,
        "Secondary Metabolite Biosynthesis": 5,
        "Nucleotide Metabolism": 6,
        "Amino Acid Metabolism": 7,
        "Metabolism of Other Amino Acids": 8,
        "Cofactors and Vitamins": 9,
        "Energy Metabolism": 10,
        "Xenobiotics": 11
    }

    color_list = [
        "#aa0f35", "#487c7c", "#009ecb", "#abdbfc", "#430057", "#a0a75c",
        "#603734", "#d97e43", "#fdc799", "#313a61", "#f5c500"
    ]

    # Assign colors to all metabolic pathways
    pathway_groups = {}
    pathway_group_colors = {}
    pathway_groups_list = []
    pathway_key_list = []
    pathway_colors = []

    for _, row in pathway_df.iterrows():
        found_group = 0
        for group, pathways in metabolism_groups.items():
            if int(row["ID"]) in pathways:
                pathway_groups[int(row["ID"])] = group
                pathway_group_colors[int(row["ID"])] = colors[group]
                pathway_groups_list.append(group)
                pathway_key_list.append(pathway_group_keys[group])
                pathway_colors.append(row["ID"])
                found_group = 1
        if not found_group:
            pathway_groups_list.append(np.nan)
            pathway_key_list.append(np.nan)
            pathway_colors.append(np.nan)

    pathway_df["Group"] = pathway_groups_list
    pathway_df["Group Key"] = pathway_key_list
    pathway_df["Color"] = pathway_colors

    pathway_df = pathway_df.dropna()

    # Setup for box & whisker plot - needs a 2d array of chiralities
    chiralities = []
    for group in colors.keys():
        chiralities.append(pathway_df[pathway_df['ID'].isin(
            metabolism_groups[group])]["Chirality"].tolist())

    return pathway_df, chiralities, pathway_group_colors, metabolism_groups, colors, color_list


def set_box_color(bp, color):
    """
    Set the color of various elements in a box plot.

    Parameters:
    bp (dict): A dictionary containing the box plot elements.
    color (str): The color to set for the box plot elements.

    Returns:
    None
    """
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def get_chiral_fraction_at_heavy_atom_counts(df):
    """
    Calculate chiral fractions at increasing heavy atom counts.

    Parameters:
    df (pandas.DataFrame): DataFrame containing compound data, including 'n_heavy_atoms' and 'n_chiral_centers' columns.

    Returns:
    tuple: A tuple containing:
        - list: Chiral fractions at each heavy atom count.
        - list: Achiral fractions at each heavy atom count.
    """
    chiral_fraction = []
    for i in range(1, int(max(df["n_heavy_atoms"]))):
        sub_df = df[df["n_heavy_atoms"] == i]
        if len(sub_df) >= 10:  # Only do this for heavy atom counts >= 10
            chiral_fraction.append(
                sub_df["n_chiral_centers"].astype(bool).sum(axis=0) /
                float(len(sub_df)))
        else:
            chiral_fraction.append(float("NaN"))

    achiral_fraction = []
    for cf in chiral_fraction:
        if not math.isnan(cf):
            achiral_fraction.append(1 - cf)
        else:
            achiral_fraction.append(float("NaN"))

    return chiral_fraction, achiral_fraction


def get_chiral_order_parameters(kegg_c_df):
    """
    Get chiral order parameters (cop) for GDB and KEGG.

    Parameters:
    keggC_df (pandas.DataFrame): DataFrame containing KEGG compound data, including 'n_heavy_atoms' and 'n_chiral_centers' columns.

    Returns:
    tuple: A tuple containing:
        - list: Chiral order parameters for GDB.
        - list: Chiral order parameters for KEGG.
    """
    gdb_counts_df = pd.read_csv("../Data/GDB_raw_counts.csv")
    gdb_cop = []
    for _, row in gdb_counts_df.iterrows():
        gdb_cop.append(row["count_chiral"] /
                       (row["count_achiral"] + row["count_chiral"]))

    print(
        f"- GDB Compound Totals:\n\tAchiral: {gdb_counts_df['count_achiral'].sum()}\n\tChiral: {gdb_counts_df['count_chiral'].sum()}")

    kegg_cop = []
    for i in range(17):
        sub_df = kegg_c_df[kegg_c_df["n_heavy_atoms"] == i]
        achiral = len(sub_df[sub_df["n_chiral_centers"] == 0])
        chiral = len(sub_df[sub_df["n_chiral_centers"] != 0])
        kegg_cop.append(chiral / (achiral + chiral))

    print(f"- CBRdb Compound Size = {len(kegg_c_df)} -")

    return gdb_cop, kegg_cop


def gna(graph, attr):
    """
    Get node attributes from a graph.

    Parameters:
    graph (networkx.Graph): The graph from which to get the node attributes.
    attr (str): The name of the attribute to retrieve.

    Returns:
    dict: A dictionary where the keys are node IDs and the values are the attribute values.
    """
    return nx.get_node_attributes(graph, attr)


def colorize_series(S, cmap='cet_CET_C6s'):
    # S should be list-like, or a dict, or a series.... of numbers.
    color_map = matplotlib.colormaps[cmap]

    def float_norm(x):
        return (Normalize(vmin=x.min(), vmax=x.max()))

    def boundary_norm(x):
        return (BoundaryNorm(boundaries=list(range(x.min(), x.max() + 1, 1)), ncolors=color_map.N))

    if type(S) != pd.Series:
        S = pd.Series(S)
    if S.dtype == pd.Float64Dtype():
        c_ser = pd.Series(map(tuple, color_map(float_norm(S)(list(S.values)))), index=S.index)
    elif S.dtype == pd.Int64Dtype():
        c_ser = pd.Series(map(tuple, color_map(boundary_norm(S)(list(S.values)))), index=S.index)
    else:
        S = pd.Series(S.factorize()[0]) / (pd.Series(S.factorize()[0]).max())
        c_ser = pd.Series(map(tuple, color_map(float_norm(S)(list(S.values)))), index=S.index)
        c_ser = c_ser.apply(to_hex)
    return c_ser


def result_to_df(results: dict, key: str, to_apply):
    """ for each result in results dict, accesses the pd.Series or pd.DataFrame referenced by key and applies the function to_apply to it."""
    if key in results.keys():
        return(results[key].apply(to_apply).to_frame())
    elif key in results[list(results.keys())[0]].keys():
        return(pd.concat({i: results[i][key].apply(to_apply) for i in results.keys()}, axis=1))
    else:
        raise ValueError(f"Key {key} not found in results or in the first result's keys.")
