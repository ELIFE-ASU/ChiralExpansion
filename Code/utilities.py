import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

tqdm.pandas()
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import time
import math
import json

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
        sub_df = kegg_df[kegg_df["compound_id"].isin(
            cpd_df.index[cpd_df[i] == True].tolist())]
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
        sub_df = kegg_df[kegg_df["compound_id"].isin(
            cpd_df.index[cpd_df[i] == True].tolist())]
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
        sub_df = kegg_df  #To ensure remainder of code works

    return sub_df["compound_id"].sample(size).values


def get_ChiralPercentage(cum_chiral, cum_achiral):
    """ Get percentage chiral at a specific location

    Args:
        cum_chiral (list): cumulative sum of chiral compounds over generations
        cum_achiral (list): cumulative sum of achiral compounds over generations
    """
    chiral_percent = []
    for i in range(len(cum_chiral)):
        chiral_percent.append(
            float(cum_chiral[i]) / (cum_chiral[i] + cum_achiral[i]))

    return chiral_percent


def get_luca_percentage(df):
    """
    Given a dataframe with sets of compounds per network expansion step, calculate the percentage of compounds 
    that show up in LUCA compounds
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
        percent_luca.append((float(luca_count) / len(luca_cpds)) *
                            100)  #Put into a percentage (0-100) scale

    return percent_luca


def linkArrivalStepHeavyAtoms(arrival_df, kegg_df):
    kegg_df = pd.merge(kegg_df,
                       arrival_df,
                       how="left",
                       left_on="compound_id",
                       right_index=True).dropna()
    kegg_df = kegg_df.rename(columns={"level_0": "arrival"
                                     })  #Cleaning up merge notation

    heavy_atom_counts = np.array(kegg_df["n_heavy_atoms"])

    heavy_atom_percentiles = []
    for ha in kegg_df["n_heavy_atoms"].to_list():
        try:  #Account for possible NaNs
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
    rxn_df = pd.read_csv(
        "../../lets-get-kegg/Data/atlas_kegg_processed_merged.csv.zip",
        index_col=0,
        compression="zip")
    rxn_df.drop_duplicates(
        subset="reaction", inplace=True
    )  #TODO: fully check for duplicates (or wait for lets-get-kegg to be fixed)

    rxn_df["reaction"].apply(degrees_PerRxn, args=(cpd_degrees,))

    return cpd_degrees


def scatter_hist(x, y, ax, ax_histy, c):
    """ 
    Scatter plot analysis for Figure 3C
    """
    achiral_color = "#FFC012"
    chiral_color = "#34CCFF"
    max_size = 100  #Maximum size of scatter points (for readability)

    ax_histy.tick_params(axis="y", labelleft=False)

    #Plot by chiral/achiral color, and size by degree
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

    #Set bins to middle of chiral center y-values to line up across axes
    bins = np.arange(-0.5, 11.5, 1)
    cnts, values, bars = ax_histy.hist(np.array(y) - 0.5,
                                       orientation='horizontal',
                                       color=c,
                                       bins=bins,
                                       edgecolor="k")

    #Make sure labels are smaller on histogram than scatter plot
    ax_histy.set_xlabel("Count", fontsize=14)
    ax_histy.tick_params(axis='both', which='major', labelsize=14)

    #Change color of histogram bars
    for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
        if i == 0:
            bar.set_facecolor(achiral_color)
        else:
            bar.set_facecolor(chiral_color)


def find_fit(X, Y, color):
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    # print('Slope: {:.6f}'.format(slope),'Intercept: {:.6f}'.format(intercept),'R-squared: {:.6f}'.format(r_value**2),'Std_err: {:.6f}'.format(std_err))
    x_dummy = np.linspace(start=0, stop=8000, num=100)
    plt.plot(x_dummy, slope * x_dummy + intercept, color=color)
    plt.fill_between(x_dummy,
                     slope * x_dummy + intercept - std_err,
                     slope * x_dummy + intercept + std_err,
                     color=color,
                     alpha=0.2)


def make_AUC_dashes(u, label, c, ls, ax):
    yvals = list(u.values())
    xvals = np.linspace(0, 1, num=len(yvals))

    #Calculate AUC using composite travezoidal rule
    area = np.trapz(yvals, dx=xvals[1])

    ax.plot(xvals, yvals, label=label, color=c, linestyle=ls, linewidth=4)

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
    for i in np.linspace(0, len(chiral_values) - 1, 100):
        m_chiral_percentiles.append(chiral_values[int(i)])

    m_achiral_percentiles = []
    for i in np.linspace(0, len(achiral_values) - 1, 100):
        m_achiral_percentiles.append(achiral_values[int(i)])

    #Step 2: find difference
    m_diffs = list(
        np.array(m_chiral_percentiles) - np.array(m_achiral_percentiles))

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

    return pd.Series([reactant_chiral_count, product_chiral_count],
                     index=["Reactant Chiral Count", "Product Chiral Count"])


def get_heatmap_array(keggC_df):
    keggC_df = keggC_df.set_index("compound_id")
    rxn_chiral_count = {}
    heatmap_array = []

    #atlas_kegg_processed_merged is the current source of truth (right, @Camerian & @Louie?)
    rxn_df = pd.read_csv(
        "../../lets-get-kegg/Data/atlas_kegg_processed_merged.csv.zip",
        index_col=0,
        compression="zip")

    print("- Calculating Reaction Chirality -")
    rxn_df[["Reactant Chiral Count", "Product Chiral Count"
           ]] = rxn_df["reaction"].progress_apply(check_rxn_chirality,
                                                  args=(keggC_df,
                                                        rxn_chiral_count))

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
    Set up KEGG RestAPI session (from lets-get-kegg)
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
    Heavily modified version of get_data from lets-get-kegg, pings kegg for pathway information
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
                    "\t")  #([0] = "cpd:CXXXXX" ; [1] = "path:mapXXXXX")
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

    print("- Getting KEGG pathway names -")
    response = session.get(f"https://rest.kegg.jp/list/pathway", timeout=10.0)
    pathway_names = dict()
    if response.ok:
        res = response.text.split("\n")
        for line in res:
            if line.startswith("map03020"):
                break  #Stop after Metabolism pathways end ()
            if line.startswith("map"):
                pathway_name = line.split(
                    "\t")  #([0] = "mapXXXXX" ; [1] = "<NAME>")
                pathway_id = int(pathway_name[0].replace("map", ""))
                if pathway_id not in pathway_names:
                    pathway_names[pathway_id] = pathway_name[1]
    else:
        # Some error in the response
        print(f"Error in /list/pathway, response {response.status_code}",
              flush=True)
        return False

    return pathway_cpd_link, pathway_names


def get_chiral_percentage_fromList(cpds, keggC_df):
    """
    Given a list of KEGG compound IDs, find the chiral percentage
    """
    sub_df = keggC_df[keggC_df["compound_id"].isin(cpds)]
    if len(sub_df):
        return np.count_nonzero(
            sub_df["n_chiral_centers"]) / len(sub_df), len(sub_df)
    else:
        return 0, 0


def get_pathway_colors(pathway_df):
    #Hardcoding metabolic pathway subgroup names & colors

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

    #Assign colors to all metabolic pathways
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

    #Setup for box & whisker plot - needs a 2d array of chiralities
    chiralities = []
    for group in colors.keys():
        chiralities.append(pathway_df[pathway_df['ID'].isin(
            metabolism_groups[group])]["Chirality"].tolist())

    return pathway_df, chiralities, pathway_group_colors, metabolism_groups, colors, color_list


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def get_chiral_fraction_atHeavyAtomCounts(df):
    """ 
    Chiral fractions at increasing heavy atom counts
    """
    chiral_fraction = []
    for i in range(1, int(max(df["n_heavy_atoms"]))):
        sub_df = df[df["n_heavy_atoms"] == i]
        if len(sub_df) >= 10:  #Only do this for heavy atom counts >= 10
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


def get_chiral_order_parameters(keggC_df):
    """
    Get chiral order parameters (cop) for GDB  & KEGG
    """
    gdb_counts_df = pd.read_csv("../Data/GDB_raw_counts.csv")
    gdb_cop = []
    for _, row in gdb_counts_df.iterrows():
        gdb_cop.append(row["count_chiral"] /
                       (row["count_achiral"] + row["count_chiral"]))
        
    print(f"- GDB Compound Totals:\n\tAchiral: {gdb_counts_df['count_achiral'].sum()}\n\tChiral: {gdb_counts_df['count_chiral'].sum()}")

    kegg_cop = []
    for i in range(17):
        sub_df = keggC_df[keggC_df["n_heavy_atoms"] == i]
        achiral = len(sub_df[sub_df["n_chiral_centers"] == 0])
        chiral = len(sub_df[sub_df["n_chiral_centers"] != 0])
        kegg_cop.append(chiral / (achiral + chiral))

    print(f"- KEGG Compound Size = {len(keggC_df)} -")

    return gdb_cop, kegg_cop
