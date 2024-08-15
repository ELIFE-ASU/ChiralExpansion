import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors
from rdkit import RDLogger
from rdkit.Chem import AllChem as Chem

plt.rcParams['axes.linewidth'] = 2.0

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def n_plot(xlab, ylab, xs=14, ys=14):
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def embed_mol(mol):
    # Sanitise the molecule
    Chem.SanitizeMol(mol)
    # Update the properties
    mol.UpdatePropertyCache()
    # kekulize the molecule
    Chem.Kekulize(mol)
    # Add hydrogens
    mol = rdkit.Chem.AddHs(mol)
    # Embed the molecule
    Chem.EmbedMolecule(mol)
    return mol


def analyze_data(fname):
    # read the data
    all_data = pd.read_csv(fname, index_col=0)
    all_data["nA"] = all_data["nA"].astype(int) + 1
    # Determine the chirality and split the data
    all_data["chiral"] = all_data["nStero"].map(lambda x: x > 0)
    chiral_data = all_data[all_data["chiral"] is True]
    achiral_data = all_data[all_data["chiral"] is False]
    # Reorganize to get the information we need
    chiral_counts = chiral_data.groupby("nA").count().unstack()["SMILES"]
    achiral_counts = achiral_data.groupby("nA").count().unstack()["SMILES"]
    # Make new df out of these series
    achiral_df = pd.DataFrame(achiral_counts, columns=["count"])
    chiral_df = pd.DataFrame(chiral_counts, columns=["count"])
    # Recombine the data to get the averages as a function of nA
    merged_df = pd.merge(achiral_df, chiral_df, on='nA', how='outer', suffixes=('_achiral', '_chiral'))
    merged_df.fillna(0, inplace=True)
    merged_df.reset_index(inplace=True)
    raw_df = merged_df.copy()
    merged_df['total'] = merged_df['count_achiral'] + merged_df['count_chiral']
    merged_df['fraction_achiral'] = merged_df['count_achiral'] / merged_df['total']
    merged_df['fraction_chiral'] = merged_df['count_chiral'] / merged_df['total']
    final_df = merged_df[['nA', 'fraction_achiral', 'fraction_chiral']]
    return final_df, raw_df


def plot_chiral(in_data, xs=14, ys=14):
    xlab = "Number of heavy atoms"
    ylab = "Fraction of Chiral Molecules"
    ave_data, raw_data = analyze_data(in_data)
    ave_data.to_csv("output.csv")
    raw_data.to_csv("raw_counts.csv")
    plt.plot(ave_data['nA'], ave_data['fraction_chiral'], ls='-', color='red', label='Chiral', lw=2, marker='o')
    plt.plot(ave_data['nA'], ave_data['fraction_achiral'], ls='-', color='blue', label='Achiral', lw=2, marker='o')
    # Make the axis ticks integers
    plt.xticks(ave_data['nA'])
    plt.legend(loc='best', fontsize=ys)
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    plt.savefig("Chirality_plot.pdf")
    plt.savefig("Chirality_plot.png", dpi=900)
    plt.show()
    plt.close()
    return None


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


def load_data(i, in_data):
    data = np.empty(0, dtype=str)
    for ii, file in enumerate([os.path.basename(f).split(".")[0] for f in in_data]):
        if file == str(i) and os.stat(in_data[ii]).st_size > 0:
            data = np.genfromtxt(os.path.abspath(in_data[ii]), delimiter="\t", dtype=str, usecols=0, comments=None)
    return data


def count_atoms(smiles):
    return len(re.findall(r'(?<!\[)C|N|O|F|P|S|Cl|Br|I|B|(?<=\[)([^\]]+)(?=\])', smiles))


def write_data_by_mass(in_data, output_data):
    # make the output directory
    if not os.path.exists(output_data):
        os.makedirs(output_data)

    # Load the data line by line
    with open(in_data, "r") as f:
        all_smiles = f.readlines()

    # Open the output files
    f1 = open(os.path.join(output_data, "1.smi"), "w")
    f2 = open(os.path.join(output_data, "2.smi"), "w")
    f3 = open(os.path.join(output_data, "3.smi"), "w")
    f4 = open(os.path.join(output_data, "4.smi"), "w")
    f5 = open(os.path.join(output_data, "5.smi"), "w")
    f6 = open(os.path.join(output_data, "6.smi"), "w")
    f7 = open(os.path.join(output_data, "7.smi"), "w")
    f8 = open(os.path.join(output_data, "8.smi"), "w")
    f9 = open(os.path.join(output_data, "9.smi"), "w")
    f10 = open(os.path.join(output_data, "10.smi"), "w")
    f11 = open(os.path.join(output_data, "11.smi"), "w")
    f12 = open(os.path.join(output_data, "12.smi"), "w")
    f13 = open(os.path.join(output_data, "13.smi"), "w")
    f14 = open(os.path.join(output_data, "14.smi"), "w")
    f15 = open(os.path.join(output_data, "15.smi"), "w")
    f16 = open(os.path.join(output_data, "16.smi"), "w")
    f17 = open(os.path.join(output_data, "17.smi"), "w")
    f18 = open(os.path.join(output_data, "18.smi"), "w")
    f19 = open(os.path.join(output_data, "19.smi"), "w")
    f20 = open(os.path.join(output_data, "20.smi"), "w")

    n_total = len(all_smiles)
    print("Total number of molecules: ", n_total, flush=True)
    # Loop over the molecules
    for i, s in enumerate(all_smiles):
        nA = count_atoms(s)
        if i % 1000 == 0:
            print(f"Processed 1k molecules", flush=True)

        # Write the data to the appropriate file
        if nA == 1:
            f1.write(s)
        elif nA == 2:
            f2.write(s)
        elif nA == 3:
            f3.write(s)
        elif nA == 4:
            f4.write(s)
        elif nA == 5:
            f5.write(s)
        elif nA == 6:
            f6.write(s)
        elif nA == 7:
            f7.write(s)
        elif nA == 8:
            f8.write(s)
        elif nA == 9:
            f9.write(s)
        elif nA == 10:
            f10.write(s)
        elif nA == 11:
            f11.write(s)
        elif nA == 12:
            f12.write(s)
        elif nA == 13:
            f13.write(s)
        elif nA == 14:
            f14.write(s)
        elif nA == 15:
            f15.write(s)
        elif nA == 16:
            f16.write(s)
        elif nA == 17:
            f17.write(s)
        elif nA == 18:
            f18.write(s)
        elif nA == 19:
            f19.write(s)
        elif nA == 20:
            f20.write(s)
        else:
            print(f"Error: {nA} atoms", flush=True)
            exit()

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    f7.close()
    f8.close()
    f9.close()
    f10.close()
    f11.close()
    f12.close()
    f13.close()
    f14.close()
    f15.close()
    f16.close()
    f17.close()
    f18.close()
    f19.close()
    f20.close()
    return None


def merge_mass_data(in_data_1, in_data_2, in_data_3, out_data, f_debug=True):
    # Get all the files
    in_data_1 = file_list_all(in_data_1)
    in_data_2 = file_list_all(in_data_2)
    in_data_3 = file_list_all(in_data_3)

    # Get the expected unique files
    f_names_1 = [os.path.basename(f).split(".")[0] for f in in_data_1]
    f_names_2 = [os.path.basename(f).split(".")[0] for f in in_data_2]
    f_names_3 = [os.path.basename(f).split(".")[0] for f in in_data_3]
    idx = np.sort(np.array(np.unique(np.hstack([f_names_1, f_names_2, f_names_3])), dtype=int))

    # Setup the output directory, check if the directory exists
    if not os.path.exists(out_data):
        os.makedirs(out_data)

    # Loop over the atom masses
    for i in idx:
        if f_debug:
            if i > 6:
                break

        print(f"Processing files with N={i} heavy atoms", flush=True)
        df1 = load_data(i, in_data_1)
        df2 = load_data(i, in_data_2)
        df3 = load_data(i, in_data_3)

        # Combine the data
        df = np.hstack([df1, df2, df3])
        # If df1 and df2 are not empty, remove the duplicates
        if df1.size != 0 and df2.size != 0:
            # Remove the duplicates
            print(f"Removing duplicates for N={i}", flush=True)
            df = np.unique(df)

        # Write the data to a file
        np.savetxt(os.path.join(out_data, f"{i}.smi"), df, fmt="%s", delimiter="\n", comments=None)


def pick_n_random(data, n_pick=1000):
    return np.random.choice(data, min(n_pick, len(data)), replace=False)


def get_chirality(mol):
    return len(rdkit.Chem.FindMolChiralCenters(mol,
                                               useLegacyImplementation=False,
                                               includeUnassigned=True,
                                               includeCIP=False))


def get_heavy_atoms(mol):
    return mol.GetNumHeavyAtoms()


def load_by_mass(in_data, out_data, n_pick=1000):
    # Get all the files
    in_data = file_list_all(in_data)
    if len(in_data) == 0:
        print("No files found", flush=True)
        exit()
    # sort the files by number of heavy atoms
    in_data = sorted(in_data, key=lambda x: int(os.path.basename(x).split(".")[0]))
    print(f"Total number of files to process: {len(in_data)}", flush=True)
    # Open the output file
    with open(out_data, "w") as file:
        file.write(",SMILES,nA,nStero\n")

        # Loop over the files
        for i, f in enumerate(in_data):
            print(f"Processing file {i}: {f}", flush=True)
            # Load the data line by line
            with open(f, "r") as ff:
                all_smiles = ff.readlines()

            if n_pick is not None:
                print(f"Picking {n_pick} random molecules", flush=True)
                all_smiles = pick_n_random(all_smiles, n_pick=n_pick)

            # Loop over the molecules
            for ii, s in enumerate(all_smiles):
                mol = rdkit.Chem.MolFromSmiles(s)
                mol = embed_mol(mol)
                nA = get_heavy_atoms(mol)
                nStereo = get_chirality(mol)
                file.write(f"{ii},{s.strip()},{nA},{nStereo}\n")
    return None


if __name__ == "__main__":
    print("Program started", flush=True)
    if os.name == "nt":
        data_dir = r"C:\Users\louie\skunkworks\data"
    else:
        data_dir = r"/scratch/lslocomb/GDB/"
    # List the three directories of the GDB data
    in_data_11 = os.path.join(data_dir, "GDB11")
    in_data_13 = os.path.join(data_dir, "GDB13")
    in_data_17 = os.path.join(data_dir, "GDB17")
    # Directory of the merged data
    merged_data_dir = os.path.join(data_dir, "GDB_merged")
    merged_data_file = os.path.join(data_dir, "GDB_merged_chirality.csv")

    # First make all the directories of the data by hand
    # Sort the GDB17 data by number of heavy atoms
    write_data_by_mass(os.path.join(data_dir, r"GDB17.smi"), in_data_17)

    # Merge the three datasets
    merge_mass_data(in_data_11, in_data_13, in_data_17, merged_data_dir, f_debug=False)

    # Load the data by mass and pick random molecules and save the chirality data
    load_by_mass(merged_data_dir, merged_data_file, n_pick=5000)

    # Plot the chirality data
    plot_chiral(merged_data_file)
