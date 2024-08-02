import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors
from rdkit.Chem import AllChem as Chem


def load_data(file_path):
    # load the data
    data = pd.read_csv(file_path)
    # return the data
    return data


def kegg_select():
    # load the ML-data
    data = load_data(os.path.join(os.getcwd(), "Data", "kegg_data_C.csv"))
    # print the columns headers
    print(data.columns)
    # select only the source ID and the InChI
    data = data[["Source_ID", "InChI"]]
    print(data.columns)
    # write the data to a new file
    data.to_csv(os.path.join(os.getcwd(), "Data", "kegg_data_C_inchi.csv"), index=False)


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


file = os.path.join(os.getcwd(), "Data", "kegg_data_C_inchi.csv")
data = load_data(file)

# select only the first 100 molecules
# data = data.head(100)

n = len(data["Source_ID"])
print("There are {} molecules in the dataset".format(n))

n_chiral = np.zeros(len(data), dtype=int)
index = np.arange(n)
for i, mol in data.iterrows():
    # print the index after every 10 molecules
    if i % 100 == 0:
        print("Processing molecule {}/{}".format(i, n), flush=True)
    inchi = mol["InChI"]
    # print(inchi)
    # Get the molecule
    mol = rdkit.Chem.MolFromInchi(inchi)
    mol = embed_mol(mol)
    # Get the chirality
    info_stero = rdkit.Chem.FindMolChiralCenters(mol, useLegacyImplementation=False, includeUnassigned=True)
    n_chiral[i] = len(info_stero)

out_data = pd.DataFrame(data={"Index": index,
                              "Cpd": data["Source_ID"],
                              "InChI": data["InChI"],
                              "Chiral_Centers": n_chiral})
# write the data to a new file
out_data.to_csv(os.path.join(os.getcwd(), "Data", "chiral_labels_new.csv"), index=False)
