import os

import numpy as np
import pandas as pd
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors
from rdkit.Chem import AllChem as Chem


def kegg_select():
    # load the ML-data
    data = pd.read_csv(os.path.join(os.getcwd(), "Data", "kegg_data_C.csv"))
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


# Load the data
data = pd.read_csv(os.path.join(os.getcwd(), "Data", "kegg_data_C_inchi.csv"))

# select only the first 100 molecules
# data = data.head(100)

# Get the number of molecules
n = len(data["Source_ID"])
print("There are {} molecules in the dataset".format(n))
# Create an array to store the number of chiral centers
n_chiral = np.zeros(len(data), dtype=int)
# Create an array to store the index
index = np.arange(n)
# Loop over the molecules
for i, mol in data.iterrows():
    # Print the index after every set molecules
    if i % 100 == 0:
        print("Processing molecule {}/{}".format(i, n), flush=True)
    inchi = mol["InChI"]
    # Get the molecule
    mol = rdkit.Chem.MolFromInchi(inchi)
    # Embed the molecule in 3D
    mol = embed_mol(mol)
    # Get the number of chiral centers
    n_chiral[i] = len(rdkit.Chem.FindMolChiralCenters(mol,
                                                      useLegacyImplementation=False,
                                                      includeUnassigned=True,
                                                      includeCIP=False))
# Create a new dataframe to store the data
out_data = pd.DataFrame(data={"Index": index,
                              "Cpd": data["Source_ID"],
                              "InChI": data["InChI"],
                              "Chiral_Centers": n_chiral})
# Write the data to a new file
out_data.to_csv(os.path.join(os.getcwd(), "Data", "chiral_labels_new.csv"), index=False)
