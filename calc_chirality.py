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



# load the ML-data
data = load_data(os.path.join(os.getcwd(), "Data", "kegg_data_C.csv"))
print(data.head())

# # get the chirality
# info_stero = rdkit.Chem.FindMolChiralCenters(mol, useLegacyImplementation=False, includeUnassigned=True)