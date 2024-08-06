import os

import numpy as np
import pandas as pd
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors
from pandarallel import pandarallel
from rdkit.Chem import AllChem as Chem

pandarallel.initialize(progress_bar=True, verbose=0)
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


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
    return (chirality_dict)


def divvy_into_cores(data, ncores):
    rows_per_core = np.floor(len(data) / ncores)
    data2 = data.sample(frac=1)
    data2['core'] = [int(np.ceil(float(i) / rows_per_core)) for i in range(1, len(data2.index) + 1)]
    return (data2.sort_values(by=['core', 'bottcher_index']))


if __name__ == '__main__':
    data_dir = "../Data"
    data = divvy_into_cores(pd.read_csv(os.path.join(data_dir, "kegg_data_C.csv.zip"), index_col=0), 8)
    data['n_chiral_centers'] = get_chirality(data['InChI'].apply(rdkit.Chem.MolFromInchi).parallel_apply(embed_mol))
    data.sort_index().to_csv(os.path.join(data_dir, "kegg_data_C_detailed.csv.zip"),
                             compression='zip',
                             encoding='utf-8')
