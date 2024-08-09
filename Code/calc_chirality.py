import os

import numpy as np
import pandas as pd
import psutil
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors
from pandarallel import pandarallel
from rdkit import RDLogger
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def embed_mol(mol):
    # Sanitise the molecule
    Chem.SanitizeMol(mol, catchErrors=True)
    # Update the properties
    mol.UpdatePropertyCache()
    # kekulize the molecule
    Chem.Kekulize(mol)
    # Add hydrogens
    mol = rdkit.Chem.AddHs(mol)
    # Embed the molecule
    # Chem.EmbedMolecule(mol)
    return mol


def standardize_mol(mol):
    # Standardize the molecule
    nrm = rdMolStandardize.Normalizer()
    nrm.normalizeInPlace(mol)
    # Embed the molecule
    mol = embed_mol(mol)
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


def divvy_into_cores(data, ncores=None):
    if ncores is None:
        ncores = get_n_cores()
    rows_per_core = np.floor(len(data) / ncores)
    data2 = data.sample(frac=1)
    data2['core'] = [int(np.ceil(float(i) / rows_per_core)) for i in range(1, len(data2.index) + 1)]
    return (data2.sort_values(by=['core', 'smiles']))


def get_n_cores():
    return len(os.sched_getaffinity(0))


if __name__ == '__main__':
    data_dir = "../Data"
    f_parallel = False
    if f_parallel:
        pandarallel.initialize(progress_bar=True, verbose=0)
        data_file = r"kegg_data_C.csv.zip" # was: r"C:\Users\louie\skunkworks\lets-get-kegg\kegg_data_C.csv.zip"
        data = divvy_into_cores(pd.read_csv(os.path.join(data_dir, data_file), index_col=0))
        data['n_chiral_centers'] = get_chirality(
            data['smiles'].apply(rdkit.Chem.MolFromSmiles).parallel_apply(standardize_mol))
        data.sort_index().to_csv(os.path.join(data_dir, "kegg_data_C_detailed.csv.zip"),
                                 compression='zip',
                                 encoding='utf-8')
    else:
        data_file = r"kegg_data_C.csv.zip" # was: r"C:\Users\louie\skunkworks\lets-get-kegg\kegg_data_C.csv.zip"
        data = pd.read_csv(os.path.join(data_dir, data_file), index_col=0)
        data['n_chiral_centers'] = get_chirality(
            data['smiles'].apply(rdkit.Chem.MolFromSmiles).apply(standardize_mol))
        data.sort_index().to_csv(os.path.join(data_dir, "kegg_data_C_detailed.csv.zip"),
                                 compression='zip',
                                 encoding='utf-8')
