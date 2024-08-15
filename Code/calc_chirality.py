import os

import numpy as np
import pandas as pd
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors
from pandarallel import pandarallel
from rdkit import RDLogger

import utilities as ut

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def divvy_into_cores(data, ncores=None):
    if ncores is None:
        ncores = get_n_cores()
    rows_per_core = np.floor(len(data) / ncores)
    data2 = data.sample(frac=1)
    data2['core'] = [int(np.ceil(float(i) / rows_per_core)) for i in range(1, len(data2.index) + 1)]
    return data2.sort_values(by=['core', 'smiles'])


def get_n_cores():
    return len(os.sched_getaffinity(0))


if __name__ == '__main__':
    data_dir = "../Data"
    f_parallel = False
    if f_parallel:
        pandarallel.initialize(progress_bar=True, verbose=0)
        data_file = r"kegg_data_C.csv.zip"
        data = divvy_into_cores(pd.read_csv(os.path.join(data_dir, data_file), index_col=0))
        data['n_chiral_centers'] = ut.get_chirality(
            data['smiles'].apply(rdkit.Chem.MolFromSmiles).parallel_apply(ut.standardize_mol))
        data.sort_index().to_csv(os.path.join(data_dir, "kegg_data_C_detailed.csv.zip"),
                                 compression='zip',
                                 encoding='utf-8')
    else:
        data_file = r"kegg_data_C.csv.zip"
        data = pd.read_csv(os.path.join(data_dir, data_file), index_col=0)
        data['n_chiral_centers'] = ut.get_chirality(
            data['smiles'].apply(rdkit.Chem.MolFromSmiles).apply(ut.standardize_mol))
        data.sort_index().to_csv(os.path.join(data_dir, "kegg_data_C_detailed.csv.zip"),
                                 compression='zip',
                                 encoding='utf-8')
