import os
import pandas as pd
import numpy as np
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors
from rdkit.Chem import AllChem as Chem
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=0)
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


only_used = False


def embed_mol(mol):
    Chem.SanitizeMol(mol)   # Sanitise the molecule
    mol.UpdatePropertyCache()  # Update the properties
    Chem.Kekulize(mol)     # kekulize the molecule
    mol = rdkit.Chem.AddHs(mol)     # Add hydrogens
    Chem.EmbedMolecule(mol)     # Embed the molecule
    return mol

def get_chirality(mol_obj_series):
    chirality_dict = {}
    for cpd, mol in mol_obj_series.items():
        try:
            chirality_dict[cpd] = len(rdkit.Chem.FindMolChiralCenters(mol, useLegacyImplementation=False, includeUnassigned=True, includeCIP=False))
        except: 
            chirality_dict[cpd] = float('NaN')
    return(chirality_dict)

def divvy_into_cores(data, ncores):
    rows_per_core = np.floor(len(data)/ncores)
    data2 = data.sample(frac=1)
    data2['core'] = [int(np.ceil(float(i)/rows_per_core)) for i in range(1,len(data2.index)+1)]
    return(data2.sort_values(by=['core','bottcher_index']))


# Load the data
print('Importing compounds. Shuffling for a more even size distribution between cores during multiprocessing.')
data = pd.read_csv(os.path.join(os.getcwd(), "Data", "kegg_data_C.csv.zip"), index_col=0)
if only_used:
    print('For this analysis... we only care about compounds which participate in reactions.')
    reactions = pd.read_csv(os.path.join(os.getcwd(), "Data", "kegg_data_R_basic.csv.zip"), index_col=0)
    used_cpds = list(reactions.loc[:,['LEFT_IDS','RIGHT_IDS']].stack().str.split(' ').explode().unique())
    print('A fraction of compounds does not, per kegg_r:', end='  ')
    print(len(data.index.difference(used_cpds))/len(data.index))
    print('Dropping those from calculation.')
    data.drop(data.index.difference(used_cpds), inplace=True)
data = divvy_into_cores(data, 8)
print('Making mol files from relevant compounds InChI strings. Should take <20 seconds.')
mols = data['InChI'].apply(rdkit.Chem.MolFromInchi)
print('Embedding molecules. Should take a few minutes to run, assuming 8 cores available. \nProbably the longest step. Last 10% takes the most time.')
mols_embedded = mols.parallel_apply(embed_mol)
print(' Calculating the number of chiral centers. Should take <1 min.')
data['Chiral_Centers'] = pd.Series(get_chirality(mols_embedded))
print('Checking for compounds for which chirality was not calculated:')
print(data.query('Chiral_Centers.isna()').index)
if 'C05798' in data.index:
    print('We know C05798 is chiral:  https://www.chiralen.com/product/cas/17453-58-6.html')
    print('It joins with a chain in one reaction, R04795, to form a molecule C02139, with 3 chiral centers.')
    print(data.loc['C02139'])
    print('For now... assuming C05798 also has 3 chiral centers. We can formally check later.')
    data.loc['C05798','Chiral_Centers'] = 3
data['Chiral_Centers'] = data['Chiral_Centers'].astype(int)
print('Saving the data to a file.')
data.sort_index().to_csv(os.path.join(os.getcwd(), "Data", "kegg_data_C_inKeggR_and_keggdataC_all.csv.zip"), compression='zip', encoding='utf-8')

