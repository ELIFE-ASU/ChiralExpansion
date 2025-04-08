import os

import pandas as pd

compound_fname = os.path.abspath("../Data/CBRdb_C.csv")
reaction_fname = os.path.abspath("../Data/CBRdb_R.csv")
domain_csv_dir = os.path.abspath(
    '../../../../ASU Dropbox/Elife/Big Data/JGI-Data/October-2023-Pull/Unprocessed-JGI-Oct2023-CSVsOfECCounts/')
omics_dir = os.path.abspath('../Data/omics/')

keggC_df = pd.read_csv(compound_fname, index_col=0)
keggC_df['is_chiral'] = keggC_df['n_chiral_centers'].gt(0).astype(int)

listlike_cols = ['ec', 'id_orig']
keggR_df = pd.read_csv(reaction_fname, low_memory=False,
                       converters=dict(zip(['ec', 'id_orig'], 2*[lambda x: x.split()])))

# for easier calculation of occurrance rates
keggR_df['compounds'] = keggR_df['reaction'].str.findall(r'(C\d{5})')
# for easier filtering-out of ATLAS reactions if requested
keggR_df['id_orig_sorted'] = keggR_df['id_orig'].map(lambda x: ' '.join(sorted(x, reverse=True)))
keggR_df = keggR_df.set_index('id_orig_sorted') 
# to show only kegg-associated reactions: keggR_df.filter(like='R', axis=0)

ec2cp = keggR_df.explode('ec').dropna(subset='ec').query('~ec.str.contains("-")').groupby('ec')['compounds'].apply(
    lambda x: x.explode().unique())
ec2rn = keggR_df.explode('ec').dropna(subset='ec').query('~ec.str.contains("-")').reset_index().groupby('ec')[
    'id'].apply(list).rename('reactions')

domain_dfs = dict()
# Read in CSVs for each domain.
for i in os.listdir(domain_csv_dir):
    domain_name = i.split('_')[1]
    # Change reference so JM code still works
    if domain_name == 'Microbiome':
        domain_name = 'Metagenome'
    # Read in domain EC count CSV
    ecs = pd.read_csv(os.path.join(domain_csv_dir, i), index_col=0, na_filter=False)
    # Remove EC number prefix for integration with CBRdb_R formatting
    ecs.index = ecs.index.str.strip('EC:')
    # Remove partial EC numbers and transpose
    ecs = ecs.loc[~ecs.index.str.contains('-'), :].T.assign(domain=domain_name)
    # Store results for each domain
    domain_dfs[domain_name] = ecs

# Combine all domain dataframes into one
ec_counts = pd.concat(domain_dfs.values(), axis=0).fillna(0)
# Change counts to present/absent
ecs_present = ec_counts.set_index('domain', append=True).mask(lambda x: x>0).fillna(1).astype(int)
# Sort EC serial numbers
ecs_present_sorted = pd.DataFrame(dict(zip(ecs_present.columns, 
                                           ecs_present.columns.map(lambda x: dict(pd.Series([int(i) for i in x.split('.')])))))
                                           ).T.sort_values(by=[0,1,2,3]).index
ecs_present = ecs_present.loc[:, ecs_present_sorted]

# Output domains for each taxon ID.
tid2domain = ecs_present.reset_index(level=1)['domain']
tid2domain.index = tid2domain.index.astype(int)
tid2domain.to_csv(os.path.join(omics_dir, 'tid2domain.csv.zip'), encoding='utf-8', compression='zip')

# Output ECs present for each taxon ID.
ecs_present = ecs_present.reset_index('domain', drop=True).rename_axis(None)
ecs_present.index = ecs_present.index.astype(int)
ecs_present.to_csv(os.path.join(omics_dir, 'tid2ecs.csv.zip'), encoding='utf-8', compression='zip')

# Calculate compound and reaction occurrence for each taxon ID.
ecs = ecs_present.T.astype(bool).assign(compounds=ec2cp, reactions=ec2rn).dropna(subset='compounds')
cps = ecs.drop(columns='reactions').explode('compounds').groupby('compounds').any().T.astype(int)
rns = ecs.drop(columns='compounds').explode('reactions').groupby('reactions').any().T.astype(int)

# Calculate occurrences for reactions and compounds, for each domain.
occ = {'compound': cps, 'reaction': rns}
for k in occ.keys():
    a = occ[k].groupby(tid2domain).sum().T
    a.loc['C'] = tid2domain.value_counts()
    a.sort_index(inplace=True)
    a.rename({'C': 'N'}, axis=0, inplace=True)
    a.rename_axis(None, axis=1, inplace=True)
    occ[k] = a
    # Output occurrence summary for each domain. N is the number of taxon IDs in that domain.
    occ[k].to_csv(os.path.join(omics_dir, f'{k}2occurrence.csv'), encoding='utf-8')
    
