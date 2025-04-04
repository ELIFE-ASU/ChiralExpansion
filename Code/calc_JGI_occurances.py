import pandas as pd
import os 

compound_fname = "https://github.com/ELIFE-ASU/CBRdb/blob/6ea6338cf00cd167d93e2557b5569156826e5cfa/CBRdb_C.csv?raw=true"
reaction_fname = "https://github.com/ELIFE-ASU/CBRdb/blob/6ea6338cf00cd167d93e2557b5569156826e5cfa/CBRdb_R.csv?raw=true"
domain_csv_dir = os.path.abspath('../../../../ASU Dropbox/Elife/Big Data/JGI-Data/October-2023-Pull/Unprocessed-JGI-Oct2023-CSVsOfECCounts/')
output_dir = os.path.abspath('../Data/')

keggC_df = pd.read_csv(compound_fname, index_col=0)
keggC_df['is_chiral'] = keggC_df['n_chiral_centers'].gt(0).astype(int)

listlike_cols = ['ec', 'module', 'orthology', 'pathway', 'rclass', 'rhea', 'id_orig']
keggR_df = pd.read_csv(reaction_fname, converters=dict(zip(listlike_cols, len(listlike_cols) * [lambda x: x.split()])), low_memory=False)

keggR_df['compounds'] = keggR_df['reaction'].str.findall(r'(C\d{5})')
keggR_df = keggR_df.join(keggR_df['reaction'].str.split(' <=> ', expand=True).apply(lambda x: x.str.findall(r'(C\d{5})')).rename(columns={0: 'left', 1: 'right'}))
keggR_df['id_orig_sorted'] = keggR_df['id_orig'].map(lambda x: ' '.join(sorted(x, reverse=True)))
keggR_df = keggR_df.set_index('id_orig_sorted') 
# to show only kegg-associated reactions: keggR_df.filter(like='R', axis=0)

ec2cp = keggR_df.explode('ec').dropna(subset='ec').query('~ec.str.contains("-")').groupby('ec')['compounds'].apply(lambda x: x.explode().unique())
ec2rn = keggR_df.explode('ec').dropna(subset='ec').query('~ec.str.contains("-")').reset_index().groupby('ec')['id'].apply(list).rename('reactions')

domain_dfs = dict()
compounds_present = dict()
compound_chirality = dict()

# Measure chemical properties for each domain. Takes ~50s.
for i in os.listdir(domain_csv_dir):
    domain_name = i.split('_')[1]
    # Change reference so JM code still works
    if domain_name == 'Microbiome':
        domain_name = 'Metagenome'
    # Read in domain EC count CSV
    domain_dfs[domain_name] = pd.read_csv(os.path.join(domain_csv_dir, i), index_col=0, na_filter=False)
    # Remove EC number prefix for integration with CBRdb_R formatting
    domain_dfs[domain_name].index = domain_dfs[domain_name].index.str.strip('EC:')
    
# Change references so JM code still works
domain_names = {"Eukaryota": "eukarya", 'Archaea': 'archaea', 'Bacteria': 'bacteria', 'Metagenome': 'metagenome'}

# Calculate occurrances for compounds
for dn, dn_save in domain_names.items():
    present = domain_dfs[dn].join(ec2cp, how='inner').explode('compounds').groupby('compounds').sum().gt(0)
    # Identify chirality of present compounds
    chirality = (present.astype(int)[present].T * keggC_df.loc[present.index, 'is_chiral']).T
    # Calculate occurrance rates
    occ_out = present.sum(axis=1).sort_values(ascending=False).rename('Occurance').rename_axis('Cpd').to_frame()
    # Store results
    compounds_present[dn_save] = present
    compound_chirality[dn_save] = chirality
    occ_out.to_csv(os.path.join(output_dir, dn_save+'_occurance.csv'), encoding='utf-8')

# Calculate occurrances for reactions
for dn, dn_save in domain_names.items():
    present = domain_dfs[dn].join(ec2rn, how='inner').explode('reactions').groupby('reactions').sum().gt(0)
    occ_out = present.sum(axis=1).sort_values(ascending=False).rename('Occurance').rename_axis('Rxn').to_frame()
    occ_out.to_csv(os.path.join(output_dir, dn+'_occurance_rxn.csv'), encoding='utf-8')

