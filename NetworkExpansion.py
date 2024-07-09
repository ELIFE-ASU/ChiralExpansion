import pandas as pd

# kegg_2018_0 - derived from reaction_edges file in chirality MS folder - reaction count suggests c.2018
kegg_2018_0 = pd.read_csv('Data/reaction_edges_2018.csv.zip', compression='zip', index_col=0).applymap(eval).applymap(set)
# kegg_2024_0 - derived from reaction entries in Jan 2024 KEGG pull in Big Data folder
kegg_2024_0 = pd.read_csv('Data/reaction_edges_2024_01.csv.zip', compression='zip', index_col=0).applymap(eval).applymap(set)
# default starting seed set
primordial_seed_set = ["C00001","C00080","C00009","C00014","C00059","C00011"]
# from BRENDA: https://bkms.brenda-enzymes.org/download.php
brenda_flagged = (pd.read_csv('Data/Reactions_BKMS.tsv', sep='\t')
          .dropna(subset='Reaction_ID_KEGG').dropna(subset=['Missing_Substrate','Missing_Product'], how='all')
          .drop_duplicates(subset=['Reaction_ID_KEGG','Reaction','Missing_Product','Missing_Substrate'], keep='first')
          .set_index('Reaction_ID_KEGG'))
# we still need chirality for some compounds; in the meantime, treat those as NaN
all_poss_cps = list(set.union(*kegg_2018_0.stack()).union(*kegg_2024_0.stack()))
chiral_labels = (pd.read_csv('Data/chiral_labels.csv', index_col=0, usecols=[1,3])
                 .assign(is_chiral = lambda x: x.Chiral_Centers>0)).join(pd.DataFrame(index=all_poss_cps), how='outer')

# turns a boolean DataFrame into a series of lists; entry shows which rows are true for a given column, i.e., which reactions are enabled.
def get_rn_sets(rndf: pd.DataFrame):
        return(rndf[rndf].apply(lambda x: set(x.dropna().index)))

def expand_from_seed_set(seeds, reactions: pd.DataFrame, max_gen: int):
    # seeds : list-like, the seed set of compounds from which to start the expansion
    # reactions : pool of reactions to draw from; if all compounds on one side of a reaction are present, the reaction is enabled
    # max_gen : max # of expansion generations calculated, e.g. if you only care about early stages.
    rns, cps = dict(), {0: dict(zip(seeds, len(seeds)*[True]))} 
    compound_pool = set(seeds)
    cp_sets = dict({1:compound_pool}) 
    for g in range(max_gen+2):
        has_R = reactions['right'].apply(lambda x: x<=compound_pool) # are all right-side compounds present?
        has_L = reactions['left'].apply(lambda x: x<=compound_pool) # are all left-side compounds present?
        rns[g+1] = (has_R+has_L)>0 # if so, reaction is enabled
        compound_pool = set.union(compound_pool, *reactions.loc[rns[g+1]].stack()) # and the other side of that reaction is added to the pool
        cps[g+1] = dict(zip(compound_pool, len(compound_pool)*[True]))
        cp_sets[g+2] = set(list(compound_pool)) #so that indexing matches w/prior expansion output.
        if (cp_sets[g+2] == cp_sets[g+1]): # if no more reactions were enabled by the addition of reactions last time...
            del cp_sets[g+2] # no need to store duplicate generations
            del cps[g+1]
            break
    rns, cps = pd.DataFrame(rns), pd.DataFrame(cps).dropna(how='all').fillna(False).T.cummax().T #store outputs
    rns = rns[rns].dropna(how='all').fillna(False)
    cp_sets = pd.Series(cp_sets)
    rn_sets = get_rn_sets(rns)
    chiral_fraction = cp_sets.apply(lambda x: chiral_labels.loc[list(x), 'is_chiral'].mean())
    return({'rns': rns, 'cps': cps, 'cp_sets': cp_sets, 'rn_sets': rn_sets, 'reaction_pool': reactions, 'seeds': seeds, 'chiral_fraction': chiral_fraction})


# used for viz + analysis, not expansion. turns dataframes like kegg_2018_0 into something that can be fed to networkx via from_pandas_edgelist
def generate_edge_df(reactions: pd.DataFrame):
    return(reactions.explode('left').explode('right').reset_index(names='reaction').query('left!=right'))
