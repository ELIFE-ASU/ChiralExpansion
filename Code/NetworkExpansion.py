import pandas as pd

"""REACTION SCOPE"""
# kegg_reactions: derived from reaction entries in July 2024 KEGG pull in Big Data folder
kegg_reactions = pd.read_csv('../Data/kegg_data_R_basic.csv.zip', index_col=0).query('NO_GLYCANS')
kegg_reactions.update(kegg_reactions.filter(like='_IDS').applymap(lambda x: set(x.split(' ')), na_action='ignore'))
kegg_reactions = kegg_reactions.rename(columns={'LEFT_IDS': 'left', 'RIGHT_IDS': 'right'})
# appendable reactions: carbonate-bicarbonate equilibrium and water dissociation. negligible effects on expansion (a few cps arrive earlier)
appendable_reactions = pd.DataFrame({'R99999': {'left': {'C00001'}, 'right': {'C01328', 'C00080'}},
                                     'R99998': {'left': {'C01353'}, 'right': {'C00288', 'C00080'}}}).T
# reactions flagged as questionable in BRENDA: https://bkms.brenda-enzymes.org/download.php
brenda_flagged = (pd.read_csv('Data/Reactions_BKMS.tsv', sep='\t')
                  .dropna(subset='Reaction_ID_KEGG').dropna(subset=['Missing_Substrate', 'Missing_Product'], how='all')
                  .drop_duplicates(subset=['Reaction_ID_KEGG', 'Reaction', 'Missing_Product', 'Missing_Substrate'],
                                   keep='first')
                  .set_index('Reaction_ID_KEGG'))
# reactions flagged in Table S3 of Goldford et al 2024, as "elementally inconsistent"
goldford_flagged_entries = open('../Data/Goldford2024_TableS3_rnums.txt', 'r').read().split(' ')
# add these flags to reaction info
kegg_reactions['NO_BRENDA_FLAG'] = kegg_reactions.index.isin(brenda_flagged.index) == False
kegg_reactions['NO_GOLDFORD_FLAG'] = kegg_reactions.index.isin(goldford_flagged_entries) == False
get_sides = lambda x: x[['left', 'right']]

"""COMPOUND SCOPES"""
# default starting seed set of compounds
primordial_seed_set = ["C00001", "C00080", "C00009", "C00014", "C00059", "C00011"]
all_poss_cps = list(kegg_reactions[['left', 'right']].stack().explode().unique())
# we still need chirality for some compounds; in the meantime, treat those as NaN
chiral_labels = (pd.read_csv('../Data/kegg_data_C_detailed.csv.zip', index_col=0)
                 .assign(is_chiral=lambda x: x.Chiral_Centers > 0)).join(pd.DataFrame(index=all_poss_cps), how='outer')

"""FUNCTIONS"""


# turns a boolean DataFrame into a series of lists; entry shows which rows are true for a given column, i.e., which reactions are enabled.
def get_rn_sets(rndf: pd.DataFrame):
    return (rndf[rndf].apply(lambda x: set(x.dropna().index)))


# removes reactions that are flagged in BRENDA as having missing substrates or products
def remove_questionable_brenda(x):
    return (x.drop(x.index.intersection(brenda_flagged.index)))


def expand_from_seed_set(seeds, reactions: pd.DataFrame, max_gen: int):
    # seeds : list-like, the seed set of compounds from which to start the expansion
    # reactions : pool of reactions to draw from; if all compounds on one side of a reaction are present, the reaction is enabled
    # max_gen : max # of expansion generations calculated, e.g. if you only care about early stages.
    rns, cps = dict(), {0: dict(zip(seeds, len(seeds) * [True]))}
    compound_pool = set(seeds)
    cp_sets = dict({1: compound_pool})
    fwd_rns, bwd_rns = set(), set()
    for g in range(max_gen + 2):
        has_R = reactions['right'].apply(lambda x: x <= compound_pool)  # are all right-side compounds present?
        has_L = reactions['left'].apply(lambda x: x <= compound_pool)  # are all left-side compounds present?
        rns[g + 1] = (has_R + has_L) > 0  # if so, reaction is enabled
        fwd_rns = fwd_rns.union(set(reactions[has_L > has_R].index))  # new compounds will be from right side
        bwd_rns = fwd_rns.union(set(reactions[has_L < has_R].index))  # new compounds will be from left side
        compound_pool = set.union(compound_pool, *reactions.loc[
            rns[g + 1]].stack())  # and the other side of that reaction is added to the pool
        cps[g + 1] = dict(zip(compound_pool, len(compound_pool) * [True]))
        cp_sets[g + 2] = set(list(compound_pool))  # so that indexing matches w/prior expansion output.
        if (cp_sets[g + 2] == cp_sets[
            g + 1]):  # if no more reactions were enabled by the addition of reactions last time...
            del cp_sets[g + 2]  # no need to store duplicate generations
            del cps[g + 1]
            break
    rns, cps = pd.DataFrame(rns), pd.DataFrame(cps).dropna(how='all').fillna(False).T.cummax().T  # store outputs
    rns = rns[rns].dropna(how='all').fillna(False)
    directions = pd.concat([pd.Series(index=fwd_rns, data='fwd'), pd.Series(index=bwd_rns, data='bwd')])
    cp_sets = pd.Series(cp_sets)
    rn_sets = get_rn_sets(rns)
    chiral_fraction = cp_sets.apply(lambda x: chiral_labels.loc[list(x), 'is_chiral'].mean())
    return ({'rns': rns, 'cps': cps, 'cp_sets': cp_sets, 'rn_sets': rn_sets, 'reaction_pool': reactions, 'seeds': seeds,
             'chiral_fraction': chiral_fraction, 'directions': directions})


# used for viz + analysis, not expansion. turns dataframes like kegg_2018_0 into something that can be fed to networkx via from_pandas_edgelist
def generate_edge_df(reactions: pd.DataFrame):
    return (reactions.explode('left').explode('right').reset_index(names='reaction').query('left!=right'))
