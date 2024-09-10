import os
import re
import networkx as nx
import pandas as pd

try:
    tests = os.listdir('../Data/')
    data_folder = '../Data/'
except:
    data_folder = 'Data/'
if not os.path.exists('Output/'):
    os.mkdir('Output/')

get_sides = lambda x: x[['left', 'right']]

# compound info
chiral_labels = ((pd.read_csv(data_folder + 'kegg_data_C.csv.zip', index_col=0)
                  .assign(is_chiral=lambda x: x.n_chiral_centers > 0)
                  .rename(columns={'n_chiral_centers': 'Chiral_Centers'})
                  .set_index('compound_id')))
chiral_labels['elements'] = chiral_labels['formula'].apply(lambda x:  set([s.strip('+-1234567890') for s in re.split("([A-Z][^A-Z]*)", x) if s]))

# kegg_reactions: derived from reaction entries in July 2024 KEGG pull in Big Data folder
# https://github.com/ELIFE-ASU/lets-get-kegg/blob/6e4ae42309970acca24e420eaf9ff7ad1f9a1f80/Data/atlas_data_kegg_R_processed.csv.zip
kegg_reactions = pd.read_csv(data_folder + 'kegg_data_R.csv.zip', index_col=0)
kegg_reactions[['left','right']] = kegg_reactions['reaction'].str.split('<=>', expand=True)
for i in ['left','right']:
    kegg_reactions[i] = kegg_reactions[i].str.split(' ').apply(lambda x: set([i for i in x if i.startswith('C')]))
kegg_reactions['has_chiral_info'] = (get_sides(kegg_reactions).applymap(list).sum(axis=1).apply(set)
                                      - set(chiral_labels.index)).apply(len)==0
kegg_reactions = kegg_reactions.query('has_chiral_info==True & reaction.str.contains("G")==False')

# reactions flagged as questionable in BRENDA: https://bkms.brenda-enzymes.org/download.php
brenda_flagged = (pd.read_csv(data_folder + 'Reactions_BKMS.tsv', sep='\t', low_memory=False)
                  .dropna(subset='Reaction_ID_KEGG').dropna(subset=['Missing_Substrate', 'Missing_Product'], how='all')
                  .drop_duplicates(subset=['Reaction_ID_KEGG', 'Reaction', 'Missing_Product', 'Missing_Substrate'],
                                   keep='first')
                  .set_index('Reaction_ID_KEGG'))
# reactions flagged in Table S3 of Goldford et al. 2024, as "elementally inconsistent"
goldford_flagged_entries = open(data_folder + 'Goldford2024_TableS3_rnums.txt', 'r').read().split(' ')
# add these flags to reaction info
kegg_reactions['NO_BRENDA_FLAG'] = kegg_reactions.index.isin(brenda_flagged.index) == False
kegg_reactions['NO_GOLDFORD_FLAG'] = kegg_reactions.index.isin(goldford_flagged_entries) == False
# appendable reactions: carbonate-bicarbonate equilibrium and water dissociation.
# negligible effects on expansion (a few cps arrive earlier)
appendable_reactions = pd.DataFrame({'R99999': {'left': {'C00001'}, 'right': {'C01328', 'C00080'}},
                                     'R99998': {'left': {'C01353'}, 'right': {'C00288', 'C00080'}}}).T

# default starting seed set of compounds
primordial_seed_set = ["C00001", "C00080", "C00009", "C00014", "C00059", "C00011"]
all_poss_cps = list(get_sides(kegg_reactions).stack().explode().unique())

# turns a boolean DataFrame into a series of lists; entry shows which rows are true for a given column, i.e.,
# which reactions are enabled.
def get_rn_sets(rndf: pd.DataFrame):
    return rndf[rndf].apply(lambda x: set(x.dropna().index))

# removes reactions that are flagged in BRENDA as having missing substrates or products
def remove_questionable_brenda(x):
    return x.drop(x.index.intersection(brenda_flagged.index))


def restrict_elemental_scope(seeds, reactions:pd.DataFrame):
    # makes sure that reactions only involve elements present in the seed set
    cps_w_all_els = chiral_labels['elements'].explode()
    cps_w_els_notinseed = cps_w_all_els[~cps_w_all_els.isin(cps_w_all_els.loc[seeds])]
    ok_cps = chiral_labels.index.difference(cps_w_els_notinseed.index)

    rns_w_all_cps = get_sides(reactions).stack().reset_index(level=1, drop=True).explode()
    rns_w_els_notinseed = rns_w_all_cps[~rns_w_all_cps.isin(ok_cps)].index
    ok_rns = reactions.copy(deep=True).drop(rns_w_els_notinseed)
    return(ok_rns)

def expand_from_seed_set(seeds, reactions: pd.DataFrame, max_gen: int):
    # seeds : list-like, the seed set of compounds from which to start the expansion
    # reactions : pool of reactions to draw from; if all compounds on one side of a reaction are present, the reaction is enabled
    # max_gen : max # of expansion generations calculated, e.g. if you only care about early stages.
    rns, cps = dict(), {0: dict(zip(seeds, len(seeds) * [True]))}
    compound_pool = set(seeds)
    reaction_pool = reactions
    cp_sets = dict({1: compound_pool})
    fwd_rns, bwd_rns = set(), set()
    for g in range(1, max_gen + 1):
        has_R = reaction_pool['right'].apply(lambda x: x <= compound_pool)  # are all right-side compounds present?
        has_L = reaction_pool['left'].apply(lambda x: x <= compound_pool)  # are all left-side compounds present?
        rns[g] = (has_R + has_L) > 0  # if so, reaction is enabled
        fwd_rns = fwd_rns.union(set(reaction_pool[has_L > has_R].index))  # new compounds will be from right side
        bwd_rns = bwd_rns.union(set(reaction_pool[has_L < has_R].index))  # new compounds will be from left side
        compound_pool = set.union(compound_pool, *reaction_pool.loc[rns[g]].stack())  # add new-compound side to the pool
        cps[g] = dict(zip(compound_pool, len(compound_pool) * [True]))
        if cps[g] == cps[g - 1]:  # if no new compounds were added,
            break  # stop expanding
    directions = pd.concat([pd.Series(index=fwd_rns, data='fwd'), pd.Series(index=bwd_rns, data='bwd')])
    rns = pd.DataFrame(rns).replace(False,float('nan')).dropna(how='all').fillna(False).T.cummax().T
    cps = pd.DataFrame(cps).dropna(how='all').fillna(False).T.cummax().T  # store outputs
    cp_sets = pd.Series({i: set(j)-{''} for i, j in cps.mul(cps.index, axis=0).fillna('').to_dict(orient='list').items()})
    try:
        rn_sets = get_rn_sets(rns)
    except:
        rn_sets = pd.Series(index=cp_sets.index, data=float('nan'))
    chiral_fraction = cp_sets.apply(lambda x: chiral_labels.loc[list(x), 'is_chiral'].mean())
    # generate info for edge lists
    steps = rns.T.idxmax()  # note first appearance of each reaction
    edge_pool = reaction_pool.loc[rns.index].copy(deep=True).assign(
        step=rns.index.map(steps))  # these reactions were enabld
    edge_pool['directed'] = edge_pool.index.isin(
        bwd_rns.union(fwd_rns))  # these actively expanded (not infilled) the network
    edge_pool['backward'] = edge_pool.index.isin(bwd_rns)  # these did so in the backward direction
    a = (get_sides(edge_pool.query('backward==False or directed==False').copy(deep=True)).rename(
        columns={'left': 'source', 'right': 'target'}))  # forward + incidental reactions go L-R
    b = (edge_pool.query('backward==True').copy(deep=True)
         [['right', 'left']].rename(columns={'right': 'source', 'left': 'target'}))  # backward reactions go R-L
    edge_pool[['source', 'target']] = pd.concat([a, b], axis=0)
    edge_pool.drop(columns=['left', 'right'], inplace=True)
    edge_pool.reset_index(drop=False, names='reaction', inplace=True)
    participants = pd.DataFrame({'reactions': rn_sets, 'compounds': cp_sets})
    return (
        {'rns': rns, 'cps': cps, 'participants': participants, 'reaction_pool': list(reaction_pool.keys()), 'seeds': seeds,
         'chiral_fraction': chiral_fraction, 'edge_pool': edge_pool})


def make_unipartite_edgelist(edge_pool):
    return edge_pool.explode('source').explode('target').reset_index(drop=True)


# used for viz + analysis, not expansion itself. turns dataframes with opposite reaction sides
# into unipartite edgelists, for from_pandas_edgelist
def generate_edge_df(reactions: pd.DataFrame):
    return reactions.explode('left').explode('right').reset_index(names='reaction').query('left!=right')


def define_reachable_components(seeds, reactions: pd.DataFrame):
    bipartite_edges = reactions.applymap(list).stack().groupby(level=0).sum().explode().reset_index()
    bipartite_graph = nx.from_pandas_edgelist(bipartite_edges, source='id', target=0, create_using=nx.Graph)
    accessible_nodes = set.union(*[nx.node_connected_component(bipartite_graph, i) for i in seeds])
    del bipartite_graph
    del bipartite_edges
    return ({'reachable_compounds': [i for i in accessible_nodes if i.startswith('C')],
             'reachable_reactions': [i for i in accessible_nodes if i.startswith('R')]})
