import os

import networkx as nx
import pandas as pd

try:
    tests = os.listdir('../Data/')
    data_folder = '../Data/'
except:
    data_folder = 'Data/'
if not os.path.exists('Output/'):
    os.mkdir('Output/')

# kegg_reactions: derived from reaction entries in July 2024 KEGG pull in Big Data folder
kegg_reactions = pd.read_csv(data_folder + 'kegg_data_R_basic.csv.zip', index_col=0).query('NO_GLYCANS')
kegg_reactions.update(kegg_reactions.filter(like='_IDS').applymap(lambda x: set(x.split(' ')), na_action='ignore'))
kegg_reactions = kegg_reactions.rename(columns={'LEFT_IDS': 'left', 'RIGHT_IDS': 'right'})
# appendable reactions: carbonate-bicarbonate equilibrium and water dissociation.
# negligible effects on expansion (a few cps arrive earlier)
appendable_reactions = pd.DataFrame({'R99999': {'left': {'C00001'}, 'right': {'C01328', 'C00080'}},
                                     'R99998': {'left': {'C01353'}, 'right': {'C00288', 'C00080'}}}).T
# reactions flagged as questionable in BRENDA: https://bkms.brenda-enzymes.org/download.php
brenda_flagged = (pd.read_csv(data_folder + 'Reactions_BKMS.tsv', sep='\t')
                  .dropna(subset='Reaction_ID_KEGG').dropna(subset=['Missing_Substrate', 'Missing_Product'], how='all')
                  .drop_duplicates(subset=['Reaction_ID_KEGG', 'Reaction', 'Missing_Product', 'Missing_Substrate'],
                                   keep='first')
                  .set_index('Reaction_ID_KEGG'))
# reactions flagged in Table S3 of Goldford et al. 2024, as "elementally inconsistent"
goldford_flagged_entries = open(data_folder + 'Goldford2024_TableS3_rnums.txt', 'r').read().split(' ')
# add these flags to reaction info
kegg_reactions['NO_BRENDA_FLAG'] = kegg_reactions.index.isin(brenda_flagged.index) == False
kegg_reactions['NO_GOLDFORD_FLAG'] = kegg_reactions.index.isin(goldford_flagged_entries) == False
get_sides = lambda x: x[['left', 'right']]

# default starting seed set of compounds
primordial_seed_set = ["C00001", "C00080", "C00009", "C00014", "C00059", "C00011"]
all_poss_cps = list(kegg_reactions[['left', 'right']].stack().explode().unique())
# we still need chirality for some compounds; in the meantime, treat those as NaN
chiral_labels = ((pd.read_csv(data_folder + 'kegg_data_C_detailed.csv.zip', index_col=0)
                  .assign(is_chiral=lambda x: x.n_chiral_centers > 0)
                  .rename(columns={'n_chiral_centers': 'Chiral_Centers'})
                  .set_index('CID'))
                 .join(pd.DataFrame(index=all_poss_cps), how='outer'))


# turns a boolean DataFrame into a series of lists; entry shows which rows are true for a given column, i.e.,
# which reactions are enabled.
def get_rn_sets(rndf: pd.DataFrame):
    return rndf[rndf].apply(lambda x: set(x.dropna().index))


# removes reactions that are flagged in BRENDA as having missing substrates or products
def remove_questionable_brenda(x):
    return x.drop(x.index.intersection(brenda_flagged.index))


def expand_from_seed_set(seeds, reactions: pd.DataFrame, max_gen: int):
    # seeds : list-like, the seed set of compounds from which to start the expansion
    # reactions : pool of reactions to draw from; if all compounds on one side of a reaction are present, the reaction is enabled
    # max_gen : max # of expansion generations calculated, e.g. if you only care about early stages.
    rns, cps = dict(), {0: dict(zip(seeds, len(seeds) * [True]))}
    compound_pool = set(seeds)
    cp_sets = dict({1: compound_pool})
    fwd_rns, bwd_rns = set(), set()
    for g in range(1, max_gen + 1):
        has_R = reactions['right'].apply(lambda x: x <= compound_pool)  # are all right-side compounds present?
        has_L = reactions['left'].apply(lambda x: x <= compound_pool)  # are all left-side compounds present?
        rns[g] = (has_R + has_L) > 0  # if so, reaction is enabled
        fwd_rns = fwd_rns.union(set(reactions[has_L > has_R].index))  # new compounds will be from right side
        bwd_rns = bwd_rns.union(set(reactions[has_L < has_R].index))  # new compounds will be from left side
        compound_pool = set.union(compound_pool, *reactions.loc[rns[g]].stack())  # add new-compound side to the pool
        cps[g] = dict(zip(compound_pool, len(compound_pool) * [True]))
        if cps[g] == cps[g - 1]:  # if no new compounds were added,
            break  # stop expanding
    rns, cps = pd.DataFrame(rns), pd.DataFrame(cps).dropna(how='all').fillna(False).T.cummax().T  # store outputs
    rns = rns[rns].dropna(how='all').fillna(False)
    directions = pd.concat([pd.Series(index=fwd_rns, data='fwd'), pd.Series(index=bwd_rns, data='bwd')])
    cp_sets, rn_sets = cps.mul(cps.index, axis=0).apply(set) - {''}, get_rn_sets(rns)
    chiral_fraction = cp_sets.apply(lambda x: chiral_labels.loc[list(x), 'is_chiral'].mean())
    # generate info for edge lists
    steps = rns.T.idxmax()  # note first appearance of each reaction
    edge_pool = reactions.loc[rns.index].copy(deep=True).assign(
        step=rns.index.map(steps))  # these reactions were enabld
    edge_pool['directed'] = edge_pool.index.isin(
        bwd_rns.union(fwd_rns))  # these actively expanded (not infilled) the network
    edge_pool['backward'] = edge_pool.index.isin(bwd_rns)  # these did so in the backward direction
    a = (edge_pool.query('backward==False or directed==False').copy(deep=True)
    [['left', 'right']].rename(
        columns={'left': 'source', 'right': 'target'}))  # forward + incidental reactions go L-R
    b = (edge_pool.query('backward==True').copy(deep=True)
         [['right', 'left']].rename(columns={'right': 'source', 'left': 'target'}))  # backward reactions go R-L
    edge_pool[['source', 'target']] = pd.concat([a, b], axis=0)
    edge_pool.drop(columns=['left', 'right'], inplace=True)
    edge_pool.reset_index(drop=False, names='reaction', inplace=True)
    participants = pd.DataFrame({'reactions': rn_sets, 'compounds': cp_sets})
    return (
        {'rns': rns, 'cps': cps, 'participants': participants, 'reaction_pool': list(reactions.keys()), 'seeds': seeds,
         'chiral_fraction': chiral_fraction, 'edge_pool': edge_pool})


def make_bipartite_edgelist(edge_pool):
    incoming = edge_pool.drop(columns='target').rename(columns={'reaction': 'target'}).explode('source')
    outgoing = edge_pool.drop(columns='source').rename(columns={'reaction': 'source'}).explode('target')
    edges = pd.concat([incoming, outgoing], axis=0).reset_index(drop=True).assign(
        reaction=lambda x: x[['source', 'target']].T.max())
    return edges


def make_unipartite_edgelist(edge_pool):
    return edge_pool.explode('source').explode('target').reset_index(drop=True)


# used for viz + analysis, not expansion itself. turns dataframes with opposite reaction sides
# into unipartite edgelists, for from_pandas_edgelist
def generate_edge_df(reactions: pd.DataFrame):
    return reactions.explode('left').explode('right').reset_index(names='reaction').query('left!=right')


def define_reachable_components(seeds, reactions: pd.DataFrame):
    bipartite_edges = reactions.applymap(list).stack().groupby(level=0).sum().explode().reset_index()
    bipartite_graph = nx.from_pandas_edgelist(bipartite_edges, source='index', target=0, create_using=nx.Graph)
    accessible_nodes = set.union(*[nx.node_connected_component(bipartite_graph, i) for i in seeds])
    del bipartite_graph
    del bipartite_edges
    return ({'reachable_compounds': [i for i in accessible_nodes if i.startswith('C')],
             'reachable_reactions': [i for i in accessible_nodes if i.startswith('R')]})
