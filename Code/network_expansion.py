import re

import networkx as nx
import pandas as pd

# find relative path to data source
data_folder = '../Data/'
get_sides = lambda x: x[['left', 'right']]


# imports a reaction dataframe, removes glycan reactions and reactions for which chiral info is missing
def import_reaction_file(fpath=data_folder + 'CBRdb_R.csv'):
    df = pd.read_csv(fpath, index_col=0).drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    if 'id' in df.columns:
        df.set_index('id', inplace=True)
    df[['left', 'right']] = df['reaction'].str.split(' <=> ', expand=True)
    for i in ['left', 'right']:
        df[i] = df[i].str.split(' ').apply(lambda x: set([j for j in x if j.startswith('C')]))
    df['has_chiral_info'] = (get_sides(df).applymap(list).sum(axis=1).apply(set)
                             - set(chiral_labels.index)).apply(len) == 0
    df = df.query('has_chiral_info & reaction.str.contains("G")==False')
    overall_reactions = ['R13204', 'R13205', 'R13206', 'R13207', 'R13208', 'R13209', 'R13210', 'R13211', 'R13222']
    df.drop(overall_reactions, inplace=True, errors='ignore')
    return df


def get_rn_sets(rndf: pd.DataFrame):
    """
    Converts a boolean DataFrame into a series of sets, where each set contains the indices of rows that are True for a given column.

    Parameters:
    rndf (pd.DataFrame): A boolean DataFrame where each cell indicates the presence (True) or absence (False) of a reaction.

    Returns:
    pd.Series: A series where each element is a set of indices (row labels) that are True for the corresponding column in the input DataFrame.
    """
    return rndf[rndf].apply(lambda x: set(x.dropna().index))


def expand_from_seed_set(seeds, reactions: pd.DataFrame, max_gen: int):
    # seeds : list-like, the seed set of compounds from which to start the expansion
    # reactions : pool of reactions to draw from; if all compounds on one side of a reaction are present, the reaction is enabled
    # max_gen : max # of expansion generations calculated, e.g. if you only care about early stages.
    rns, cps = dict(), {0: dict(zip(seeds, len(seeds) * [True]))}
    compound_pool, reaction_pool = set(seeds), reactions
    fwd_rns, bwd_rns = set(), set()
    # actual expansion code starts here
    for g in range(1, max_gen + 1):
        has_r = reaction_pool['right'].apply(lambda x: x <= compound_pool)  # are all right-side compounds present?
        has_l = reaction_pool['left'].apply(lambda x: x <= compound_pool)  # are all left-side compounds present?
        rns[g] = (has_r + has_l) > 0  # if so, reaction is enabled
        fwd_rns = fwd_rns.union(set(reaction_pool[has_l > has_r].index))  # new compounds will be from right side
        bwd_rns = bwd_rns.union(set(reaction_pool[has_l < has_r].index))  # new compounds will be from left side
        compound_pool = set.union(compound_pool,
                                  *reaction_pool.loc[rns[g]].stack())  # add new-compound side to the pool
        cps[g] = dict(zip(compound_pool, len(compound_pool) * [True]))
        if cps[g] == cps[g - 1]:  # if no new compounds were added,
            break  # stop expanding
    # end of actual expansion code, start of processing
    rns = pd.DataFrame(rns).replace(False, float('nan')).dropna(how='all').fillna(False).T.cummax().T
    cps = pd.DataFrame(cps).dropna(how='all').fillna(False).T.cummax().T  # store outputs
    cp_sets = pd.Series(
        {i: set(j) - {''} for i, j in cps.mul(cps.index, axis=0).fillna('').to_dict(orient='list').items()})
    try:
        rn_sets = get_rn_sets(rns)
    except:
        rn_sets = pd.Series(index=cp_sets.index, data=float('nan'))
    chiral_fraction = cp_sets.apply(lambda x: chiral_labels.loc[list(x), 'is_chiral'].mean())
    participants = pd.DataFrame({'reactions': rn_sets, 'compounds': cp_sets})
    try:
        participants.loc[0, 'reactions'] = set(seeds) - set(seeds)
    except:
        participants.loc[0, 'reactions'] = pd.NA
    arrival_step = participants.stack().explode().drop_duplicates().dropna().reset_index().set_index(0)['level_0']
    participants = participants.join(participants.diff().fillna(participants).add_prefix('new '))
    # generate info for edge lists
    steps = rns.T.idxmax()  # note first appearance of each reaction
    edge_pool = reaction_pool.loc[rns.index].copy(deep=True).assign(
        step=rns.index.map(steps))  # these reactions were enabled
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
    return (
        {'rns': rns, 'cps': cps, 'participants': participants, 'reaction_pool': list(reaction_pool.index),
         'seeds': seeds, 'arrival_step': arrival_step,
         'chiral_fraction': chiral_fraction, 'edge_pool': edge_pool})


# identifies what compounds and reactions in the database are reachable from at least one seed compound
def define_reachable_components(seeds, reactions: pd.DataFrame):
    bipartite_edges = reactions.applymap(list).stack().groupby(level=0).sum().explode().reset_index()
    bipartite_graph = nx.from_pandas_edgelist(bipartite_edges, source='id', target=0, create_using=nx.Graph)
    accessible_nodes = set.union(*[nx.node_connected_component(bipartite_graph, i) for i in seeds])
    del bipartite_graph
    del bipartite_edges
    return ({'reachable_compounds': [i for i in accessible_nodes if i.startswith('C')],
             'reachable_reactions': [i for i in accessible_nodes if i.startswith('R') or i.startswith('A')]})


def generate_annotated_bipartite_edge_list(single_result):
    """
    Generates an annotated bipartite edge list by first creating a preliminary edge list and then adding context on the essentially of each edge.

    Parameters:
    single_result (dict): A dictionary containing the results of the network expansion, including the edge pool.

    Returns:
    dict: The updated single_result dictionary with the annotated bipartite edge list.
    """
    # use make_bipartite_edgelist to generate prelim edgelist
    single_result['bipartite_edgelist'] = make_bipartite_edge_list(single_result['edge_pool'])
    # use add_bipartite_edge_context to add context on essentiality of each edge
    single_result = add_bipartite_edge_context(single_result)
    return single_result


# generates a bipartite edge list from expand_from_seed_set's edge_pool output
def make_bipartite_edge_list(edge_pool: pd.DataFrame):
    df = edge_pool.copy(deep=True)
    df['reaction'] = df['reaction'].str.replace('A', 'T')  # so that max identifies reaction, not compound
    sub_edges = df.explode('source').drop(columns='target').rename(columns={'reaction': 'target'})
    prod_edges = df.explode('target').drop(columns='source').rename(columns={'reaction': 'source'})
    all_edges = pd.concat([sub_edges, prod_edges], axis=0, ignore_index=True)
    all_edges['reaction'] = all_edges[['source', 'target']].max(axis=1)
    all_edges = all_edges.sort_values(by=['step', 'reaction', 'source', 'target']).reset_index(drop=True).loc[:,
                all_edges.nunique().sort_values().index]
    all_edges.update(
        all_edges[['source', 'target', 'reaction']].applymap(lambda x: x.replace('T', 'A')))  # revert back to original
    return all_edges


def get_single_result(x, single_result):
    """
    Retrieves the arrival step of a given compound or reaction from the single_result dictionary.

    Parameters:
    x (str): The compound or reaction identifier.
    single_result (dict): A dictionary containing the arrival steps of compounds and reactions.

    Returns:
    int: The arrival step of the given compound or reaction if it exists in single_result['arrival_step'].
    """
    if x in single_result['arrival_step']:
        return single_result['arrival_step'][x]


# adds context to each edge regarding its essentiality
def add_bipartite_edge_context(single_result):
    df = single_result['bipartite_edgelist']
    df = (df.join(
        df[['reaction', 'source', 'target']].applymap(lambda x: get_single_result(x, single_result)).add_suffix(
            '_step')))
    impactful_precursors = df.query('directed').query(
        '(target==reaction & reaction_step>source_step) | (reaction==source & reaction_step==target_step)').index
    immediate_precursors = df.loc[impactful_precursors].query('~(reaction_step-source_step!=1)').index
    single_result['bipartite_edgelist'] = df.assign(impactful_precursor=df.index.isin(impactful_precursors),
                                                    immediate_precursor=df.index.isin(immediate_precursors))
    return single_result


# VARIABLES AND DATASETS
chiral_labels = ((pd.read_csv(data_folder + 'CBRdb_C.csv')
                  .assign(is_chiral=lambda x: x.n_chiral_centers > 0)
                  .rename(columns={'n_chiral_centers': 'Chiral_Centers'})
                  .set_index('compound_id')))
chiral_labels['elements'] = chiral_labels['formula'].apply(
    lambda x: set([s.strip('+-1234567890') for s in re.split("([A-Z][^A-Z]*)", x) if s]))

# ec_rns: KEGG reaction-enzyme entry linkages.
ec_rns = pd.read_csv('https://rest.kegg.jp/link/enzyme/reaction', sep='\t', header=None).applymap(
    lambda x: x.split(':')[1]).groupby(0)[1].apply(list).apply('|'.join)
seed_candidates = pd.read_csv(f'{data_folder}/SampleSeeds.csv', index_col=0)
