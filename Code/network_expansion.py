import re
import os
import networkx as nx
import pandas as pd

# find relative path to data source
data_folder = '../Data/'
reaction_file = os.path.join(data_folder, 'CBRdb_R.csv')
compound_file = os.path.join(data_folder, 'CBRdb_C.csv')
seed_file = os.path.join(data_folder, 'SampleSeeds.csv')
get_sides = lambda x: x[['left', 'right']]


def single_expansion(seed_ids, rxn_db_fp=reaction_file, cpd_db_fp=compound_file, max_gen=1000, edgelist=True):

    # STEP 1: load the databases underlying expansion process
    if isinstance(rxn_db_fp, str):
        # import reaction database if a file path is given
        rxns = import_reaction_file(rxn_db_fp)
    elif isinstance(rxn_db_fp, pd.DataFrame):
        # if pre-processed reaction DF is given, use that
        rxns = rxn_db_fp
        if 'id' in rxns.columns:
            rxns = rxns.set_index('id')

    if isinstance(cpd_db_fp, str):
        # import compound database if a file path is given
        cpds = pd.read_csv(cpd_db_fp, index_col=0)
        cpds['is_chiral'] = cpds['n_chiral_centers'].gt(0)
    elif isinstance(cpd_db_fp, pd.DataFrame):
        # if pre-processed compound DF is given, use that
        cpds = cpd_db_fp
        if 'compound_id' in cpds.columns:
            cpds = cpds.set_index('compound_id')
    
    # STEP 2: perform the expansion
    result = expand_from_seed_set(seeds=seed_ids, reactions=rxns, max_gen=max_gen)

    # STEP 3: generate edge lists if requested
    if edgelist == True:
        result = generate_annotated_bipartite_edge_list(result)
        result = generate_immediate_precursor_unipartite_edge_list(result)
    
    return result


def get_single_expansion_compound_summary(result, cpd_db):

    scope_chirality = cpd_db.loc[result['cps'].index, 'is_chiral'].copy()
    cps_chirality = result['cps'].mul(scope_chirality, axis=0)
    cps_heavyatom = result['cps'].astype(float).mul(result['cps'].index.map(cpd_db['n_heavy_atoms']), axis=0)
    chir_heavyatom = cps_heavyatom[cps_chirality.eq(1)]
    achir_heavyatom = cps_heavyatom[cps_chirality.eq(0)]
    
    result.update({'scope_chirality': scope_chirality,
                     'cps_chirality': cps_chirality,
                     'cps_heavyatom': cps_heavyatom})

    summary = pd.DataFrame(
        {'compound count': result['cps'].count(),
        'reaction count': result['rns'].sum(),
        'chiral compound count': cps_chirality.gt(0).sum(),
        'achiral compound count': cps_chirality.eq(0).sum(),
        'chiral fraction': cps_chirality.mean(),
        'chiral avg heavy atoms': chir_heavyatom.mean(),
        'achiral avg heavy atoms': achir_heavyatom.mean(),
        'chiral std heavy atoms': chir_heavyatom.std(),
        'achiral std heavy atoms': achir_heavyatom.std()}).fillna(0)
    
    new_counts = summary.filter(like='count').diff().fillna(summary).add_prefix('new ')
    summary = summary.join(new_counts).convert_dtypes()

    return summary


def import_reaction_file(fpath=reaction_file):
    df = pd.read_csv(fpath, index_col=0, dtype='object')
    unnamed_cols = df.columns[df.columns.str.contains("Unnamed")]
    if len(unnamed_cols) > 0:
        df.drop(columns=unnamed_cols, errors='ignore', inplace=True)
    if 'id' in df.columns:
        df.set_index('id', inplace=True)
    df[['left', 'right']] = df['reaction'].str.split(' <=> ', expand=True).apply(lambda x: x.str.findall(r'(C\d{5})'))
    df['has_chiral_info'] = (df.left + df.right).explode().isin(C_df.index).groupby(level=0).all()
    if df['has_chiral_info'].all():
        df.drop(columns='has_chiral_info', inplace=True)
    df['compounds'] = df.loc[:, ['left', 'right']].sum(axis=1)
    df.loc[:, ['left', 'right']] = df.loc[:, ['left', 'right']].map(set)
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


def expand_from_seed_set(seeds, reactions: pd.DataFrame, max_gen=1000):
    # seeds : list-like, the seed set of compounds from which to start the expansion
    # reactions : pool of reactions to draw from; if all compounds on one side of a reaction are present, the reaction is enabled
    # max_gen : max # of expansion generations calculated, default 1000. make small if you only care about early stages.
    rns, cps = dict(), {0: dict(zip(seeds, len(seeds) * [True]))}
    compound_pool, reaction_pool = set(seeds), reactions[['left', 'right']]
    fwd_rns, bwd_rns = set(), set()
    # actual expansion code starts here
    for g in range(1, max_gen + 1):
        # are all right-side compounds present?
        has_r = reaction_pool['right'].apply(lambda x: x <= compound_pool)
        # are all left-side compounds present?
        has_l = reaction_pool['left'].apply(lambda x: x <= compound_pool)
        # if so, reaction is enabled
        rns[g] = (has_r + has_l) > 0
        # new compounds will be from right side
        fwd_rns = fwd_rns.union(set(reaction_pool[has_l > has_r].index)) 
        # or new compounds will be from left side
        bwd_rns = bwd_rns.union(set(reaction_pool[has_l < has_r].index))
        # add new-compound side to the pool
        compound_pool = set.union(compound_pool,
                                  *reaction_pool.loc[rns[g]].stack())
        cps[g] = dict(zip(compound_pool, len(compound_pool) * [True]))
        # if no new compounds were added...
        if cps[g] == cps[g - 1]:  
            # stop expanding
            break
    # end of actual expansion code, start of processing
    rns, cps = pd.DataFrame(rns), pd.DataFrame(cps)
    rns = rns[rns.any(axis=1)].cummax(axis=1)
    cps = cps[cps.any(axis=1)].cummax(axis=1)
    cp_sets = pd.Series(
         {i: set(j) - {''} for i, j in cps.mul(cps.index, axis=0).fillna('').to_dict(orient='list').items()})
    try:
        rn_sets = get_rn_sets(rns)
    except:
        rn_sets = pd.Series(index=cp_sets.index, data=float('nan'))
    chiral_fraction = cp_sets.apply(lambda x: C_df.loc[list(x), 'is_chiral'].mean())
    participants = pd.DataFrame({'reactions': rn_sets, 'compounds': cp_sets})
    try:
        participants.loc[0, 'reactions'] = set(seeds) - set(seeds)
    except:
        participants.loc[0, 'reactions'] = pd.NA
    arrival_step = participants.stack().explode().drop_duplicates().dropna().reset_index().set_index(0)['level_0']
    participants = participants.join(participants.diff().fillna(participants).add_prefix('new '))
    # generate info for edge lists
    steps = rns.idxmax(axis=1)  # note first appearance of each reaction
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
        {'rns': rns, 'cps': cps, 'participants': participants,
         'seeds': seeds, 'arrival_step': arrival_step,
         'chiral_fraction': chiral_fraction, 'edge_pool': edge_pool})


# identifies what compounds and reactions in the database are reachable from at least one seed compound
def define_reachable_components(seeds, reactions: pd.DataFrame):
    bipartite_edges = reactions.map(list).stack().groupby(level=0).sum().explode().reset_index()
    bipartite_graph = nx.from_pandas_edgelist(bipartite_edges, source='id', target=0, create_using=nx.Graph)
    accessible_nodes = set.union(*[nx.node_connected_component(bipartite_graph, i) for i in seeds])
    del bipartite_graph
    del bipartite_edges
    return ({'reachable_compounds': [i for i in accessible_nodes if i.startswith('C')],
             'reachable_reactions': [i for i in accessible_nodes if not i.startswith('C')]})


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
    # change notation, so that max still points to reaction (not compound) in cases where reaction is a single atlas entry
    df['reaction'] = df['reaction'].str.replace('A', 'N')  
    sub_edges = df.explode('source').drop(columns='target').rename(columns={'reaction': 'target'})
    prod_edges = df.explode('target').drop(columns='source').rename(columns={'reaction': 'source'})
    all_edges = pd.concat([sub_edges, prod_edges], axis=0, ignore_index=True)
    all_edges['reaction'] = all_edges[['source', 'target']].max(axis=1)
    all_edges = all_edges.sort_values(by=['step', 'reaction', 'source', 'target']).reset_index(drop=True).loc[:,
                all_edges.nunique().sort_values().index]
    cols = ['source', 'target', 'reaction']
    all_edges.loc[:, cols] = all_edges.loc[:, cols].map(lambda x: x.replace('N', 'A'))  # revert back to original
    return all_edges


# adds context to each edge regarding its essentiality
def add_bipartite_edge_context(single_result):
    df = single_result['bipartite_edgelist']
    steps = single_result['arrival_step']
    df_arrival_step = df.loc[:,['reaction', 'source', 'target']].map(lambda x: steps.at[x], na_action='ignore').add_suffix('_step')
    df = df.join(df_arrival_step)
    impactful_precursors = df.query('directed').query(
        '(target==reaction & reaction_step>source_step) | (reaction==source & reaction_step==target_step)').index
    single_result['bipartite_edgelist'] = df.assign(impactful_precursor=df.index.isin(impactful_precursors))
    return single_result


def generate_immediate_precursor_unipartite_edge_list(single_result):
    """
    Get each compound's immediate precursor compounds (last-arriving prerequisites).
    """
    # Declare shorthand vars
    (s, t) = ('source', 'target')
    arr = single_result['arrival_step']
    # Get all edges between compounds.
    ep = single_result['edge_pool'].explode(s).explode(t).query(f'{s}!={t}')
    # Remove "infill" reactions; keep only the first instance of each edge.
    ep = ep.query('directed').sort_values(by='step').drop_duplicates(subset=[s, t])
    # Reset index so it's edge-based not reaction-based.
    ep = ep.reset_index(drop=True)
    # Keep only edges where source arrived immediately before target.
    ep = ep.loc[ep.eval(f'{t}.map(@arr) - {s}.map(@arr) == 1')].reset_index(drop=True)
    # Add this to the single_result dict.
    single_result.update({'immediate_precursor_unipartite_edgelist': ep})
    # Return the outcome
    return single_result


def bulk_measure_reaction_chirality(reactions: pd.DataFrame, compounds: pd.DataFrame):

    ccs = compounds['n_chiral_centers']

    def _explode_side_components(side_Series: pd.Series) -> pd.DataFrame:
        return (side_Series.explode()
                .str.split(' ', expand=True)
                .rename(columns={0:'n', 1:'id'})
                .rename_axis('rid')
                .assign(ccs = lambda x: x.id.map(ccs)))

    def _sum_chiral_centers(df: pd.DataFrame) -> pd.Series:
        """ Calculate the total number of chiral centers per index. """
        return(df.drop(columns='id').astype(int)
            .prod(axis=1)
            .groupby(level=0)
            .sum().copy(deep=True))

    def _find_n_gte_1(s, letter='n'):
        """ Find a positive integer value of n such that a string expression s containing n evaluates to a positive integer. """
        n = 1
        keep_going = True
        while keep_going:
            out = eval(s.replace(letter, str(n)))
            if out > 0:
                keep_going = False
            else:
                n += 1
        return n

    # Identify reactions with alpha coefficients (n/m/x/y/z/w).
    alpha_ids = reactions.query('~reaction.str.isupper()').index

    # Identify reactions with negative coefficients.
    negative_ids = reactions.query('reaction.str.startswith("-") or reaction.str.contains(" -")').index

    # Drop those. Separate reactions into sides and their components. Each component has a coefficient and id.
    sides = (reactions.reaction.str.split(' <=> ', expand=True)).map(lambda s: s.split(' + ')).drop(negative_ids)
    L = _explode_side_components(sides[0])
    R = _explode_side_components(sides[1])

    # Calculate #CCs per side for reactions without alpha coefficients, then isolate remaining reactions.
    rxn_ccs = pd.DataFrame({'L': _sum_chiral_centers(L.drop(alpha_ids)),
                            'R': _sum_chiral_centers(R.drop(alpha_ids))})
    L = L.drop(rxn_ccs.index)
    R = R.drop(rxn_ccs.index)

    # Isolate remaining reactions and format alpha coefficients for evaluation.
    _format_mult_str = lambda Ser: Ser.str.replace(r'(?<=\d)(?=[a-z])|(?<=[a-z])(?=[a-z])', '*', regex=True)
    L['n'] = _format_mult_str(L['n'])
    R['n'] = _format_mult_str(R['n'])

    # Update selected coefficients to eliminate false "minus" signs in next step.
    to_correct = {'n-x+x': 'n', 'm-1+n+1': 'n+m'}
    L['n'] = L['n'].replace(to_correct)
    R['n'] = R['n'].replace(to_correct)

    # ASSESS ALPHA COEFFICIENTS
    # If all alpha coeffs are set to 1, coeffs with a minus sign somewhere (e.g. "n-1") could net to zero.
    # To avoid this, for each reaction, we find the smallest value(s) of n/m/x/y/z/w such that all coefficients net to >= 1.
    # NOTE: NO attempt is made to balance the actual reaction. i.e., we do not check how a given alpha-substitution affects
    # the overall stoichiometry.

    min_vals = pd.concat([L, R]).query('n.str.contains("-")')

    # Since some reactions have multiple alpha coeffs (e.g. n and m), label which one is having its value set through this process.
    min_vals['letter'] = min_vals['n'].str.findall(r'[a-z]').map(''.join)

    # If there's more than one letter in a single coefficient with a minus sign (e.g. n-m C00001, n+m-5 C00002)...
    # the smallest acceptable values would co-vary. Future updates should handle those cases if they occur, i.e.:
    # TODO: For reactions w/'-' and multiple letters, add a check: do the answers from single-letter components produce a positive coeff?
    exceptions = min_vals.query('letter.str.len()>1').copy(deep=True)
    if len(exceptions.index) > 0:
        print('multi-alpha exception(s) found in reactions: ', exceptions.index)
    
    min_vals = min_vals.query('letter.str.len()==1')
    min_vals = min_vals.assign(v = lambda x: x['n'].map(_find_n_gte_1))
    min_vals = min_vals.groupby(by=['rid', 'letter'])['v'].max()

    # Replace the alpha coefficients for which a miniumum value was found.
    for (rn, ltr), v in min_vals.items():
        L.loc[[rn], 'n'] = L.loc[[rn], 'n'].str.replace(ltr, str(v))
        R.loc[[rn], 'n'] = R.loc[[rn], 'n'].str.replace(ltr, str(v))

    # Then replace outstanding alpha coeffs with 1.
    letter_coeffs = pd.concat([L, R]).query('n.str.islower()')['n'].str.findall(r'[a-z]').explode().unique()
    for ltr in letter_coeffs:
        L.loc[:, 'n'] = L['n'].str.replace(ltr, '1')
        R.loc[:, 'n'] = R['n'].str.replace(ltr, '1')
    L['n'], R['n'] = L['n'].map(eval), R['n'].map(eval)

    # Finally, calculate the total #CCs per side.
    rxn_ccs_alpha = pd.DataFrame(
        {'L': (L.ccs * L.n).groupby(level=0).sum(),
        'R': (R.ccs * R.n).groupby(level=0).sum()})

    rxn_ccs = pd.concat([rxn_ccs, rxn_ccs_alpha]).rename_axis('id')
    rxn_ccs.columns = ['Reactant Chiral Count', 'Product Chiral Count']

    return rxn_ccs


# VARIABLES AND DATASETS
C_df = (pd.read_csv(compound_file)
                  .assign(is_chiral=lambda x: x.n_chiral_centers > 0)
                  .set_index('compound_id'))

R_df = import_reaction_file(reaction_file)

rxn_ccs = bulk_measure_reaction_chirality(R_df, C_df)

# ec_rns: KEGG reaction-enzyme entry linkages.
ec_rns = pd.read_csv(reaction_file, usecols=['id', 'ec'], index_col=0).iloc[:,0].str.split().explode().reset_index()
ec_rns = ec_rns.query('~ec.str.contains("-", na=True)').groupby('ec')['id'].apply(list)
rn_cps = pd.read_csv(reaction_file, usecols=['id', 'reaction'], index_col=0).iloc[:,0].str.findall(r'(C\d{5})')
seed_candidates = pd.read_csv(seed_file, index_col=0)
