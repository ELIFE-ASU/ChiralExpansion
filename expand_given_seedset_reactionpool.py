# For each reaction pool, for each seed set, run expansion and save results.
from Code.NetworkExpansion import *
from Code.externalize_results import *
import networkx as nx
import matplotlib.pyplot as plt
import os

write_outputs = True
max_generation = 80
seed_groups = ['JM', 'JM|G17', 'JM|G24', 'G24', 'JM|G24|TW', 'JM|G24|G17|TW']

for reaction_file in [i for i in os.listdir('Data/reaction_pools/') if i.endswith('.csv.zip')]:
    model_run_name = reaction_file.replace('_processed','').replace('.csv.zip','').replace('_data_R','')
    
    # define chemical scope
    reaction_scope = get_sides(import_reaction_file('reaction_pools/'+reaction_file))
    compound_scope = set.union(*reaction_scope.stack())
    seed_scope = seed_candidates.loc[seed_candidates.index.isin(compound_scope)]
    seed_sets = {i: seed_scope.query('Source.str.contains("'+i+'")').index for i in seed_groups}

    # run network expansions
    results = dict()
    for name, seeds in seed_sets.items():
        results[name] = expand_from_seed_set(seeds=seeds, reactions=reaction_scope, max_gen=max_generation)
        results[name]['bipartite_edgelist'] = make_bipartite_edgelist(results[name]['edge_pool'])

    # summarize results
    def result_to_df(key: str, to_apply):
        return pd.concat({i: results[i][key].apply(to_apply) for i in results.keys()}, axis=1)
    summaries = {'compound count': result_to_df(key='cps', to_apply=lambda x: x.sum()),
                'reaction count': result_to_df(key='rns', to_apply=lambda x: x.sum()),
                'chiral fraction': result_to_df(key='chiral_fraction', to_apply=lambda x: x)}

    # plot results
    plot_expansion_summary(summaries, filename=model_run_name+'_seed_set_comparison.pdf')

    # write output files
    if write_outputs:
        output_results = dict()
        for i in results.keys():
            output_results[model_run_name+'_'+i] = results[i]
        write_edgelist_csvs(results=output_results)


