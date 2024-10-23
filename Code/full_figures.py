import pandas as pd
from NetworkExpansion import *
from utilities import *

### Global variables & methods (usually overwriting those from NetworkExpansion.py)
max_generation = 1000 #Arbitrarily large value
seed_candidates = pd.read_csv('../Data/SampleSeeds.csv.zip', index_col=0, compression='zip')
keggC_df = pd.read_csv("../../lets-get-kegg/Data/kegg_data_C.csv.zip", index_col=0, compression="zip")

def fig2():
    ### 1) Get full seed set & reaction set
    seed_name = "JM"
    seed_groups = [seed_name]
    rxn_db = "../../lets-get-kegg/Data/kegg_data_R_processed.csv.zip" #Default in NetworkExpansion.import_reaction_file
    reaction_scope = get_sides(import_reaction_file(rxn_db))

    compound_scope = set.union(*reaction_scope.stack())
    seed_scope = seed_candidates.loc[seed_candidates.index.isin(compound_scope)]
    seed_sets = {i: seed_scope.query('Source.str.contains("'+i+'")').index for i in seed_groups}
    print(seed_sets)

    ### 2) Run network expansion
    model_run_name = "JM Testing"
    results = dict()
    for name, seeds in seed_sets.items():
        print('Expanding case: ', model_run_name, name, sep='\t')
        results[name] = expand_from_seed_set(seeds=seeds, reactions=reaction_scope, max_gen=max_generation)
        results[name]['bipartite_edgelist'] = make_bipartite_edgelist(results[name]['edge_pool'])

    ### 3) Process expansion (including chiral %s at each step & TODO heavy atom count)
    def result_to_df(key: str, to_apply):
        return pd.concat({i: results[i][key].apply(to_apply) for i in results.keys()}, axis=1)
    summaries = {'compound count': result_to_df(key='cps', to_apply=lambda x: x.sum()),
                'reaction count': result_to_df(key='rns', to_apply=lambda x: x.sum()),
                'chiral compound count': getCpdCount(results[seed_name]["cps"], keggC_df, ischiral_flag=True),
                'achiral compound count': getCpdCount(results[seed_name]["cps"], keggC_df, ischiral_flag=False),
                'chiral fraction': result_to_df(key='chiral_fraction', to_apply=lambda x: x),
                'chiral avg heavy atoms': getHeavyAtoms_perGen_fromKegg(results[seed_name]["cps"], keggC_df, ischiral_flag=True, calc="mean"),
                'achiral avg heavy atoms': getHeavyAtoms_perGen_fromKegg(results[seed_name]["cps"], keggC_df, ischiral_flag=False, calc="mean"),
                'chiral std heavy atoms': getHeavyAtoms_perGen_fromKegg(results[seed_name]["cps"], keggC_df, ischiral_flag=True, calc="std"),
                'achiral std heavy atoms': getHeavyAtoms_perGen_fromKegg(results[seed_name]["cps"], keggC_df, ischiral_flag=False, calc="std")    
                }
    print(summaries)

    ### 4) Chiral / achiral / LUCA compounds panel

    ### 5) Heavy atom count panel

    ### 6) Get X random seed sets (size: ??), both chiral & achiral

    ### 7) Run X iterations of network expansion, process chirality %
    ### 7a) Save data & add logic to read it, to save time in full figure code

    ### 8) % chiral panel
    pass

def main():
    print("---- Generating Figure 2 ----")
    fig2()

if __name__ == "__main__":
    main()