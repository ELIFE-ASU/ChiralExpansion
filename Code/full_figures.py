import pandas as pd
import numpy as np
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
    # print(seed_sets)

    ### 2) Run network expansion
    model_run_name = "JM Testing"
    results = dict()
    for name, seeds in seed_sets.items():
        # print('Expanding case: ', model_run_name, name, sep='\t')
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
    ### Commented out code determines achiral/chiral crossover (Step 13) & first chiral compound (Step 3)
    ### TODO: automate this
    # chiral_count = summaries["chiral compound count"][0].to_list()
    # achiral_count = summaries["achiral compound count"][0].to_list()
    # for i in range(len(summaries["chiral compound count"][0].to_list())):
    #     print(f"-- Step {i+1} --")
    #     print(f"Chiral Count:  {chiral_count[i]}")
    #     print(f"Achiral Count: {achiral_count[i]}")

    ### 4) Chiral / achiral / TODO: LUCA compounds panel (Fig2B)
    achiral_chiral_crossover_step = 13
    first_chiral_step = 3
    
    # Matplotlib defaults

    font = {"sans-serif" : "DejaVu Sans",
        "size" : 22}

    for key, value in font.items():
        plt.rcParams.update({"font." + key : value})

    achiral_color = "#FFC012"
    chiral_color = "#34CCFF"
    luca_color = "indigo"

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(summaries["chiral compound count"], color=chiral_color, marker="o", linewidth=4, markersize=10,
         label="Chiral Compounds")
    ax.plot(summaries["achiral compound count"],  color=achiral_color, marker="o", linewidth=4, markersize=10,
            label="Achiral Compounds")
    ax.axvline(x=13, color="k", linestyle="--", alpha=0.9)
    ax.axvline(x=3, color="k", linestyle=":", alpha=0.5)

    ax.annotate(f"Achiral-to-Chiral\nCrossover\nStep {achiral_chiral_crossover_step}", xy=(achiral_chiral_crossover_step, 2500), 
                xytext=(21,2850), color="k", arrowprops={"arrowstyle": "simple", "color": "k"}, fontsize=14, 
                ha='center', va='center', zorder=3)
    ax.annotate(f"First Chiral\nCompound\nStep {first_chiral_step}", xy=(first_chiral_step, 2500), 
                xytext=(8,2850), color="k", arrowprops={"arrowstyle": "simple", "color": "k"}, fontsize=14, 
                ha='center', va='center', zorder=3)

    ax.set_ylim([0, 3100])
    #ax.set_xlim([0,48])
    ax.set_ylabel("Compounds")
    ax.set_xlabel("Expansion Step")

    ax.legend(prop={"size":14})

    plt.savefig("../Figures/Fig2B.svg", format="svg")
    plt.savefig("../Figures/Fig2B.png", format="png")
    plt.close()


    ### 5) Heavy atom count panel (Fig2C)
    fig, ax = plt.subplots(figsize=(10,8))

    ax.errorbar(x=np.arange(0, len(summaries["achiral avg heavy atoms"][0].to_list()), 1), 
                y=summaries["achiral avg heavy atoms"][0].to_list(), yerr=summaries["achiral std heavy atoms"][0].to_list(),
                linewidth=3, label="Achiral", color=achiral_color)

    ax.errorbar(x=np.arange(0, len(summaries["chiral avg heavy atoms"][0].to_list()), 1), 
                y=summaries["chiral avg heavy atoms"][0].to_list(), yerr=summaries["chiral std heavy atoms"][0].to_list(),
                linewidth=3, label="Chiral", alpha=0.6, color=chiral_color)

    ax.axvline(x=achiral_chiral_crossover_step, color="k", alpha=0.9, linestyle="--")
    ax.axhspan(6, 13, alpha=0.5, color='grey', label="Modeled Crossover") ### TODO: depends on Fig 1 (check this?)
    ax.axvline(x=first_chiral_step, color="k", alpha=0.5, linestyle=":")

    ax.set_ylim([0,50])
    #ax.set_xlim([0,48])

    ax.set_ylabel("Average Heavy Atom Count")
    ax.set_xlabel("Expansion Step")
    
    ax.legend(prop={"size":14})

    plt.savefig("../Figures/Fig2C.svg", format="svg")
    plt.savefig("../Figures/Fig2C.png", format="png")
    plt.close()

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