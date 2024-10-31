import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #To ignore future warnings in network expansion code

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from NetworkExpansion import *
from utilities import *

### Global variables & methods (usually overwriting those from NetworkExpansion.py)
max_generation = 1000 #Arbitrarily large value
seed_candidates = pd.read_csv('../Data/SampleSeeds.csv.zip', index_col=0, compression='zip')
keggC_df = pd.read_csv("../../lets-get-kegg/Data/kegg_data_C.csv.zip", index_col=0, compression="zip")

def fig2(seed_name, random_seed_set_size, n_iter):
    ### Get full seed set & reaction set
    seed_groups = [seed_name]
    rxn_db = "../../lets-get-kegg/Data/kegg_data_R_processed.csv.zip" #Default in NetworkExpansion.import_reaction_file
    reaction_scope = get_sides(import_reaction_file(rxn_db))

    compound_scope = set.union(*reaction_scope.stack())
    seed_scope = seed_candidates.loc[seed_candidates.index.isin(compound_scope)]
    seed_sets = {i: seed_scope.query('Source.str.contains("'+i+'")').index for i in seed_groups}
    print("- Seed Set -")

    ### Run network expansion on original seed set
    results = dict()
    for name, seeds in seed_sets.items():
        print(seeds)
        results[name] = expand_from_seed_set(seeds=seeds, reactions=reaction_scope, max_gen=max_generation)
        results[name]['bipartite_edgelist'] = make_bipartite_edgelist(results[name]['edge_pool'])

    ### Process expansion (including chiral %s at each step & heavy atom count)
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
    ### Determines achiral/chiral crossover & first chiral compound
    achiral_chiral_crossover_step = 0
    first_chiral_step = 0
    chiral_count = summaries["chiral compound count"][0].to_list()
    achiral_count = summaries["achiral compound count"][0].to_list()
    for i in range(len(summaries["chiral compound count"][0].to_list())):
        if chiral_count[i] > 0 and first_chiral_step == 0:
            first_chiral_step = i
        if chiral_count[i] > achiral_count[i] and achiral_chiral_crossover_step == 0:
            achiral_chiral_crossover_step = i

    ### Chiral / achiral / TODO: LUCA compounds panel (Fig2B)

    # Colors
    achiral_color = "#FFC012"
    chiral_color = "#34CCFF"
    luca_color = "indigo"

    _, ax = plt.subplots(figsize=(10,8))
    ax.plot(summaries["chiral compound count"], color=chiral_color, marker="o", linewidth=4, markersize=10,
         label="Chiral Compounds")
    ax.plot(summaries["achiral compound count"],  color=achiral_color, marker="o", linewidth=4, markersize=10,
            label="Achiral Compounds")
    ax.axvline(x=achiral_chiral_crossover_step, color="k", linestyle="--", alpha=0.9)
    ax.axvline(x=first_chiral_step, color="k", linestyle=":", alpha=0.5)

    ax.annotate(f"Achiral-to-Chiral\nCrossover\nStep {achiral_chiral_crossover_step}", xy=(achiral_chiral_crossover_step, 2500), 
                xytext=(achiral_chiral_crossover_step+8,2850), color="k", arrowprops={"arrowstyle": "simple", "color": "k"}, fontsize=14, 
                ha='center', va='center', zorder=3)
    ax.annotate(f"First Chiral\nCompound\nStep {first_chiral_step}", xy=(first_chiral_step, 2500), 
                xytext=(first_chiral_step+5,2850), color="k", arrowprops={"arrowstyle": "simple", "color": "k"}, fontsize=14, 
                ha='center', va='center', zorder=3)

    ax.set_ylim([0, 3100])
    ax.set_ylabel("Compounds")
    ax.set_xlabel("Expansion Step")

    ax.legend(prop={"size":14})

    plt.savefig("../Figures/Fig2B.svg", format="svg")
    plt.savefig("../Figures/Fig2B.png", format="png")
    plt.close()


    ### Heavy atom count panel (Fig2C)
    _, ax = plt.subplots(figsize=(10,8))

    ax.errorbar(x=np.arange(0, len(summaries["achiral avg heavy atoms"][0].to_list()), 1), 
                y=summaries["achiral avg heavy atoms"][0].to_list(), yerr=summaries["achiral std heavy atoms"][0].to_list(),
                linewidth=3, label="Achiral", color=achiral_color)

    ax.errorbar(x=np.arange(0, len(summaries["chiral avg heavy atoms"][0].to_list()), 1), 
                y=summaries["chiral avg heavy atoms"][0].to_list(), yerr=summaries["chiral std heavy atoms"][0].to_list(),
                linewidth=3, label="Chiral", alpha=0.6, color=chiral_color)

    ax.axvline(x=achiral_chiral_crossover_step, color="k", alpha=0.9, linestyle="--")
    ax.axhspan(6, 13, alpha=0.5, color='grey', label="Modeled Crossover") ### TODO: depends on Fig 1 (check this?)
    ax.axvline(x=first_chiral_step, color="k", alpha=0.5, linestyle=":")

    ax.set_ylim([0,55])

    ax.set_ylabel("Average Heavy Atom Count")
    ax.set_xlabel("Expansion Step")
    
    ax.legend(prop={"size":14})

    plt.savefig("../Figures/Fig2C.svg", format="svg")
    plt.savefig("../Figures/Fig2C.png", format="png")
    plt.close()

    ### Get X random seed sets, both chiral & achiral
    randAchiral_results = dict()
    randChiral_results = dict()
    random_achiral_seeds = dict()
    random_chiral_seeds = dict()
    for i in range(n_iter):
        random_achiral_seeds[f"Random_{i}"] = getRandomSeedSet(keggC_df, random_seed_set_size, chirality="achiral")
        random_chiral_seeds[f"Random_{i}"] = getRandomSeedSet(keggC_df, random_seed_set_size, chirality="chiral")
        
    print(f"- Expanding {n_iter} Achiral Seed Sets -")
    for name, seeds in tqdm(random_achiral_seeds.items(), total=n_iter):
        randAchiral_results[name] = expand_from_seed_set(seeds=seeds, reactions=reaction_scope, max_gen=max_generation)

    print(f"- Expanding {n_iter} Chiral Seed Sets -")
    for name, seeds in tqdm(random_chiral_seeds.items(), total=n_iter):
        randChiral_results[name] = expand_from_seed_set(seeds=seeds, reactions=reaction_scope, max_gen=max_generation)

    print("- Analyzing & Plotting Random Expansions -")
    achiral_full_chiral_count = list()
    achiral_full_achiral_count = list()
    result_count = 0
    for _, result in randAchiral_results.items():
        seed_name = f"Random_{result_count}"
        achiral_full_chiral_count.append(getCpdCount(randAchiral_results[seed_name]["cps"], keggC_df, ischiral_flag=True)[0].to_list())
        achiral_full_achiral_count.append(getCpdCount(randAchiral_results[seed_name]["cps"], keggC_df, ischiral_flag=False)[0].to_list())
        result_count += 1

    chiral_full_chiral_count = list()
    chiral_full_achiral_count = list()
    result_count = 0
    for _, result in randChiral_results.items():
        seed_name = f"Random_{result_count}"
        chiral_full_chiral_count.append(getCpdCount(randChiral_results[seed_name]["cps"], keggC_df, ischiral_flag=True)[0].to_list())
        chiral_full_achiral_count.append(getCpdCount(randChiral_results[seed_name]["cps"], keggC_df, ischiral_flag=False)[0].to_list())
        result_count += 1

    achiral_chiralPercentage = []
    chiral_chiralPercentage = []
    achiral_avg_percentages_chiral = []
    achiral_std_percentages_chiral = []
    chiral_avg_percentages_chiral = []
    chiral_std_percentages_chiral = []

    for i in range(len(achiral_full_chiral_count)):
        achiral_chiralPercentage.append(get_ChiralPercentage(np.cumsum(achiral_full_chiral_count[i]), np.cumsum(achiral_full_achiral_count[i])))

    for i in range(len(chiral_full_chiral_count)):
        chiral_chiralPercentage.append(get_ChiralPercentage(np.cumsum(chiral_full_chiral_count[i]), np.cumsum(chiral_full_achiral_count[i])))

    for i in range(max_generation):
        percentages = [] 
        for result in achiral_chiralPercentage:
            if len(result) > i:
                percentages.append(result[i])

        achiral_avg_percentages_chiral.append(np.mean(percentages))
        achiral_std_percentages_chiral.append(np.std(percentages))

        percentages = [] 
        for result in chiral_chiralPercentage:
            if len(result) > i:
                percentages.append(result[i])

        chiral_avg_percentages_chiral.append(np.mean(percentages))
        chiral_std_percentages_chiral.append(np.std(percentages))

    #Remove extra np.NaN values
    achiral_avg_percentages_chiral = [x for x in achiral_avg_percentages_chiral if x==x]
    achiral_std_percentages_chiral = [x for x in achiral_std_percentages_chiral if x==x]
    chiral_avg_percentages_chiral = [x for x in chiral_avg_percentages_chiral if x==x]
    chiral_std_percentages_chiral = [x for x in chiral_std_percentages_chiral if x==x]

    ### Figure 2D
    _, ax = plt.subplots(figsize=(10,8))
    #Chiral errorbar 
    ax.errorbar(x=np.arange(0, len(chiral_avg_percentages_chiral), 1), y=chiral_avg_percentages_chiral, 
             yerr=chiral_std_percentages_chiral, color=chiral_color, marker="o", linewidth=3, 
             markersize=10, label="Chiral")

    #Achiral errorbar
    ax.errorbar(x=np.arange(0, len(achiral_avg_percentages_chiral), 1), y=achiral_avg_percentages_chiral, 
                yerr=achiral_std_percentages_chiral, color=achiral_color, marker="o", linewidth=3, 
                markersize=10, label="Achiral")

    #Original (Figure 2B) percentages
    original_chiralPercent = []
    for i in range(len(chiral_count)):
        original_chiralPercent.append(chiral_count[i] / (chiral_count[i] + achiral_count[i]))
    ax.plot(original_chiralPercent, "o-", color="k", linewidth=3, zorder=5, label="Original Expansion $\mathbb{C}$")

    ax.axvline(x=14, color="k", linestyle="--", alpha=0.9)
    ax.axvline(x=3, color="k", linestyle=":", alpha=0.5)

    ax.set_ylim([0,1.05])

    ax.set_xlabel("Expansion Step")
    ax.set_ylabel("Chiral Order Parameter $\mathbb{C}$")
    ax.legend(prop={"size": 18}) # bbox_to_anchor=(1.05,0.60))

    plt.savefig("../Figures/Fig2D.svg", format="svg")
    plt.savefig("../Figures/Fig2D.png", format="png")
    plt.close()

    return achiral_chiral_crossover_step

def fig3(seed_name, crossover_step):
    ### Get full seed set & reaction set
    seed_groups = [seed_name]
    rxn_db = "../../lets-get-kegg/Data/kegg_data_R_processed.csv.zip" #Default in NetworkExpansion.import_reaction_file
    reaction_scope = get_sides(import_reaction_file(rxn_db))

    compound_scope = set.union(*reaction_scope.stack())
    seed_scope = seed_candidates.loc[seed_candidates.index.isin(compound_scope)]
    seed_sets = {i: seed_scope.query('Source.str.contains("'+i+'")').index for i in seed_groups}
    print("- Seed Set -")

    ### Run network expansion on original seed set
    results = dict()
    for name, seeds in seed_sets.items():
        print(seeds)
        results[name] = expand_from_seed_set(seeds=seeds, reactions=reaction_scope, max_gen=max_generation)

    expansion_arrival_df = linkArrivalStepHeavyAtoms(results[name]['arrival_step'], keggC_df)
    print(expansion_arrival_df)

    sns.set_palette(sns.color_palette("copper"))
    g = sns.jointplot(data=expansion_arrival_df.sort_values(by="n_heavy_atoms"), x="arrival", y="n_chiral_centers", 
                    hue="heavy atom percentile", ylim=(-1,31), xlim=(-1,60), s=100, marginal_kws={'common_norm':False})
    h = sns.regplot(x="arrival", y="n_chiral_centers", data=expansion_arrival_df, ax=g.ax_joint, 
                    scatter=False, color="k")
    g.ax_joint.axvline(x=crossover_step, color="k", linestyle="--", label=f"Original Crossover\n(Step {crossover_step})")
    g.set_axis_labels(xlabel="Expansion Step", ylabel="Chiral Centers")

    g.figure.set_size_inches((16, 8))

    g.ax_joint.legend(title="Heavy Atom\nPercentiles", prop={"size":12})

    plt.savefig("../Figures/Fig3b.png", format="png")
    plt.savefig("../Figures/Fig3b.svg", format="svg")
    plt.close()

    ### Get degrees for each compound & merge into expansion arrival dataframe 
    cpd_degrees = dict.fromkeys(expansion_arrival_df["compound_id"], 0)
    cpd_degrees = get_CpdDegrees(cpd_degrees=cpd_degrees)

    print(expansion_arrival_df)
    expansion_arrival_df["degree"] = expansion_arrival_df["compound_id"].map(cpd_degrees)
    print(expansion_arrival_df)

    # #Only include up to expansion step 14 to match network (TODO: why is this here?)
    # sub_df = df[df["Expansion Step"] <= 14]

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    scatter_hist(list(expansion_arrival_df["degree"]), list(expansion_arrival_df["n_chiral_centers"]), ax, ax_histy, "k")

    ax.set_xscale("log")

    ax.set_xlabel("Degree")
    ax.set_ylabel("Chiral Centers")

    plt.savefig("../Figures/Fig3c.png", format="png")
    plt.savefig("../Figures/Fig3c.svg", format="svg")
    plt.close()
    

def main():
    pd.set_option('future.no_silent_downcasting', True) #To avoid future warnings
    # Matplotlib defaults
    font = {"sans-serif" : "DejaVu Sans",
        "size" : 22}
    for key, value in font.items():
        plt.rcParams.update({"font." + key : value})
        
    # print("---- Generating Figure 2 ----")
    # #~Three minutes per 1000 random expansions (depends on chirality & size)
    # crossover_step = fig2(seed_name="JM", random_seed_set_size=10, n_iter=1000)

    crossover_step = 13 ##TODO: take out in full run

    print("---- Generating Figure 3 ----")
    fig3(seed_name="JM", crossover_step=crossover_step) 

if __name__ == "__main__":
    main()