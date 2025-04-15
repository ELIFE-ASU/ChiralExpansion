import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning) #To ignore future warnings in network expansion code
warnings.filterwarnings("ignore")

import matplotlib.gridspec as gridspec
import matplotlib.ticker as plticker
import seaborn as sns
import pickle
import csv
from networkx.algorithms import bipartite
import netwulf as nw
from netgraph import get_multipartite_layout

from network_expansion import *
from utilities import *

### Global variables & methods (usually overwriting those from network_expansion.py)
max_generation = 1000  # Arbitrarily large value
seed_candidates = pd.read_csv('../Data/SampleSeeds.csv',
                              index_col=0)
keggC_df = pd.read_csv("../Data/CBRdb_C.csv")
keggR_df = pd.read_csv("../Data/CBRdb_R.csv")


def fig1():
    # GDB data
    gdb_df = pd.read_csv("../Data/GDB_output.csv")
    # Chiral fractions at heavy atom counts
    chiral_fraction, achiral_fraction = get_chiral_fraction_at_heavy_atom_counts(
        keggC_df)
    gdb_cop, kegg_cop = get_chiral_order_parameters(keggC_df)
    print(f"- CBRdb Reaction Size = {len(keggR_df)} -")

    plt.rcParams['figure.constrained_layout.use'] = False
    achiral_color = "#FFC012"
    chiral_color = "#34CCFF"
    plt.figure(figsize=(24, 8))
    gs = gridspec.GridSpec(4, 6)

    ax1 = plt.subplot(gs[:, 0:2])
    ax1.plot(gdb_df["fraction_achiral"],
             linewidth=3,
             label="Achiral",
             color=achiral_color)
    ax1.plot(gdb_df["fraction_chiral"],
             linewidth=3,
             label="Chiral",
             color=chiral_color)
    ax1.axvspan(7, 12, alpha=0.5, color='grey')

    ax1.xaxis.grid(visible=True, which="both")
    ax1.yaxis.grid(visible=True, which="both")
    ax1.set_xlim(0, 16)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks(list(np.arange(0, 16)), minor=True)
    ax1.set_xlabel("Heavy Atoms Per Molecule")
    ax1.set_ylabel("Fraction of All Molecules")
    ax1.set_title("Chemical Space")
    ax1.annotate("B", xy=(0, 1.07), color="k", weight="bold", annotation_clip=False)

    ax2 = plt.subplot(gs[:, 2:4])
    ax2.plot(achiral_fraction,
             linewidth=3,
             label="Achiral",
             color=achiral_color)
    ax2.plot(chiral_fraction, linewidth=3, label="Chiral", color=chiral_color)
    ax2.axvspan(7,
                12,
                alpha=0.5,
                color='grey',
                label="Phase Boundary")

    ax2.legend(prop={"size": 20}, bbox_to_anchor=(1.01, 0.78), frameon=False)
    ax2.xaxis.grid(visible=True, which="both")
    ax2.yaxis.grid(visible=True, which="both")
    ax2.set_xlim(0, 16)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xticks(list(np.arange(0, 16)), minor=True)
    ax2.set_xlabel("Heavy Atoms Per Molecule")
    ax2.set_title("Metabolism")
    ax2.annotate("C", xy=(0, 1.07), color="k", weight="bold", annotation_clip=False)

    ax3 = plt.subplot(gs[1:, 4:])
    ax3.plot(gdb_cop, linewidth=3, label="Chemical Space", color="k", linestyle="--")
    ax3.plot(kegg_cop, linewidth=3, label="Metabolism", color="k", linestyle=":")

    ax3.legend(frameon=False, prop={"size": 20})
    ax3.set_xlim(0, 16)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_xticks(list(np.arange(0, 16)), minor=True)
    ax3.set_xlabel("Heavy Atoms Per Molecule")
    ax3.set_ylabel(r"Chiral Order Parameter $\mathbb{C}$")
    ax3.annotate(
        xy=(0.7, 0.7),
        text=r"$\mathbb{C} = \frac{C}{C+A}$",
    )
    ax3.annotate("D", xy=(0, 1.08), color="k", weight="bold", annotation_clip=False)

    plt.subplots_adjust(wspace=0.5)

    # Axis orientation is important, keeping figure 1 together
    plt.savefig("../Figures/Fig1.svg", format="svg")
    plt.savefig("../Figures/Fig1.png", format="png")
    plt.close()


def fig2(seed_name, rxn_db, min_ss_size=2, max_ss_size=10, n_iter=1000):
    ### Get full seed set & reaction set
    seed_groups = [seed_name]
    reaction_scope = get_sides(import_reaction_file(rxn_db))

    compound_scope = set.union(*reaction_scope.stack())
    seed_scope = seed_candidates.loc[seed_candidates.index.isin(compound_scope)]
    seed_sets = {
        i: seed_scope.query('Source.str.contains("' + i + '")').index
        for i in seed_groups
    }

    print(f"- Seed Set ({seed_name}) -")

    ### Run network expansion on original seed set
    results = dict()
    print("- Writing curated seed set to ../Data/curated_SeedSets.csv -")
    for name, seeds in seed_sets.items():
        print(f"- Seed Set Size: {len(seeds)}")
        print(seeds)
        curated_SS_df = pd.DataFrame(columns=[seed_name])
        curated_SS_df[seed_name] = list(seeds)
        curated_SS_df.to_csv("../Data/curated_SeedSets.csv")  ### Will be re-used for LUCA in SI figure notebook

        results[name] = expand_from_seed_set(seeds=seeds,
                                             reactions=reaction_scope,
                                             max_gen=max_generation)
        results[name]['bipartite_edgelist'] = make_bipartite_edge_list(results[name]['edge_pool'])

    ### Process expansion (including chiral %s at each step & heavy atom count)

    summaries = {
        'compound count':
            result_to_df(results, key='cps', to_apply=lambda x: x.sum()),
        'reaction count':
            result_to_df(results, key='rns', to_apply=lambda x: x.sum()),
        'chiral compound count':
            get_compound_chiral_count(results[seed_name]["cps"], keggC_df, chiral_flag=True),
        'achiral compound count':
            get_compound_chiral_count(results[seed_name]["cps"], keggC_df, chiral_flag=False),
        'chiral fraction':
            result_to_df(results, key='chiral_fraction', to_apply=lambda x: x),
        'chiral avg heavy atoms':
            get_heavy_atoms_per_gen_from_kegg(results[seed_name]["cps"],
                                              keggC_df,
                                              chiral_flag=True,
                                              calc="mean"),
        'achiral avg heavy atoms':
            get_heavy_atoms_per_gen_from_kegg(results[seed_name]["cps"],
                                              keggC_df,
                                              chiral_flag=False,
                                              calc="mean"),
        'chiral std heavy atoms':
            get_heavy_atoms_per_gen_from_kegg(results[seed_name]["cps"],
                                              keggC_df,
                                              chiral_flag=True,
                                              calc="std"),
        'achiral std heavy atoms':
            get_heavy_atoms_per_gen_from_kegg(results[seed_name]["cps"],
                                              keggC_df,
                                              chiral_flag=False,
                                              calc="std")
    }
    ### Determines achiral/chiral crossover & first chiral compound
    achiral_chiral_crossover_step = 0
    first_chiral_step = 0
    chiral_count = summaries["chiral compound count"][0].to_list()
    achiral_count = summaries["achiral compound count"][0].to_list()
    for i in range(len(summaries["chiral compound count"][0].to_list())):
        if chiral_count[i] > 0 and first_chiral_step == 0:
            first_chiral_step = i
        if chiral_count[i] > achiral_count[
            i] and achiral_chiral_crossover_step == 0:
            achiral_chiral_crossover_step = i

    print("- Chirality Statistics -")
    print(f"\tAchiral To Chiral Crossover Step: {achiral_chiral_crossover_step}")
    print(f"\tFirst Chiral Compound Appearence Step: {first_chiral_step}")
    print(
        f"\tExpansion Size At First Chiral Compound Appearence: {summaries['compound count'].iloc[first_chiral_step][seed_name]} Compounds, {summaries['reaction count'].iloc[first_chiral_step][seed_name]} Reactions")
    print(
        f"\tAverage Achiral Heavy Atoms At Crossover Step: {round(summaries['achiral avg heavy atoms'].iloc[achiral_chiral_crossover_step][0], 3)}")
    print(
        f"\tFinal Chiral Percentage (Generation {len(chiral_count)}): {round(chiral_count[-1] / (chiral_count[-1] + achiral_count[-1]), 3)}")
    print(f"\tFinal Chiral Cpd Count: {chiral_count[-1]}")
    print(f"\tFinal Achiral Cpd Count: {achiral_count[-1]}")
    print(f"\tFinal Total Cpd Count: {chiral_count[-1] + achiral_count[-1]}")

    ### Network figure
    print("- Generating Network Figure -")
    result_network = generate_annotated_bipartite_edge_list(results[seed_name])
    transition_edge_df = result_network['bipartite_edgelist'].query('step <= @achiral_chiral_crossover_step').copy(
        deep=True)
    networks = {i: nx.from_pandas_edgelist(transition_edge_df[transition_edge_df[i]],
                                           source='source', target='target', edge_attr=True, create_using=nx.DiGraph())
                for i in ['impactful_precursor', 'immediate_precursor', 'directed']}
    compound_nodes = {i: set(j.nodes()).difference(reaction_scope.index) for i, j in networks.items()}
    networks_uni = {i: bipartite.projected_graph(networks[i], nodes=compound_nodes[i]) for i in networks.keys()}
    print(networks_uni["impactful_precursor"])

    attributes = pd.read_csv("../Data/CBRdb_C.csv",
                             index_col=0)
    attributes.drop(labels=["smiles", "formula", "molecular_weight", "n_heavy_atoms"], axis=1,
                    inplace=True)
    attributes["is_chiral"] = attributes['n_chiral_centers'].map(lambda x: x > 0)
    attributes['is_chiral_color'] = attributes['is_chiral'].map({True: '#34CCFF', False: '#FFC012'}).fillna("#FFFFFF")

    steps = []
    for index, _ in attributes.iterrows():
        step = 0
        step_found = False
        for new_cpds in results[seed_name]["participants"]["new compounds"]:
            if index in new_cpds:
                steps.append(step)
                step_found = True
            step += 1
            if step > achiral_chiral_crossover_step:
                break
        if not step_found:
            steps.append(np.nan)

    attributes['gen'] = steps
    attributes.sort_values(by='gen', inplace=True)
    gen_color = 'Greys'
    attributes['gen_color'] = colorize_series(attributes['gen'], cmap=gen_color).tolist()

    ### Network figure
    show = 'impactful_precursor'
    rotate = False

    G1 = networks_uni[show]  # using impactful_precursor from network generation
    # Remove G nodes that are not in attributes df
    nodes_to_keep = attributes.index.to_list()
    nodes_to_remove = [n for n in nodes_to_keep if n not in G1.nodes()]
    print(f"\tFiltering  network to remove {len(nodes_to_remove)} nodes, as they are not in KEGG...")
    G1.remove_nodes_from(nodes_to_remove)

    nx.set_node_attributes(G1, attributes['is_chiral_color'], 'is_chiral_color')
    layers = [list(i) for i in attributes.loc[list(G1.nodes())].groupby(by='gen').groups.values()][:-1]
    pos = get_multipartite_layout(edges=list(G1.edges), layers=layers, reduce_edge_crossings=True)
    # for i in layers[-1]: del pos[i] # remove last layer
    for n in pos.keys(): pos[n][0] = (pos[n][0] + 0.2) ** 2  # spread out later gens to reduce extreme angles
    if rotate: pos = {i: j[::-1] for i, j in pos.items()}

    fig, ax = plt.subplots(figsize=(15, 15), layout='constrained')
    ax.axis("off")
    nx.draw_networkx(G1.subgraph(pos.keys()), pos=pos, ax=ax, edge_color=[0.5, 0.5, 0.5, 0.5],
                     node_color=list(gna(G1.subgraph(pos.keys()), 'is_chiral_color').values()),
                     edgecolors='k', width=0.03, arrows=False,
                     linewidths=0.2, node_size=100, node_shape='s', alpha=0.99,
                     with_labels=False)

    plt.legend(loc='lower left', borderpad=8, frameon=False, handles=[
        plt.Line2D([0], [0], marker='s', color='w', markeredgecolor='k', markeredgewidth=0.2, markerfacecolor='#34CCFF',
                   markersize=20, label='Chiral'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFC012', markeredgecolor='k', markeredgewidth=0.2,
                   markersize=20, label='Achiral')], prop={'size': 20})

    ax.annotate("A", xy=(1.31, 1), color="k", weight="bold", annotation_clip=False)

    plt.savefig("../Figures/Fig2A.png", format="png")
    plt.savefig("../Figures/Fig2A.svg", format="svg")
    plt.close()

    # Read in LUCA compounds & get percentages
    print("- Calculating LUCA Percentage Per Expansion Step -")
    percent_luca = get_luca_percentage(results[seed_name]["participants"])
    del (results)
    full_luca_exp = 0
    for step in range(len(percent_luca)):
        if percent_luca[step] == percent_luca[-1]:
            full_luca_exp = step + 1  # Since step starts at 0, add 1 to reflect network expansion algorithm
            full_luca_exp = step + 1  # Since step starts at 0, add 1 to reflect network expansion algorithm
            break
    print(f"\tFull LUCA Expansion Step: {full_luca_exp}")
    print(
        f"\tExpansion Size At Full LUCA Expansion: {summaries['compound count'].iloc[full_luca_exp][seed_name]} Compounds, {summaries['reaction count'].iloc[full_luca_exp][seed_name]} Reactions")
    print(f"\tPercent LUCA found in primordial expansion: {round(percent_luca[-1], 3)}")

    ### Chiral / achiral / LUCA compounds panel (Fig2B)
    # Colors
    achiral_color = "#FFC012"
    chiral_color = "#34CCFF"
    luca_color = "indigo"

    _, ax = plt.subplots(figsize=(10, 8))
    ax2 = ax.twinx()  # LUCA histogram

    ax.plot(summaries["chiral compound count"],
            color=chiral_color,
            marker="o",
            linewidth=4,
            markersize=10,
            label="Chiral Compounds")
    ax.plot(summaries["achiral compound count"],
            color=achiral_color,
            marker="o",
            linewidth=4,
            markersize=10,
            label="Achiral Compounds")
    ax.axvline(x=achiral_chiral_crossover_step,
               color="k",
               linestyle="--",
               alpha=0.9)
    ax.axvline(x=first_chiral_step, color="k", linestyle=":", alpha=0.5)

    ax.annotate(
        f"Achiral-to-Chiral\nCrossover\nStep {achiral_chiral_crossover_step}",
        xy=(achiral_chiral_crossover_step, 6000),
        xytext=(achiral_chiral_crossover_step + 8, 6150),
        color="k",
        arrowprops={
            "arrowstyle": "simple",
            "color": "k"
        },
        fontsize=14,
        ha='center',
        va='center',
        zorder=3)
    ax.annotate(f"First Chiral\nCompound\nStep {first_chiral_step}",
                xy=(first_chiral_step, 4750),
                xytext=(first_chiral_step + 10, 5250),
                color="k",
                arrowprops={
                    "arrowstyle": "simple",
                    "color": "k"
                },
                fontsize=14,
                ha='center',
                va='center',
                zorder=3)

    # ax.set_ylim([0, 3100])
    ax.set_ylabel("Cumulative Molecules")
    ax.set_xlabel("Expansion Generation")
    ax.annotate("B", xy=(-2, 6600), color="k", weight="bold", annotation_clip=False)

    ax2.set_ylim([0, 100])
    ax2.bar(np.arange(0, len(percent_luca), 1),
            percent_luca,
            alpha=0.15,
            color=luca_color,
            label="Percent LUCA Coverage")

    ax2.set_ylabel("Percent LUCA Coverage", color=luca_color)

    # Set y-axis tick labels to luca color
    ax2.tick_params(colors=luca_color)

    # Complete legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lgd = ax.legend(lines + lines2,
                    labels + labels2,
                    facecolor="white",
                    bbox_to_anchor=(1.08, 1),
                    prop={"size": 18})

    plt.savefig("../Figures/Fig2B.svg",
                format="svg",
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.savefig("../Figures/Fig2B.png",
                format="png",
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.close()

    ### Heavy atom count panel (Fig2C)
    _, ax = plt.subplots(figsize=(10, 8))

    ax.errorbar(x=np.arange(
        0, len(summaries["achiral avg heavy atoms"][0].to_list()), 1),
        y=summaries["achiral avg heavy atoms"][0].to_list(),
        yerr=summaries["achiral std heavy atoms"][0].to_list(),
        linewidth=3,
        label="Achiral",
        color=achiral_color)

    ax.errorbar(x=np.arange(
        0, len(summaries["chiral avg heavy atoms"][0].to_list()), 1),
        y=summaries["chiral avg heavy atoms"][0].to_list(),
        yerr=summaries["chiral std heavy atoms"][0].to_list(),
        linewidth=3,
        label="Chiral",
        alpha=0.6,
        color=chiral_color)

    ax.axvline(x=achiral_chiral_crossover_step,
               color="k",
               alpha=0.9,
               linestyle="--")
    ax.axhspan(
        8, 9, alpha=0.5, color='grey',
        label="Phase Boundary\nin Chemical Space")
    ax.axvline(x=first_chiral_step, color="k", alpha=0.5, linestyle=":")

    ax.set_ylim([0, 55])

    ax.set_ylabel("Average Heavy Atoms Per Molecule")
    ax.set_xlabel("Expansion Generation")

    ax.legend(prop={"size": 14})
    ax.annotate("C", xy=(-1.5, 56), color="k", weight="bold", annotation_clip=False)

    plt.savefig("../Figures/Fig2C.svg", format="svg")
    plt.savefig("../Figures/Fig2C.png", format="png")
    plt.close()

    ### Get X random seed sets, both chiral & achiral
    randAchiral_results = dict()
    randChiral_results = dict()
    random_achiral_seeds = dict()
    random_chiral_seeds = dict()
    for random_ss_size in range(min_ss_size, max_ss_size + 1):
        for i in range(n_iter):
            random_achiral_seeds[f"Random_{random_ss_size}_{i}"] = get_random_seed_set(
                keggC_df, random_ss_size, chirality="achiral")
            random_chiral_seeds[f"Random_{random_ss_size}_{i}"] = get_random_seed_set(
                keggC_df, random_ss_size, chirality="chiral")

    print(f"- Expanding {n_iter * (max_ss_size - min_ss_size + 1)} Achiral Seed Sets -")
    achiral_full_chiral_count = list()
    achiral_full_achiral_count = list()
    for name, seeds in tqdm(random_achiral_seeds.items(), total=n_iter * (max_ss_size - min_ss_size + 1)):
        randAchiral_results = expand_from_seed_set(
            seeds=seeds, reactions=reaction_scope, max_gen=max_generation)
        achiral_full_chiral_count.append(
            get_compound_chiral_count(randAchiral_results["cps"], keggC_df, chiral_flag=True)[0].to_list())
        achiral_full_achiral_count.append(
            get_compound_chiral_count(randAchiral_results["cps"], keggC_df, chiral_flag=False)[0].to_list())

    print(f"- Expanding {n_iter * (max_ss_size - min_ss_size + 1)} Chiral Seed Sets -")
    chiral_full_chiral_count = list()
    chiral_full_achiral_count = list()
    for name, seeds in tqdm(random_chiral_seeds.items(), total=n_iter * (max_ss_size - min_ss_size + 1)):
        randChiral_results = expand_from_seed_set(
            seeds=seeds, reactions=reaction_scope, max_gen=max_generation)
        chiral_full_chiral_count.append(
            get_compound_chiral_count(randChiral_results["cps"], keggC_df, chiral_flag=True)[0].to_list())
        chiral_full_achiral_count.append(
            get_compound_chiral_count(randChiral_results["cps"], keggC_df, chiral_flag=False)[0].to_list())

    achiral_chiralPercentage = []
    chiral_chiralPercentage = []
    achiral_avg_percentages_chiral = []
    achiral_std_percentages_chiral = []
    chiral_avg_percentages_chiral = []
    chiral_std_percentages_chiral = []

    for i in range(len(achiral_full_chiral_count)):
        achiral_chiralPercentage.append(
            get_chiral_percentage(np.cumsum(achiral_full_chiral_count[i]),
                                  np.cumsum(achiral_full_achiral_count[i])))

    for i in range(len(chiral_full_chiral_count)):
        chiral_chiralPercentage.append(
            get_chiral_percentage(np.cumsum(chiral_full_chiral_count[i]),
                                  np.cumsum(chiral_full_achiral_count[i])))

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

    # Remove extra np.NaN values
    achiral_avg_percentages_chiral = [
        x for x in achiral_avg_percentages_chiral if x == x
    ]
    achiral_std_percentages_chiral = [
        x for x in achiral_std_percentages_chiral if x == x
    ]
    chiral_avg_percentages_chiral = [
        x for x in chiral_avg_percentages_chiral if x == x
    ]
    chiral_std_percentages_chiral = [
        x for x in chiral_std_percentages_chiral if x == x
    ]

    ### Figure 2D
    _, ax = plt.subplots(figsize=(10, 8))
    # Chiral errorbar
    ax.errorbar(x=np.arange(0, len(chiral_avg_percentages_chiral), 1),
                y=chiral_avg_percentages_chiral,
                yerr=chiral_std_percentages_chiral,
                color=chiral_color,
                marker="o",
                linewidth=3,
                markersize=10,
                label="Chiral-Only Seeds")

    # Achiral errorbar
    ax.errorbar(x=np.arange(0, len(achiral_avg_percentages_chiral), 1),
                y=achiral_avg_percentages_chiral,
                yerr=achiral_std_percentages_chiral,
                color=achiral_color,
                marker="o",
                linewidth=3,
                markersize=10,
                label="Achiral-Only Seeds")

    # Original (Figure 2B) percentages
    original_chiralPercent = []
    for i in range(len(chiral_count)):
        original_chiralPercent.append(chiral_count[i] /
                                      (chiral_count[i] + achiral_count[i]))
    ax.plot(original_chiralPercent,
            "o-",
            color="k",
            linewidth=3,
            zorder=5,
            label="Primordial Expansion")

    ax.axvline(x=achiral_chiral_crossover_step, color="k", linestyle="--", alpha=0.9)
    ax.axvline(x=first_chiral_step, color="k", linestyle=":", alpha=0.5)

    ax.set_ylim([0, 1.05])

    ax.set_xlabel("Expansion Generation")
    ax.set_ylabel(r"Chiral Order Parameter $\mathbb{C}$")
    ax.legend(prop={"size": 18})  # bbox_to_anchor=(1.05,0.60))
    ax.annotate("D", xy=(-2, 1.07), color="k", weight="bold", annotation_clip=False)

    plt.savefig("../Figures/Fig2D.svg", format="svg")
    plt.savefig("../Figures/Fig2D.png", format="png")
    plt.close()

    return achiral_chiral_crossover_step, first_chiral_step, attributes, networks_uni


def fig3(seed_name, rxn_db, crossover_step, attributes, networks_uni):
    print("- Generating Network Figure (click 'Post To Python' in browser pop-up window to continue code) -")
    size_nodes_by = 'degree'
    max_node_size = 25

    G = networks_uni['directed'].copy()
    deg = G.degree()
    nodes_to_remove = [n[0] for n in deg if n[1] < 50]
    print(f"\tFiltering network to remove {len(nodes_to_remove)} nodes due to degree...")
    G.remove_nodes_from(nodes_to_remove)

    node_sizes = pd.Series({i: (np.log2(j + 2)) ** 1.5 for i, j in G.degree()})
    nx.set_node_attributes(G, node_sizes, 'size')
    nx.set_node_attributes(G, attributes['is_chiral_color'], 'group')

    nodes_without_group = list()
    for node, data in G.nodes(data=True):
        if 'group' not in data:
            nodes_without_group.append(node)

    print(f"\tRemoving {len(nodes_without_group)} nodes without group attribute...")
    for n in nodes_without_group:
        G.remove_node(n)

    fig = plt.figure(figsize=(12, 12), layout='constrained')
    stylized_network, config = nw.visualize(G, config={"zoom": 1.08,
                                                       "node_charge": -100,
                                                       "node_gravity": 0.4,
                                                       "link_distance": 0,
                                                       "link_distance_variation": 0,
                                                       "node_collision": True,
                                                       "wiggle_nodes": False,
                                                       "freeze_nodes": False,
                                                       "node_fill_color": "#79aaa0",
                                                       "node_stroke_color": "#555555",
                                                       "node_label_color": "#000000",
                                                       "node_size": 25,
                                                       "node_stroke_width": 2,
                                                       "node_size_variation": 0.75,
                                                       "display_node_labels": False,
                                                       "scale_node_size_by_strength": True,
                                                       "link_color": "#000000",
                                                       "link_width": 2,
                                                       "link_alpha": 0.01,
                                                       "link_width_variation": 0.5,
                                                       "display_singleton_nodes": True,
                                                       "min_link_weight_percentile": 0,
                                                       "max_link_weight_percentile": 1})
    fig, ax = nw.draw_netwulf(stylized_network)

    a = stylized_network.copy()
    a['nodeStrokeWidth'] = 0.0
    for i in list(range(len(a['nodes']))):
        a['nodes'][i]['radius'] = a['nodes'][i]['radius'] * 0.4
        a['nodes'][i]['color'] = attributes['gen_color'].loc[a['nodes'][i]['id']]

    nw.draw_netwulf(a, fig=fig, ax=ax, draw_links=False)

    nodes = pd.DataFrame(a['nodes'])
    gens_shown = \
        pd.DataFrame(
            {'gen': nodes['id'].map(attributes['gen']), 'color': nodes['color']}).drop_duplicates().sort_values(
            by='gen').set_index('gen')['color']
    handles = [plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='k', markeredgewidth=0, markerfacecolor=j,
                          markersize=10, label=str(i)) for i, j in gens_shown.items()]
    plt.legend(loc='lower left', frameon=False, handles=handles, handletextpad=0.01, title='GEN', prop={"size": 10})
    plt.annotate("A", xy=(25, 600), color="k", weight="bold", annotation_clip=False)

    plt.savefig("../Figures/Fig3A.svg", format="svg")
    plt.savefig("../Figures/Fig3A.png", format="png")
    plt.close()

    ### Get full seed set & reaction set
    seed_groups = [seed_name]
    reaction_scope = get_sides(import_reaction_file(rxn_db))

    compound_scope = set.union(*reaction_scope.stack())
    seed_scope = seed_candidates.loc[seed_candidates.index.isin(compound_scope)]
    seed_sets = {
        i: seed_scope.query('Source.str.contains("' + i + '")').index
        for i in seed_groups
    }
    print("- Seed Set -")

    ### Run network expansion on original seed set
    results = dict()
    for name, seeds in seed_sets.items():
        print(seeds)
        results[name] = expand_from_seed_set(seeds=seeds,
                                             reactions=reaction_scope,
                                             max_gen=max_generation)

    expansion_arrival_df = link_arrival_step_heavy_atoms(
        results[name]['arrival_step'], keggC_df)

    sns.set_palette(sns.color_palette("copper"))
    g = sns.jointplot(data=expansion_arrival_df.sort_values(by="n_heavy_atoms"),
                      x="arrival",
                      y="n_chiral_centers",
                      hue="heavy atom percentile",
                      ylim=(-1, 39),
                      xlim=(-1, 39),
                      s=100,
                      marginal_kws={'common_norm': False})
    h = sns.regplot(x="arrival",
                    y="n_chiral_centers",
                    data=expansion_arrival_df,
                    ax=g.ax_joint,
                    scatter=False,
                    color="k")
    # Last line in h is the regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=h.get_lines()[-1].get_xdata(),
                                                                   y=h.get_lines()[-1].get_ydata())
    print(f"- Regression Slope: {round(slope, 3)}, Intercept: {round(intercept, 3)}")
    g.ax_joint.axvline(x=crossover_step,
                       color="k",
                       linestyle="--",
                       label=f"Original Crossover\n(Step {crossover_step})")
    g.set_axis_labels(xlabel="Expansion Generation", ylabel="Chiral Centers")

    g.figure.set_size_inches((16, 8))
    g.ax_joint.legend(title="Heavy Atom\nPercentiles",
                      bbox_to_anchor=(0.75, 1),
                      prop={"size": 12})
    plt.annotate("B", xy=(-0.5, 37), color="k", weight="bold", annotation_clip=False)

    plt.savefig("../Figures/Fig3B.png", format="png")
    plt.savefig("../Figures/Fig3B.svg", format="svg")
    plt.close()

    ### Get degrees for each compound & merge into expansion arrival dataframe
    cpd_degrees = dict.fromkeys(expansion_arrival_df["compound_id"], 0)
    cpd_degrees = get_compound_degrees(cpd_degrees=cpd_degrees)

    expansion_arrival_df["degree"] = expansion_arrival_df["compound_id"].map(
        cpd_degrees)
    print("- Writing out full degree/compoundID/chiral-center-count dataframe to Data/expansion_arrival.csv -")
    expansion_arrival_df.to_csv("../Data/expansion_arrival.csv")

    print("- Writing out top 50 highest degree nodes to Data/top50_highestDegreeNodes.csv -")
    # expansion_arrival_df.sort_values(by=["degree", "n_chiral_centers"], inplace=True, ascending=[False, True])
    expansion_arrival_df.sort_values(by="degree", inplace=True, ascending=False)
    columns_to_write = ["compound_id", "formula", "n_chiral_centers", "degree"]
    expansion_arrival_df.head(50).to_csv("../Data/top50_highestDegreeNodes.csv", columns=columns_to_write, index=False)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2,
                          2,
                          width_ratios=(4, 1),
                          height_ratios=(1, 4),
                          left=0.1,
                          right=0.9,
                          bottom=0.1,
                          top=0.9,
                          wspace=0.05,
                          hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    scatter_hist(list(expansion_arrival_df["degree"]),
                 list(expansion_arrival_df["n_chiral_centers"]), ax, ax_histy,
                 "k")

    ax.set_xscale("log")

    ax.set_xlabel("Degree")
    ax.set_ylabel("Chiral Centers")
    ax.annotate("C", xy=(0.7, 45), color="k", weight="bold", annotation_clip=False)

    plt.savefig("../Figures/Fig3C.png", format="png")
    plt.savefig("../Figures/Fig3C.svg", format="svg")
    plt.close()


def fig4():
    ### Plotting details
    chiral_ls = "dashed"
    achiral_ls = "dotted"
    chnops_color = "dimgrey"
    kegg_color = "darkgrey"
    color_number = {
        "LUCA": "#10D410",
        "Archaea": "#D65E00",
        "Bacteria": "#0B77B5",
        "Eukarya": "#E69F03",
        "Metagenome": "#009E73",
        "Biosphere": "#4B0092"
    }

    ### Load data
    data_dir = "../Data/"
    df = dict()
    for domain in [
        "Archaea", "Metagenome", "Bacteria", "Eukarya", "Biosphere", "LUCA"
    ]:
        df[domain] = pd.read_csv(data_dir +
                                 "species_chirality_in_%s.csv" % domain)

    ## Reaxys Data
    chnops = pickle.load(
        file=open("../Data/chiral_chnops_sampling_FULL.p", "rb"))
    kegg = pickle.load(file=open("../Data/chiral_kegg_sampling_FULL.p", "rb"))
    mws = np.arange(100, 7100, 100)

    chnops_avgs = []
    for s in chnops:
        chnops_avgs.append(np.mean(s))
    kegg_avgs = []
    for s in kegg:
        kegg_avgs.append(np.mean(s))

    # Find unversalities, and rank-order each
    # Read in occurence data & create universality percentages
    occurrence_df = pd.read_csv('../Data/omics/compound2occurrence.csv', header=0, index_col=0)
    order = ['Bacteria', 'Archaea', 'Eukaryota', 'Metagenome']
    num_samples = occurrence_df.loc['N', order].copy()
    num_bacteria, num_archaea, num_eukarya, num_metagenomes = num_samples.to_list()
    occurrence_df = occurrence_df[order].drop(index=['N']).mask(lambda x: x==0)
    # Translate occurances into % universality
    universality_df = (occurrence_df / num_samples)

    print(f"- Total JGI Sample Count: {num_bacteria + num_archaea + num_eukarya + num_metagenomes} -")
    
    a_cpd_occurances = occurrence_df['Archaea'].dropna().to_dict()
    b_cpd_occurances = occurrence_df['Bacteria'].dropna().to_dict()
    e_cpd_occurances = occurrence_df['Eukaryota'].dropna().to_dict()
    m_cpd_occurances = occurrence_df['Metagenome'].dropna().to_dict()
    
    a_cpd_universality = universality_df['Archaea'].dropna().to_dict()
    b_cpd_universality = universality_df['Bacteria'].dropna().to_dict()
    e_cpd_universality = universality_df['Eukaryota'].dropna().to_dict()
    m_cpd_universality = universality_df['Metagenome'].dropna().to_dict()

    a_chiral_universality, a_achiral_universality = find_chiral(
        a_cpd_universality, keggC_df)
    a_chiral_universality = {
        k: v for k, v in sorted(a_chiral_universality.items(),
                                key=lambda item: item[1],
                                reverse=True)
    }
    a_achiral_universality = {
        k: v for k, v in sorted(a_achiral_universality.items(),
                                key=lambda item: item[1],
                                reverse=True)
    }

    b_chiral_universality, b_achiral_universality = find_chiral(
        b_cpd_universality, keggC_df)
    b_chiral_universality = {
        k: v for k, v in sorted(b_chiral_universality.items(),
                                key=lambda item: item[1],
                                reverse=True)
    }
    b_achiral_universality = {
        k: v for k, v in sorted(b_achiral_universality.items(),
                                key=lambda item: item[1],
                                reverse=True)
    }

    e_chiral_universality, e_achiral_universality = find_chiral(
        e_cpd_universality, keggC_df)
    e_chiral_universality = {
        k: v for k, v in sorted(e_chiral_universality.items(),
                                key=lambda item: item[1],
                                reverse=True)
    }
    e_achiral_universality = {
        k: v for k, v in sorted(e_achiral_universality.items(),
                                key=lambda item: item[1],
                                reverse=True)
    }

    m_chiral_universality, m_achiral_universality = find_chiral(
        m_cpd_universality, keggC_df)
    m_chiral_universality = {
        k: v for k, v in sorted(m_chiral_universality.items(),
                                key=lambda item: item[1],
                                reverse=True)
    }
    m_achiral_universality = {
        k: v for k, v in sorted(m_achiral_universality.items(),
                                key=lambda item: item[1],
                                reverse=True)
    }

    ### Full Gridspec figure (formatting is important here)
    plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(10, 10)
    num_bins = 100

    ax1 = plt.subplot(gs[0:5, 0:6])

    list_domain = ["Metagenome", "Bacteria", "Archaea", "Eukarya"]

    handles = [
    ]  # Handle array used for legend compatibility with Reaxys sampling

    for domain in list_domain:
        x = df[domain].total_size
        y = df[domain].ratio_chiral
        s = ax1.scatter(x,
                        y,
                        label=domain,
                        s=5,
                        color=color_number[domain],
                        alpha=0.8,
                        zorder=2)
        handles.append(s)  # Adding lines to an array for legend labeling
        find_fit(x, y, color_number[domain])

    # Add Reaxys averages to scatter plot
    ax1.scatter(mws, chnops_avgs, color=chnops_color, s=15, zorder=2)

    ax1.scatter(mws, kegg_avgs, color=kegg_color, s=15, zorder=2)

    # Add scaling fits (fit a polyfit line to averages)
    find_fit(mws, chnops_avgs, chnops_color)
    find_fit(mws, kegg_avgs, kegg_color)

    # Line for biosphere
    x = df["Biosphere"].total_size
    chiral_y = df["Biosphere"].ratio_chiral.values
    s = ax1.scatter(x, chiral_y, s=100, color="k")  # , label="Biosphere")
    ax1.annotate("Biosphere", (x - 150, chiral_y + 0.01),
                 color="k",
                 weight='bold',
                 fontsize=16,
                 ha='center',
                 va='center')
    # handles.append(s)
    ax1.axhline(chiral_y, 0, 1, ls='--', color="k", alpha=1, zorder=1)

    # Line for Luca
    x = df["LUCA"].total_size
    chiral_y = df["LUCA"].ratio_chiral.values
    s = ax1.scatter(x, chiral_y, s=100, color="k", zorder=2)  # , label="LUCA")
    ax1.annotate("LUCA", (x - 250, chiral_y + 0.01),
                 color="k",
                 weight='bold',
                 fontsize=16,
                 ha='center',
                 va='center',
                 zorder=2)
    ax1.axhline(chiral_y, 0, 1, ls='--', color="k", alpha=1, zorder=1)

    # Add Reaxys to Legend
    chnops_patch = plt.Line2D([], [],
                              color=chnops_color,
                              marker="o",
                              markersize=2,
                              label='Synthesizable Chemistry: CHNOPS')
    kegg_patch = plt.Line2D([], [],
                            color=kegg_color,
                            marker="o",
                            markersize=2,
                            label='Synthesizable Chemistry: All Metabolic Elements')
    handles.extend([chnops_patch, kegg_patch])

    # Labels
    font_size = 14
    ax1.set_xlabel("System Size (Unique Molecules)"
                   )  # TODO: why isn't this label showing up
    ax1.set_ylabel(r"Chiral Order Parameter $\mathbb{C}$")
    ax1.legend(bbox_to_anchor=(1.43, 1.2),
               markerscale=5,
               handles=handles,
               ncol=3,
               prop={"size": 20})

    # Axis
    ax1.set_ylim(0.4, 0.8)
    ax1.set_xlim(-300, 7900)
    ax1.annotate("A", xy=(-100, 0.78), color="k", weight="bold", annotation_clip=False)

    ### Inset distribution axes
    ax2 = plt.subplot(gs[0:5, 7:])

    # Graph Reaxys graphs first so they're in the back
    ax2.hist([x for sub in kegg for x in sub],
             num_bins,
             label="Reaxys:\nKEGG Elements",
             facecolor=kegg_color,
             alpha=0.8,
             orientation="horizontal")
    ax2.hist([x for sub in chnops for x in sub],
             num_bins,
             label="Reaxys:\nCHNOPS Elements",
             facecolor=chnops_color,
             alpha=0.8,
             orientation="horizontal")

    list_domain = ["Bacteria", "Metagenome", "Archaea", "Eukarya"]
    for domain in list_domain:
        x = df[domain].total_size
        y = df[domain].ratio_chiral
        num_bins = 100
        n, bins, patches = ax2.hist(y,
                                    num_bins,
                                    label=domain,
                                    facecolor=color_number[domain],
                                    alpha=0.8,
                                    orientation="horizontal")

    # Line for biosphere
    x = df["Biosphere"].total_size
    y = df["Biosphere"].ratio_chiral

    # Line for Luca
    x = df["LUCA"].total_size
    y = df["LUCA"].ratio_chiral

    # Labels
    ax2.set_ylabel(r"Chiral Order Parameter $\mathbb{C}$")
    ax2.set_xlabel("Number of Genomic\nSamples")

    ax2.set_xscale('log')
    ax2.tick_params(axis="both", which="major")
    ax2.set_ylim(0.4, 0.8)
    ax2.annotate("B", xy=(700, 0.78), color="k", weight="bold", annotation_clip=False)

    ax3 = plt.subplot(gs[6:, 0:5])
    ax4 = plt.subplot(gs[6:, 6:])

    print("-- Table A Data --")
    m_chiral = make_auc_dashes(m_chiral_universality, "Metagenome, Chiral",
                               color_number["Metagenome"], chiral_ls, ax3)
    m_achiral = make_auc_dashes(m_achiral_universality, "Metagenome, Achiral",
                                color_number["Metagenome"], achiral_ls, ax3)

    e_chiral = make_auc_dashes(e_chiral_universality, "Eukarya, Chiral",
                               color_number["Eukarya"], chiral_ls, ax3)
    e_achiral = make_auc_dashes(e_achiral_universality, "Eukarya, Achiral",
                                color_number["Eukarya"], achiral_ls, ax3)

    b_chiral = make_auc_dashes(b_chiral_universality, "Bacteria, Chiral",
                               color_number["Bacteria"], chiral_ls, ax3)
    b_achiral = make_auc_dashes(b_achiral_universality, "Bacteria, Achiral",
                                color_number["Bacteria"], achiral_ls, ax3)

    a_chiral = make_auc_dashes(a_chiral_universality, "Archaea, Chiral",
                               color_number["Archaea"], chiral_ls, ax3)
    a_achiral = make_auc_dashes(a_achiral_universality, "Archaea, Achiral",
                                color_number["Archaea"], achiral_ls, ax3)

    ax3.grid()
    ax3.set_ylabel("Universality")
    ax3.set_xlabel("Normalized rank-order")

    # Legend
    handles = []
    chiral_patch = plt.Line2D([], [],
                              color="k",
                              linestyle="--",
                              linewidth=5,
                              label='Chiral')
    achiral_patch = plt.Line2D([], [],
                               color="k",
                               linestyle=":",
                               linewidth=5,
                               label='Achiral')

    ## Add Reaxys patches to handles
    handles.extend([chiral_patch, achiral_patch])
    ax3.legend(handles=handles, prop={"size": 20})

    # Hide the right and top spines
    ax3.spines[['right', 'top']].set_visible(False)
    ax3.set_xticklabels(["", "", "", "", "", ""])
    ax3.annotate("C", xy=(0, 1.05), color="k", weight="bold", annotation_clip=False)

    ### Ax4, chiral differences
    get_chiral_difference(m_chiral, m_achiral, color_number["Metagenome"],
                          "Metagenome", ax4)
    get_chiral_difference(a_chiral, a_achiral, color_number["Archaea"],
                          "Archaea", ax4)
    get_chiral_difference(b_chiral, b_achiral, color_number["Bacteria"],
                          "Bacteria", ax4)
    get_chiral_difference(e_chiral, e_achiral, color_number["Eukarya"],
                          "Eukarya", ax4)

    ax4.grid()

    ax4.annotate("More Chiral",
                 xy=(40, 0.1),
                 xytext=(69, 0.15),
                 color="k",
                 arrowprops={
                     "arrowstyle": "simple",
                     "color": "k"
                 },
                 fontsize=16,
                 ha='center',
                 va='center',
                 zorder=3)
    ax4.annotate("More Achiral",
                 xy=(55, -0.15),
                 xytext=(30, -0.24),
                 color="k",
                 arrowprops={
                     "arrowstyle": "simple",
                     "color": "k"
                 },
                 fontsize=16,
                 ha='center',
                 va='center',
                 zorder=3)

    ax4.set_xticklabels(["", "", "", "", "", ""])
    ax4.set_yticklabels([-0.6, -0.4, -0.2, 0, 0.2, 0.4])
    ax4.set_ylabel("Universality Difference")
    ax4.set_xlabel("Normalized rank-order")

    ax4.spines[['right', 'top']].set_visible(False)
    ax4.annotate("D", xy=(0, 0.21), color="k", weight="bold", annotation_clip=False)

    plt.tight_layout(pad=0.25)

    plt.savefig("../Figures/Fig4.svg", format="svg")
    plt.savefig("../Figures/Fig4.png", format="png")


def fig5(seed_name, rxn_db, crossover_step, first_chiral_step):
    ### Panel 5A - chiral rxn counts
    heatmap_array, rxn_df = get_heatmap_array(keggC_df)

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(heatmap_array)

    # ax.set_yticklabels(["", 0, "", 1, "", 2])

    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False)

    ax.set_ylabel("Substrate Chiral Compounds", fontsize=18)
    ax.set_xlabel("Product Chiral Compounds", fontsize=18)

    for i in range(5):
        for j in range(5):
            if i == 1 and j == 1:
                c = "k"
            else:
                c = "silver"
            text = ax.text(j,
                           i,
                           heatmap_array[i][j],
                           ha="center",
                           va="center",
                           color=c,
                           fontsize=14)

    # Add the color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Reaction Count")
    ax.annotate("A", xy=(-0.5, -0.55), color="k", weight="bold", annotation_clip=False)

    plt.savefig("../Figures/Fig5A.svg", format="svg")
    plt.savefig("../Figures/Fig5A.png", format="png")
    plt.close()

    ### Panel 5B - achiral/chiral modulating &conserving
    ### Get full seed set & reaction set
    seed_groups = [seed_name]
    reaction_scope = get_sides(import_reaction_file(rxn_db))

    compound_scope = set.union(*reaction_scope.stack())
    seed_scope = seed_candidates.loc[seed_candidates.index.isin(compound_scope)]
    seed_sets = {
        i: seed_scope.query('Source.str.contains("' + i + '")').index
        for i in seed_groups
    }
    print("- Seed Set -")

    ### Run network expansion on original seed set
    results = dict()
    for name, seeds in seed_sets.items():
        print(f"- Seed Set Size: {len(seeds)}")
        print(seeds)
        results[name] = expand_from_seed_set(seeds=seeds,
                                             reactions=reaction_scope,
                                             max_gen=max_generation)

    chiral_modulating, chiral_conserving, achiral_conserving = get_chiral_modulations(
        results[name]["participants"], rxn_df)
    total_rxns = chiral_modulating[-1] + chiral_conserving[-1] + achiral_conserving[-1]
    print(
        f"- Total Chiral Modulating Reactions: {chiral_modulating[-1]}, {round((chiral_modulating[-1] / total_rxns) * 100, 3)}% -")
    print(
        f"- Total Chiral Conserving Reactions: {chiral_conserving[-1]}, {round((chiral_conserving[-1] / total_rxns) * 100, 3)}% -")
    print(
        f"- Total Achiral Conserving Reactions: {achiral_conserving[-1]}, {round((achiral_conserving[-1] / total_rxns) * 100, 3)}% -")

    fig, ax = plt.subplots(figsize=(12, 12))

    ax.plot(chiral_modulating,
            linestyle="solid",
            linewidth=3,
            color='k',
            label="Chiral Modulating Rxns")
    ax.plot(chiral_conserving,
            linestyle="dashed",
            linewidth=3,
            color='k',
            label="Chiral Conserving Rxns")
    ax.plot(achiral_conserving,
            linestyle="dotted",
            linewidth=3,
            color='k',
            label="Achiral Conserving Rxns")

    ax.axvline(x=crossover_step, color="k", linestyle="--", alpha=0.9)
    ax.axvline(x=first_chiral_step, color="k", linestyle=":", alpha=0.5)
    ax.annotate(f"Achiral-to-Chiral\nCrossover\nStep {crossover_step}",
                xy=(crossover_step, 83000),
                xytext=(crossover_step + 8, 85000),
                color="k",
                arrowprops={
                    "arrowstyle": "simple",
                    "color": "k"
                },
                fontsize=14,
                ha='center',
                va='center',
                zorder=3)
    ax.annotate(f"First Chiral\nCompound\nStep {first_chiral_step}",
                xy=(first_chiral_step, 70000),
                xytext=(first_chiral_step + 10, 72000),
                color="k",
                arrowprops={
                    "arrowstyle": "simple",
                    "color": "k"
                },
                fontsize=14,
                ha='center',
                va='center',
                zorder=3)

    ax.legend(facecolor="white", framealpha=1, prop={"size": 12})
    ax.set_xlabel("Expansion Generation")
    ax.set_ylabel("Total Reactions")
    ax.annotate("B", xy=(-1.5, 94000), color="k", weight="bold", annotation_clip=False)

    plt.savefig("../Figures/Fig5B.svg", format="svg")
    plt.savefig("../Figures/Fig5B.png", format="png")
    plt.close()

    ### Get pathway names & cpd lists from KEGG (see slack for API calls)
    session = prepare_session()
    pathway_cpd_links, pathway_names = get_pathway_data(session=session)
    pathway_chiralPercentage = dict()
    pathway_size = dict()

    ### Get chirality of each pathway
    print("- Calculating chiral percentage of pathways -")
    for pathway in tqdm(pathway_cpd_links.keys()):
        pathway_chiralPercentage[pathway], pathway_size[
            pathway] = get_chiral_percentage_from_list(
            pathway_cpd_links[pathway], keggC_df)

    pathway_df = pd.DataFrame(
        [pathway_chiralPercentage, pathway_names,
         pathway_size]).transpose().dropna().sort_values(by=[0],
                                                         ascending=False)
    pathway_df = pathway_df.rename(columns={
        0: "Chirality",
        1: "Name",
        2: "Size"
    })

    pathway_df['ID'] = pathway_df.index
    # Get color setup & information
    print("- Getting color schema -")
    pathway_df, chiralities, pathway_group_colors, metabolism_groups, colors, color_list = get_pathway_colors(
        pathway_df)

    ### Port over colors & plot individual panels
    # Label with pathway number
    labels = pathway_df['ID'].tolist()

    # Define each angle for plotting
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = pathway_df['Chirality'].tolist()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.plot(angles, stats, 'o-', linewidth=2, color="dimgrey")
    ax.fill(angles, stats, alpha=0.25, color="dimgrey")
    ax.set_thetagrids(
        angles * 180 / np.pi,
        labels,
    )

    # Make all angles rotated according to placement on the image
    angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
    angles = np.rad2deg(angles)
    count = 0
    for label, angle in zip(ax.get_xticklabels(), angles):
        x, y = label.get_position()
        try:
            if count % 2:
                lab = ax.text(x,
                              y - 0.05,
                              label.get_text(),
                              transform=label.get_transform(),
                              ha=label.get_ha(),
                              va=label.get_va(),
                              color=str(pathway_group_colors[int(
                                  label.get_text())]),
                              fontsize=18)
            else:
                lab = ax.text(x,
                              y,
                              label.get_text(),
                              transform=label.get_transform(),
                              ha=label.get_ha(),
                              va=label.get_va(),
                              color='w',
                              fontsize=1)
            count += 1
        except:
            lab = ax.text(x,
                          y,
                          label.get_text(),
                          transform=label.get_transform(),
                          ha=label.get_ha(),
                          va=label.get_va(),
                          color='k',
                          fontsize=1)
        lab.set_rotation(angle)
        labels.append(lab)
    ax.set_xticklabels([])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", ""], fontsize=14)

    # label chiral percentage
    label_pos = ax.get_rlabel_position()
    ax.text(np.radians(label_pos + 10),
            ax.get_rmax() / 2. + 0.05,
            "$\mathbb{C}$",
            rotation=label_pos + 4,
            ha='center',
            va='center',
            fontsize=30,
            weight="bold")
    ### Figure label annotation
    ax.text(np.radians(label_pos + 120),
            ax.get_rmax() / 2. + 0.05,
            "C",
            rotation=0,  # Rotation of text (previous was label_pos - 10)
            ha='center',
            va='center',
            fontsize=22,
            weight="bold")

    plt.savefig("../Figures/Fig5C.svg", format="svg")
    plt.savefig("../Figures/Fig5C.png", format="png")
    plt.close()

    ### Boxplot of pathway chirality distribution per metabolic group
    fig, ax = plt.subplots(figsize=(12, 8))

    c = colors["Carbohydrate Metabolism"]
    bplot = ax.boxplot(
        chiralities,
        patch_artist=True,
        positions=np.array(range(len(chiralities))) * 1.0 - 0.4,
        widths=0.6,
        medianprops=dict(
            color="black"))  # color=colors["Carbohydrate Metabolism"])
    for patch, c in zip(bplot["boxes"], color_list):
        patch.set_color(c)

    ax.set_ylabel(r"Chiral Order\nParameter $\mathbb{C}$", fontsize=30)
    ax.xaxis.set_ticks([])

    # legend of label colors
    markers = [
        plt.Line2D([0, 0], [0, 0],
                   color=color,
                   marker='o',
                   linestyle='',
                   markersize=15) for color in colors.values()
    ]
    ax.legend(markers,
              colors.keys(),
              bbox_to_anchor=(1, 0.7),
              loc="upper left",
              numpoints=1,
              frameon=False,
              prop={"size": 14})
    ax.grid(True)
    ax.annotate("D", xy=(-1, 1.06), color="k", weight="bold", annotation_clip=False)
    plt.tight_layout()

    plt.savefig("../Figures/Fig5D.svg", format="svg")
    plt.savefig("../Figures/Fig5D.png", format="png")
    plt.close()

    ### Pathway size by chirality
    fig, ax = plt.subplots(figsize=(16, 8))

    pathway_df.sort_values(by="Group Key", inplace=True)
    ax.scatter(pathway_df['Size'],
               pathway_df['Chirality'],
               c=pathway_df['Group'].map(colors),
               s=50)

    ax.set_ylabel(r"Chiral Order\nParameter $\mathbb{C}$", fontsize=30)
    ax.set_xlabel("System Size (Unique Molecules Per Pathway)")
    loc = plticker.MultipleLocator(
        base=20.0)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.set_ylim(-.05, 1.05)
    ax.annotate("C", xy=(-6, 1.07), color="k", weight="bold", annotation_clip=False)

    plt.savefig("../Figures/Fig5E.svg", format="svg")
    plt.savefig("../Figures/Fig5E.png", format="png")
    plt.close()


def main():
    pd.set_option('future.no_silent_downcasting',
                  True)  # To avoid future warnings
    # Matplotlib defaults
    font = {"sans-serif": "DejaVu Sans", "size": 22}
    for key, value in font.items():
        plt.rcParams.update({"font." + key: value})

    # Seed set
    seed_name = "JM|G24|G17|TW"
    rxn_db_fp = "../Data/CBRdb_R.csv"

    print("---- Generating Figure 1 ----")
    fig1()

    print("---- Generating Figure 2 ----")
    # ~Five hours for full expansion (n_iter=100, min=2, max=10)
    min_ss_size = 2
    max_ss_size = 10
    n_iter = 100
    crossover_step, first_chiral_step, attributes, networks_uni = fig2(
        seed_name=seed_name,
        rxn_db=rxn_db_fp,
        min_ss_size=min_ss_size,
        max_ss_size=max_ss_size,
        n_iter=n_iter)

    print("---- Generating Figure 3 ----")
    fig3(seed_name=seed_name,
         rxn_db=rxn_db_fp,
         crossover_step=crossover_step,
         attributes=attributes,
         networks_uni=networks_uni)

    print("---- Generating Figure 4 ----")
    fig4()

    print("---- Generating Figure 5 ----")
    fig5(seed_name=seed_name,
         rxn_db=rxn_db_fp,
         crossover_step=crossover_step,
         first_chiral_step=first_chiral_step)


if __name__ == "__main__":
    main()
