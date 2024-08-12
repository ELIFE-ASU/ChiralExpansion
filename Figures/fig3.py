import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def expand_children(df, level):
    df['class_' + str(level)] = df['children'].apply(lambda x: x['name'])
    df['children'] = df['children'].apply(lambda x: x['children'])
    return df.explode('children')


def categorize_chirality(n):
    if n == 0:
        return False
    elif n > 0:
        return True
    else:
        return float('NaN')


heavy_percent_labs = ['0 - 20th', '21 - 40th', '41 - 60th', '61 - 80th', '81 - 100th']

cpd_appearances = pd.read_csv("../Data/compound_appearances.csv",
                              index_col=0,
                              usecols=['Compound', 'Timestep_Appear']).rename(
    columns={'Timestep_Appear': 'Expansion Step'})
cpd_info = pd.read_csv("../Data/kegg_data_C_detailed.csv.zip", index_col=0).set_index('CID')
cpd_appearances['Heavy Atoms'] = cpd_appearances.index.map(cpd_info['N Heavy'])
cpd_appearances['Chiral Centers'] = cpd_appearances.index.map(cpd_info['n_chiral_centers'])
percentiles = cpd_appearances['Heavy Atoms'].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cpd_appearances['Heavy Atoms Percentile'] = pd.cut(cpd_appearances['Heavy Atoms'],
                                                   bins=percentiles,
                                                   labels=heavy_percent_labs)

brite = json.load(open("../Data/br08001.json"))  # NEED THE UPDATED VERSION OF THIS FILE.
brite = pd.DataFrame(brite)['children'].apply(pd.Series).set_index('name').explode('children')
brite = expand_children(expand_children(brite, 1), 2).reset_index(drop=False, names='class_0').sort_index(axis=1)
brite[['id', 'name']] = brite['children'].apply(lambda x: pd.Series(x['name'].split('  ')))
brite.drop_duplicates(subset=['id', 'class_0'], inplace=True)
brite.set_index('id', inplace=True)  # cpd_class is column class_0 --> labels; classes=list(df['class_0'].unique())

# Find compounds with classifications under BRITE that appear in the expansion
brite_cpd_appearances = brite.join(cpd_appearances, how='inner').rename(columns={'class_0': 'Labels'}).reset_index()
brite_cpd_appearances['Heavy Atoms'] = brite_cpd_appearances['id'].map(cpd_info['N Heavy'])
brite_cpd_appearances['Chiral Centers'] = brite_cpd_appearances['id'].map(cpd_info['n_chiral_centers'])
brite_cpd_appearances['is_chiral'] = brite_cpd_appearances['Chiral Centers'].apply(categorize_chirality)
# Find chirality of each label. (prev: "Make sub-dataframe for each class")
# Results differ from before, because this accommodates multiple labels per cpd.
print(brite_cpd_appearances.groupby('Labels')['is_chiral'].mean(), flush=True)

font = {"size": 22}
for key, value in font.items():
    plt.rcParams.update({"font." + key: value})
g = sns.jointplot(brite_cpd_appearances,
                  x='Expansion Step',
                  y='Chiral Centers',
                  hue='Labels',
                  palette="colorblind",
                  ylim=(-1, 31),
                  xlim=(0, 40), s=100)
g.figure.set_size_inches((16, 8))
g.ax_joint.axvline(x=14,
                   color="k",
                   alpha=0.5,
                   linestyle="--",
                   label="Achiral-to-Chiral Crossover\n(GEN 14)")
g.ax_joint.legend(bbox_to_anchor=(0.32, 0.45), prop={"size": 12})
plt.savefig("fig3.svg", format="svg")
plt.show()

# Plotting all cpds, (eventually) coloring by num heavy atoms
df = cpd_appearances
average_heavy_atoms = df.groupby(by=['Expansion Step', 'Chiral Centers'])['Heavy Atoms'].mean().reset_index()
average_heavy_atoms['Heavy Atoms Percentile'] = pd.cut(average_heavy_atoms['Heavy Atoms'],
                                                       bins=percentiles,
                                                       labels=heavy_percent_labs)
sns.set_palette(sns.color_palette("copper_r"))
g = sns.jointplot(data=average_heavy_atoms.sort_values(by='Heavy Atoms', ascending=False),
                  # CHANGES BASED ON SORTING. do we want to take (relative) percentile value of the mean number of heavy atoms, of a given stack of dots?
                  x='Expansion Step',
                  y='Chiral Centers',
                  hue='Heavy Atoms Percentile',
                  palette="copper_r",
                  ylim=(-1, 31),
                  xlim=(0, 40),
                  s=100,
                  marginal_kws={'common_norm': False})

"""
# Doesn't calculate average and plot percentile of that average.
g = sns.jointplot(data=df.sort_values(by='Heavy Atoms', ascending=False), 
    #CHANGES BASED ON SORTING. do we want to take (relative) percentile value of the mean number of heavy atoms, of a given stack of dots?
    x='Expansion Step', y='Chiral Centers', hue='Heavy Atoms Percentile', 
    palette="copper_r", ylim=(-1,31), xlim=(0,40), s=100, marginal_kws={'common_norm':False})

"""
h = sns.regplot(ax=g.ax_joint, data=df, x="Expansion Step", y="Chiral Centers", scatter=False, color="k")
m, b = np.polyfit(df["Expansion Step"], df["Chiral Centers"], 1)
g.ax_joint.legend(bbox_to_anchor=(0.32, 0.49), prop={"size": 12})
g.ax_joint.axvline(x=14, color="k", alpha=0.5, linestyle="--", label="Achiral-to-Chiral Crossover\n(GEN 14)")
g.ax_joint.legend(bbox_to_anchor=(0.32, 0.45), prop={"size": 12}, title='Heavy Atoms\nPercentile')
g.figure.set_size_inches((16, 8))
plt.savefig("fig3b.svg", format="svg")
plt.show()

# Plotting all cpds, (eventually) coloring by num heavy atoms
df = cpd_appearances

average_heavy_atoms = df.groupby(by=['Expansion Step', 'Chiral Centers'])['Heavy Atoms'].mean().reset_index()
average_heavy_atoms['Heavy Atoms Percentile'] = pd.cut(average_heavy_atoms['Heavy Atoms'],
                                                       bins=percentiles,
                                                       labels=heavy_percent_labs)
sns.set_palette(sns.color_palette("copper_r"))

# Doesn't calculate average and plot percentile of that average.
g = sns.jointplot(data=df.sort_values(by='Heavy Atoms', ascending=False),
                  # CHANGES BASED ON SORTING. do we want to take (relative) percentile value of the mean number of heavy atoms, of a given stack of dots?
                  x='Expansion Step',
                  y='Chiral Centers',
                  hue='Heavy Atoms Percentile',
                  palette="copper_r",
                  ylim=(-1, 31),
                  xlim=(0, 40),
                  s=100,
                  marginal_kws={'common_norm': False})

h = sns.regplot(ax=g.ax_joint, data=df, x="Expansion Step", y="Chiral Centers", scatter=False, color="k")
m, b = np.polyfit(df["Expansion Step"], df["Chiral Centers"], 1)
g.ax_joint.legend(bbox_to_anchor=(0.32, 0.49), prop={"size": 12})
g.ax_joint.axvline(x=14, color="k", alpha=0.5, linestyle="--", label="Achiral-to-Chiral Crossover\n(GEN 14)")
g.ax_joint.legend(bbox_to_anchor=(0.32, 0.45), prop={"size": 12}, title='Heavy Atoms\nPercentile')
g.figure.set_size_inches((16, 8))
plt.savefig("fig3b.svg", format="svg")
plt.show()

degree_df = pd.read_csv("../Data/undirected_cp_cp_degrees_from_reaction_edges.csv")
df = df.merge(degree_df, left_index=True, right_on=["Cpd"])
color_map = {'Chiral': "#34CCFF", 'Achiral': "#FFC012", 'Unknown': "#800080"}
df['is_chiral'] = df['Chiral Centers'].apply(categorize_chirality).map({True: 'Chiral', False: 'Achiral'}).fillna(
    'Unknown')
df['Degree/10'] = df['Degree'] / 10

bins = np.arange(-0.5, 11.5, 1)
df['is_chiral_bool'] = df['Chiral Centers'] > 0

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(nrows=2,
                      ncols=2,
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
df.plot.scatter(ax=ax,
                alpha=1.0,
                x='Degree',
                y='Chiral Centers',
                c=df['is_chiral'].map(color_map),
                s='Degree/10',
                logx=True)
df.query('is_chiral_bool')['Chiral Centers'].plot.hist(ax=ax_histy,
                                                       orientation='horizontal',
                                                       bins=bins,
                                                       edgecolor="k",
                                                       facecolor=color_map['Chiral'])
df[~df['is_chiral_bool']]['Chiral Centers'].plot.hist(ax=ax_histy,
                                                      orientation='horizontal',
                                                      bins=bins,
                                                      edgecolor="k",
                                                      facecolor=color_map['Achiral'])
ax_histy.set_xlabel("Count", fontsize=14)
ax_histy.tick_params(axis='both', which='major', labelsize=14)
ax.set_yticklabels(["", "0", "5", "10"])
ax.set_xscale("log")
ax.set_xlabel("Degree")
ax.set_ylabel("Chiral Centers")
plt.savefig("fig3c.png", format="png")
plt.show()

# END OF PART I
