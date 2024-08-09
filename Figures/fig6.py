import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import matplotlib.ticker as plticker
from matplotlib.gridspec import GridSpec
import os

font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 22}
plt.rcParams.update({'font.size': 22})

colors = {"Carbohydrate Metabolism": "#aa0f35", 
          "Glycan Biosynthesis": "#487c7c", 
          "Terpenoids and Polyketides": "#009ecb", 
          "Lipid Metabolism": "#abdbfc", 
          "Secondary Metabolite Biosynthesis": "#430057", 
          "Nucleotide Metabolism": "#a0a75c", 
          "Amino Acid Metabolism": "#603734",
          "Metabolism of Other Amino Acids": "#d97e43", 
          "Cofactors and Vitamins": "#313a61", 
          "Energy Metabolism": "#fdc799",
          "Xenobiotics": "#f5c500"}
pathway_group_keys = dict(zip(list(colors.keys()), range(1,len(colors.keys())+1)))

def set_box_color(bp, color):
  plt.setp(bp['boxes'], color=color)
  plt.setp(bp['whiskers'], color=color)
  plt.setp(bp['caps'], color=color)
  plt.setp(bp['medians'], color=color)

#need to port over file for this calculation + accompanying metadata, and correct typo in the name.
df = pd.read_csv("../Data/pathway_appearances.csv").set_index('Pathway')
pathway_names =json.load(open("../Data/pathway_names.json",'r'))

pw_df = pd.DataFrame(pathway_names)['children'].apply(pd.Series).set_index('name').explode('children')
pw_df['met_class'] = pw_df['children'].apply(lambda x: x['name'])
pw_df['subclass'] = pw_df['children'].apply(lambda x: [i['name'] for i in x['children']])
pw_df = pw_df.explode('subclass').reset_index().drop(columns='children')
pw_df[['id','subclass']] = pw_df['subclass'].str.split('  ', expand=True)
pw_df.set_index(pw_df['id'].astype(int), inplace=True)
df = pw_df.join(df, how='inner').sort_values(by='Chirality')
# df.index.difference(pw_df.index) #https://bio2rdf.org/describe/?uri=http://bio2rdf.org/kegg:map01130&inf=skos-trans&sas=yes&sid=2996
# [231, 1130] no longer present in kegg... Puromycin biosynthesis [kegg:ko00231] and biosynthesis of antibiotics map01130
# 970 says it's Genetic Information Processing; Translation, not metabolism.  pw_df.loc[970] 
to_replace = {'Metabolism of Terpenoids And Polyketides': 'Terpenoids and Polyketides',
             'Metabolism of Cofactors And Vitamins': 'Cofactors and Vitamins',
             'Biosynthesis of Other Secondary Metabolites': 'Secondary Metabolite Biosynthesis',
             'Xenobiotics Biodegradation And Metabolism': 'Xenobiotics', 
             'Glycan Biosynthesis And Metabolism': 'Glycan Biosynthesis'}
df['Metabolic Group'] = df['met_class'].apply(lambda x: x.title().replace('Of ','of ')).replace(to_replace)
df['color'] = df['Metabolic Group'].map(colors).fillna('k')
df['group_key'] = df['Metabolic Group'].map(pathway_group_keys)
df['angle'] = np.linspace(0, 2*np.pi, len(df.index), endpoint=False) #Define each angle for plotting
df['text_angle'] = df['angle'].apply(lambda x: x if np.cos(x)>0 else x+np.pi).apply(np.rad2deg) #define text rotation angle

metabolic_group_members = df.groupby(by='Metabolic Group').groups

gs = GridSpec(10,2)
fig = plt.figure(figsize=(20,10), layout='tight')

# polar plot of chiral fractions
ax = fig.add_subplot(gs[0:6,0], polar=True, projection='polar')
plt.yticks([0.2,0.4,0.6,0.8], fontsize=14)
xticks = plt.xticks(ticks=df['angle'], labels='  '+df['id']+'  ', size=8)
for label, angle in zip(ax.get_xticklabels(), df['text_angle']):
  x,y = label.get_position() 
  lab = ax.text(x, y, label.get_text(), transform=label.get_transform(), ha=label.get_ha(), va=label.get_va(), 
                color=df.loc[int(label.get_text()),'color'], fontsize=8)
  lab.set_rotation(angle)
ax.set_xticklabels([])
ax.plot(df['angle'], df['Chirality'], 'o-', linewidth=2, color="dimgrey")
ax.fill(df['angle'], df['Chirality'], alpha=0.25, color="dimgrey")
ax.yaxis.set_tick_params(rotation='auto')
label_pos = ax.get_rlabel_position()
ax.text(x=np.radians(label_pos+10), y=ax.get_rmax()/2. + 0.05, s="$\mathbb{C}$",
        rotation=None, ha='center', va='center', fontsize=30, weight="bold") #        rotation=label_pos+4, 

#boxplot of chiral fraction distribution
ax2 = fig.add_subplot(gs[0:6,1])
sns.boxplot(data=df.query('`Metabolic Group`.isin(@colors.keys())'),
            x='Metabolic Group', y='Chirality', hue='Metabolic Group', 
            ax=ax2, order=colors.keys(), palette=colors)
ax2.set_ylabel("$\mathbb{C}$", fontsize=30)
ax2.set_xlabel('')
ax2.set_xticks([])
ax2.grid(True)

#legend of label colors
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', markersize=15) for color in colors.values()]
ax2.legend(markers, colors.keys(), bbox_to_anchor=(1, 0.7), loc="upper left", numpoints=1, frameon=False, prop={"size":14})
ax.grid(True)

### Pathway size by chirality
ax3 = fig.add_subplot(gs[6:,:])

df.dropna(subset=['group_key']).plot.scatter(x='Size', y='Chirality', c='color', ax=ax3)
ax3.set_ylabel("$\mathbb{C}$", fontsize=30)
ax3.set_xlabel("System Size (Compounds Per Pathway)")

loc = plticker.MultipleLocator(base=20.0) # this locator puts ticks at regular intervals
ax3.xaxis.set_major_locator(loc)
ax3.set_ylim(-.25, 1.3)
plt.savefig("fig6.svg", format="svg")
plt.show()

"""
df.sort_values(by='group_key', inplace=True)
current_groups = pd.Series(metabolic_group_members).groups).apply(list) #pd.Series(metabolism_groups)
# metabolism_groups_2 - see fig6.ipynb in original code for figures directory.
# {541, 542} and 470 in old not new... # 
metabolism_groups_2.apply(set) - current_groups.apply(set)
# 72, 471, 472, 473 in new not old... 
current_groups.apply(set) - metabolism_groups_2.apply(set)
# why is there a difference?
"""

