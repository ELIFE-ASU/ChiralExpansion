import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt

font = {"size" : 18}
for key, value in font.items():
    plt.rcParams.update({"font." + key : value})
color_number = {"LUCA":"#10D410", "Archaea":"#D65E00", "Bacteria":"#0B77B5", 
                "Eukarya":"#E69F03", "Metagenome":"#009E73", "Biosphere": "#4B0092"} 


# PART 1






# PART 2

data_folder = '/Users/csm/ASU Dropbox/Elife/Manuscripts/In Progress/chirality_John/Figures/Data'
realm = ['archaea', 'bacteria', 'eukarya', 'metagenome']
files = [f'{data_folder}/{i}_occurance.csv' for i in realm]
nums = [1194, 10434, 267, 6112]
occs = pd.DataFrame({i: pd.read_csv(j).drop_duplicates().set_index('Cpd').iloc[:,0] for i,j in zip(realm,files)})
univ = occs/nums

chir = pd.read_csv('../Data/kegg_data_C_detailed.csv.zip').set_index('CID')[['n_chiral_centers']]
chir_cpds_found = chir.query('n_chiral_centers>0').index.intersection(occs.index)
achir_cpds_found = chir.query('n_chiral_centers==0').index.intersection(occs.index)

fig, [ax, ax2] = plt.subplots(1,2, figsize=(16,8))
rop_c, rop_a = dict(), dict()
for i in univ.loc[chir_cpds_found].items():
    k = i[1].dropna().sort_values(ascending=False).reset_index()
    k.index = k.index/len(k.index)
    k.plot(ax=ax, xlim=(0,1), grid=True, label=i[0], lw=3,
        c=color_number[i[0].title()], ls='dashed')
    rop_c[i[0]] = k
for i in univ.loc[achir_cpds_found].items():
    k = i[1].dropna().sort_values(ascending=False).reset_index()
    k.index = k.index/len(k.index)
    k.plot(ax=ax, xlim=(0,1), grid=True, label=i[0], lw=3,
        c=color_number[i[0].title()], ls='dotted',
        ylabel='Universality', xlabel='Normalized rank-order') #or percentile?
    rop_a[i[0]] = k

handles = []
chiral_patch = plt.Line2D([], [], color="k", linestyle="--", linewidth=5, label='Chiral')
achiral_patch = plt.Line2D([], [], color="k", linestyle=":", linewidth=5, label='Achiral')
handles.extend([chiral_patch, achiral_patch])
ax.legend(bbox_to_anchor=(1.4, 1.11), handles = handles, ncol = 2, prop={"size": 20})
ax.spines[['right', 'top']].set_visible(False)

diffs = pd.DataFrame()
xlabel, ylabel = "Normalized rank-order", "Chiral - Achiral Universality\nDifference"
for k in rop_a.keys():
    diffs[k] = (rop_c[k][k].quantile(np.linspace(0,1,100)) - 
             rop_a[k][k].quantile(np.linspace(0,1,100)))
diffs.columns = [i.title() for i in diffs.columns]
[diffs[i].plot(ax=ax2, c=color_number[i], grid=True, lw=3,
               label=i, xlabel=xlabel, ylabel=ylabel) for i in diffs]
ax2.legend(bbox_to_anchor=(0.7, 1.2), prop={"size":20}, ncol=4)
ax2.set_xticklabels([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels([-0.6, -0.4, -0.2, 0, 0.2, 0.4])
ax2.spines[['right', 'top']].set_visible(False)

plt.show()
