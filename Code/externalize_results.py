import os
import pandas as pd
import matplotlib.pyplot as plt

# purpose: store raw and processed output of network expansions.

def write_output_files(results: dict):
    for name, info in results.items():
        if not os.path.exists(f'Output/{name}'):
            os.mkdir(f'Output/{name}')
        for infotype, df in info.items():
            if type(df) == pd.DataFrame:
                output_df = df.copy(deep=True).fillna('')
                if infotype in ['participants', 'edge_pool']:
                    output_df = make_lists_printable(output_df)
                output_df.to_csv(f'Output/{name}/{infotype}.csv.zip', compression='zip', encoding='utf-8')
            elif type(df) in [list, set]:
                with open(f'Output/{name}/{infotype}.txt', 'w') as f:
                    f.write(' '.join(df))
            elif type(df) == pd.Series:
                df.to_csv(f'Output/{name}/{infotype}.csv.zip', compression='zip', encoding='utf-8')
    output_folders = set(os.listdir('Output')).intersection(results.keys())
    print('\n\nResult folder(s) in Output:\n', '\n'.join(output_folders), sep='')
    return None


# purpose: convert list-like entries in a dataframe to strings for easy read/write

def make_lists_printable(df):
    list_like_cols = df.loc[:, df.iloc[0].apply(type).apply(lambda x: x in [set, list])]
    printable_cols = list_like_cols.applymap(' '.join)
    printable_df = df.copy(deep=True)
    printable_df.loc[:, printable_cols.columns] = printable_cols
    return (printable_df)


# purpose: output only info necessary to replicate expansion: seed scope, reaction scope, and expansion record

def write_edgelist_csvs(results):
    for i in results.keys():
        expansion_scope =  '#'+'|'.join(results[i]['seeds'])+'|'+'|'.join(results[i]['reaction_pool'])
        open('Output/'+i+'_bipartite_edgelist.csv','w').write(expansion_scope+'\n')
        results[i]['bipartite_edgelist'].to_csv('Output/'+i+'_bipartite_edgelist.csv', sep=',', index=True, mode='a')
    return([i for i in os.listdir('Output/') if i.endswith('.csv')])


# purpose: plot summary of expansion results

def plot_expansion_summary(summaries, filename='expansion_comparison_plot.pdf'):
    order = summaries['compound count'].iloc[-1].sort_values(ascending=False).index
    fig, ax = plt.subplots(len(summaries.keys()), 1, figsize=(4, 7), sharex=True, layout='compressed')
    for ict, iname in enumerate(summaries.keys()):
        summaries[iname].loc[:, order].plot(ax=ax[ict], ylabel=iname, legend=False, grid=True, xlabel='expansion step', cmap='coolwarm')
    ax[2].legend(loc='center', bbox_to_anchor=(0.5, -0.5), title='', ncols=2)
    ax[2].axhline(0.5, color='gray', linestyle='dotted', lw=0.7)
    plt.savefig('Output/'+filename, dpi=300, edgecolor='w', facecolor='w')
    return(None)

