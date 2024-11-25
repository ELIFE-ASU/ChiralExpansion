import os

import matplotlib.pyplot as plt


def make_lists_printable(df):
    """
    Convert list-like entries in a dataframe to strings for easy read/write.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing list-like entries.

    Returns:
    pandas.DataFrame: A dataframe with list-like entries converted to strings.
    """
    list_like_cols = df.loc[:, df.iloc[0].apply(type).apply(lambda x: x in [set, list])]
    printable_cols = list_like_cols.applymap(' '.join)
    printable_df = df.copy(deep=True)
    printable_df.loc[:, printable_cols.columns] = printable_cols
    return printable_df


def write_expansion_csv(result, filename, destination_dir='Output/'):
    """
    Output only info necessary to replicate expansion: seed scope, reaction scope, and expansion record.

    Parameters:
    result (dict): A dictionary containing the expansion results, including 'seeds' and 'edge_pool'.
    filename (str): The name of the file to write the expansion data to.
    destination_dir (str, optional): The directory where the output file will be saved. Defaults to 'Output/'.

    Returns:
    list: A list of CSV filenames in the destination directory.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    if not destination_dir.endswith('/'):
        destination_dir += '/'
    expansion_scope = '#' + '|'.join(result['seeds'])
    open(destination_dir + filename + '_expansion.csv', 'w').write(expansion_scope + '\n')
    make_lists_printable(result['edge_pool']).to_csv(destination_dir + filename + '_expansion.csv', sep=',', index=True,
                                                     mode='a')
    return [i for i in os.listdir(destination_dir) if i.endswith('.csv')]


def plot_expansion_summary(summaries, filename='expansion_comparison_plot.pdf'):
    """
    Plot a summary of expansion results.

    Parameters:
    summaries (dict): A dictionary containing summary dataframes for different metrics.
    filename (str, optional): The name of the file to save the plot to. Defaults to 'expansion_comparison_plot.pdf'.

    Returns:
    None
    """
    order = summaries['compound count'].iloc[-1].sort_values(ascending=False).index
    fig, ax = plt.subplots(len(summaries.keys()), 1, figsize=(4, 2.2 * len(summaries.keys())), sharex=True,
                           layout='compressed')
    for ict, iname in enumerate(summaries.keys()):
        summaries[iname].loc[:, order].plot(ax=ax[ict], ylabel=iname, legend=False, grid=True, xlim=0, ylim=0,
                                            xlabel='expansion step')
        if iname == 'chiral fraction': ax[ict].axhline(0.5, color='gray', linestyle='dotted', lw=0.7)
    ax[-1].legend(loc='center', bbox_to_anchor=(0.5, -0.5), title='', ncols=3)
    plt.savefig('Output/' + filename, dpi=300, edgecolor='w', facecolor='w')
    return None
