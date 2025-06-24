import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from utilities import colorize_series
from netgraph import get_multipartite_layout

def generate_networks(exp):

    edgelists = {'all': exp['unipartite_edgelist'].drop_duplicates(subset=['source','target'])}
    edgelists.update({iname : exp['unipartite_edgelist'].query(iname).drop_duplicates(subset=['source','target'])
                    for iname in ['directed', 'contingent', 'immediate_precursor']})
    networks = {iname: nx.from_pandas_edgelist(iedges, create_using=nx.DiGraph, edge_attr=True)
                for iname, iedges in edgelists.items()}
    exp.update({'networks': networks, 'edgelists': edgelists})

    return None


def annotate_networks(exp):

    chir_vals = {False: 'Achiral', True: 'Chiral'}
    chir_hues = {'Achiral': '#FFC012', 'Chiral': '#34CCFF', 'LUCA': 'indigo', 
                'Metabolism': 'k', 'Chemical Space': 'gray', 'Phase Boundary': 'silver'}
    chiral_color = exp['scope_chirality'].map(chir_vals).map(chir_hues)

    def _degree_by_chirality(inet):
        adj = nx.to_pandas_adjacency(inet, nonedge=float('nan'))
        out_degs = adj.groupby(exp['scope_chirality'].map(chir_vals)).count()
        in_degs = adj.T.groupby(exp['scope_chirality'].map(chir_vals)).count()
        both_degs = out_degs + in_degs
        opp_deg = pd.concat([both_degs.loc['Achiral', exp['scope_chirality']],
                            both_degs.loc['Chiral', ~exp['scope_chirality']]]).to_frame(name='opposite degree').T
        deg = pd.concat([out_degs.add_suffix(' out-degree', axis=0), in_degs.add_suffix(' in-degree', axis=0), both_degs.add_suffix(' degree', axis=0), opp_deg]).T
        deg.columns = deg.columns.str.lower()
        return deg
    
    networks = exp['networks']
    node_attrs = dict()
    sort_attrs = dict()
    for iname, inet in networks.items():

        attrs = _degree_by_chirality(inet).assign(
            degree=dict(inet.to_undirected().degree()), 
            is_chiral_color=chiral_color,
            is_chiral=exp['scope_chirality'],
            chiral_sign=exp['scope_chirality'].map({True:1, False:-1}),
            arrival_step=exp['arrival_step'])
        attrs['out-degree'] = dict(inet.out_degree())
        nx.set_node_attributes(inet, attrs.to_dict(orient='index'))
        
        sortable = attrs.copy(deep=True)
        sortable.update(attrs.filter(like='degree').mul(attrs['chiral_sign'], axis=0))

        node_attrs[iname] = attrs
        sort_attrs[iname] = sortable

    exp.update({'node_attrs': node_attrs, 'sort_attrs': sort_attrs})

    return None


def assign_gen_colors(exp, max_colorized_gen=6):

    entries_to_colorize = exp['arrival_step'].mask(lambda x: x > max_colorized_gen)
    gen_color = colorize_series(entries_to_colorize, cmap='Greys').set_axis(entries_to_colorize.index)

    for iname, inet in exp['networks'].items():
        nx.set_node_attributes(inet, gen_color.to_dict(), 'gen_color')
        exp['node_attrs'][iname]['gen_color'] = gen_color
        exp['sort_attrs'][iname]['gen_color'] = gen_color
    
    return None


def gens_as_layers(exp, to_show='immediate_precursor', sort_by=None, ascending=False):

    inet = exp['networks'][to_show]
    to_sort = exp['sort_attrs'][to_show].copy(deep=True)
    if sort_by is not None:
        to_sort.sort_values(by=sort_by, ascending=ascending, inplace=True)
    layers = pd.Series(to_sort.groupby(by='arrival_step').groups).map(list).to_list()

    return layers


def position_layers(exp, to_show, layers, max_layer=1000):
    show_layers = layers[:max_layer+1]
    show_edges = exp['edgelists'][to_show].query('step <= @max_layer')
    show_edges = [tuple(i) for i in show_edges[['source', 'target']].values]
    pos = get_multipartite_layout(edges=show_edges, layers=show_layers, reduce_edge_crossings=False)
    return pos


def spread_position_layers(pos, layers, max_layer=1000):

    df = pd.Series(layers[:max_layer+1]).rename_axis('layer').reset_index(name='entries')
    df['len'] = df['entries'].map(len)
    df['running'] = df['len'].rolling(window=3, center=True).mean().fillna(0) ** 0.4
    df['scaled'] = ((df['running'].cumsum() / df['running'].sum()) * 0.9) + 0.05

    pd_pos = pd.DataFrame(pos).T
    pos_map = dict(zip(pd_pos.loc[:,0].drop_duplicates().to_list(), df['scaled'].to_list()))
    pd_pos[0] = pd_pos[0].map(pos_map)
    pos_new = {i: j.values for i, j in pd_pos.T.items()}

    return pos_new


def subgraph_at_gen(exp, to_show='immediate_precursor', max_gen=1000):
    show_edges = exp['edgelists'][to_show].query('step <= @max_gen')
    show_edges = [tuple(i) for i in show_edges[['source', 'target']].values]
    G = exp['networks'][to_show].edge_subgraph(show_edges).copy()
    return G


def apply_spread_minimum(pos, frac_threshold=0.025):
    positions = pd.DataFrame(pos).T
    xvals = positions[0].drop_duplicates().reset_index(drop=True)
    newxs = xvals.copy(deep=True)
    for gen in newxs.index[1:]:
        diff = float(newxs.loc[gen] - newxs.loc[gen-1])
        if diff < frac_threshold:
            newxs.update((newxs.loc[gen:] - diff) + frac_threshold)
    newxs = newxs.transform(lambda x: (x-x.min())/(x.max()-x.min()) )#* 0.95 + 0.025)
    xval2layer = xvals.reset_index().set_index(0)['index']
    positions[0] = positions[0].map(xval2layer).map(newxs)
    pos_new = {i: j.values for i,j in positions.T.items()}
    return pos_new


def layered_plot(exp, to_show, sort_by, ascending, max_layer, frac_threshold=0.025, fig=None, ax=None):

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    layers = gens_as_layers(exp, to_show=to_show, sort_by=sort_by, ascending=ascending)
    pos = position_layers(exp, to_show=to_show, layers=layers, max_layer=max_layer)
    pos_new = spread_position_layers(pos=pos, layers=layers, max_layer=max_layer)
    pos_new = apply_spread_minimum(pos_new, frac_threshold=frac_threshold)
    G = subgraph_at_gen(exp, to_show=to_show, max_gen=max_layer)
    pos_colors = pd.DataFrame(G.nodes(data='is_chiral_color'))[1].to_list()

    ax.axis("off")
    nx.draw_networkx(G, pos=pos_new, ax=ax, edge_color=[0.4,0.4,0.4,0.1], 
                        node_color=pos_colors,
                        width=0.2, arrows=False,
                        linewidths=0.5,  node_size=30, node_shape='_',
                        with_labels=False)
    
    annot_locs = pd.DataFrame(pos_new).T.sort_values(by=[0,1], ascending=[True, False]).drop_duplicates(subset=0).reset_index(drop=True)
    annot_locs = annot_locs.T.to_dict(orient='list')
    for igen, ipos in annot_locs.items():
        if igen < len(annot_locs.keys()):
            ax.annotate(str(igen), xy=ipos, ha='center', va='bottom', c='k', fontsize=plt.rcParams['font.size']*0.8)
    plt.margins(0,0)

    return G, fig
