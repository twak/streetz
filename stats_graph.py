
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import utils
from utils import l2, l2p

from math import floor

import random

GRAPH = None

def build_graph(vertices, edges):
    global GRAPH

    if GRAPH == None:

        GRAPH = nx.Graph()

        for e in edges:
            GRAPH.add_edge(e[0], e[1], length=l2 (e, vertices) )

    return GRAPH


def reset_graph_cache():
    global GRAPH, TIMES_VISITED
    GRAPH = None
    TIMES_VISITED = None


def transport_ratio( vertices, edges, table_data, table_row_names, minn=1, maxx=3, bins = 32, norm = True ):

    global TIMES_VISITED

    out = np.zeros((bins), dtype=np.int)
    g = build_graph(vertices, edges)

    count = 0
    total = 0.

    TIMES_VISITED = np.zeros((len(vertices)), dtype=np.int)

    def compute (a,b):

        nonlocal count, total
        global TIMES_VISITED

        if a != b:
            path_length = 0

            try:
                path_steps = nx.shortest_path(g, a, b, weight='length')
                for pi in range(len(path_steps) - 1):
                    path_length += l2p(path_steps[pi], path_steps[pi + 1], vertices)
                    TIMES_VISITED[path_steps[pi]] = TIMES_VISITED[path_steps[pi]] + 1

            except nx.exception.NetworkXNoPath:
                return

            crow_flies = l2p(a, b, vertices)

            if crow_flies == 0:
                return

            transport_ratio = path_length / crow_flies

            idx = floor( ( transport_ratio -minn) * bins / (maxx - minn))
            idx = min(idx, bins - 1)
            out[idx] += 1

            count += 1
            total += transport_ratio

    if False: # compute for all ~1hr
        for a in range(len(vertices)):
            print(f'{a}/{len(vertices)}')
            for b in range(len(vertices)):
                print(f'  {b}/{len(vertices)} transport ratio: {total/count}')
                compute (a,b)
        table_row_names.append(f"Transport ratio")
    else:
        samples = 300
        for i in range(samples):
            print(f'{i}/{samples}: transport ratio: {total/max(1,count)}')
            a = random.randint(0, len(vertices))
            b = random.randint(0, len(vertices))
            compute(a,b)

        table_row_names.append(f"Transport ratio (n={samples})")

    table_data.append( ("%.2f" % (total/count) ))

    if norm:
        out = out / float ( count )

    # FastPlot(2048, 2048, vertices, edges, scale=0.1, edge_cols = np.array(edge_cols) ).run()

    return out

def plot_transport_ratio(all_city_stat, name, fig, subplots, subplot_idx, minn=1, maxx=3, bins = 32 ):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text(name)
    axs.spines['top'].set_color('lightgray')
    axs.spines['right'].set_color('lightgray')

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = list ( map (lambda x : " %.2f" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Transport ratio")

        utils.plot(r, plt, bins, idx, all_city_stat)

        x_pos = x_pos[::2]
        x_lab = x_lab[::2]

        x_lab[len(x_lab)-1] = "> %.2f" % maxx
        x_pos[len(x_pos)-1] = bins -1

        plt.xticks(x_pos, x_lab)

def betweenness_centrality( vertices, edges, table_data, table_row_names, minn=0, maxx=1, bins = 32 ):

    global TIMES_VISITED

    out = np.zeros((bins), dtype=np.int)

    if TIMES_VISITED == None:
        return out

    most_times_visited =TIMES_VISITED.max()
    TIMES_VISITED = TIMES_VISITED.astype(np.float) / most_times_visited

    total = 0.

    for v in TIMES_VISITED:

        idx = floor ( v * bins / (maxx - minn) )
        idx = min(idx, bins-1)
        out[idx] = out[idx] + 1
        total += v



    table_data.append( "%.2f" % ( total / len (edges) ) ) # mean edges at a vertex)
    table_row_names.append("Mean edge length (m)")

    return out


def plot_betweenness_centrality(all_city_stat, name, fig, subplots, subplot_idx, minn=1, maxx=3, bins = 32 ):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text(name)
    axs.spines['top'].set_color('lightgray')
    axs.spines['right'].set_color('lightgray')

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = list ( map (lambda x : " %.2f" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Transport ratio")

        utils.plot(r, plt, bins, idx, all_city_stat)

        x_pos = x_pos[::2]
        x_lab = x_lab[::2]

        x_lab[len(x_lab)-1] = "> %.2f" % maxx
        x_pos[len(x_pos)-1] = bins -1

        plt.xticks(x_pos, x_lab)