
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import utils
from utils import l2, l2p
from stats_segs import build_V2E

from math import floor

import random

GRAPH = None
SAMPLES = 300

def build_graph(vertices, edges):
    global GRAPH

    if GRAPH == None:

        GRAPH = nx.Graph()

        for e in edges:
            GRAPH.add_edge(e[0], e[1], length=l2(e, vertices))
            GRAPH.add_edge(e[1], e[0], length=l2(e, vertices))

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
    total_steps = 0
    total_length = 0.

    TIMES_VISITED = np.zeros((len(vertices)), dtype=np.int)

    def compute (a,b):

        nonlocal count, total, total_steps, total_length
        global TIMES_VISITED

        if a != b:
            path_length = 0
            path_steps = 0

            try:
                path_steps = nx.shortest_path(g, a, b, weight='length')
                for pi in range(len(path_steps) - 1):
                    path_length += l2p(path_steps[pi], path_steps[pi + 1], vertices)
                    TIMES_VISITED[path_steps[pi]] = TIMES_VISITED[path_steps[pi]] + 1
                path_steps = len (path_steps)

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
            total_steps += path_steps
            total_length += path_length

    if False: # compute for all ~1hr
        for a in range(len(vertices)):
            print(f'    {a}/{len(vertices)}')
            for b in range(len(vertices)):
                print(f'      {b}/{len(vertices)} transport ratio: {total/count}')
                compute (a,b)
        table_row_names.append(f"Transport ratio")
    else:
        for i in range(SAMPLES):
            print(f'    {i}/{SAMPLES}: transport ratio: {total/max(1,count)}')
            a = random.randint(0, len(vertices))
            b = random.randint(0, len(vertices))
            compute(a,b)

        table_row_names.append(f"Transport ratio (n={SAMPLES})")

    table_data.append(("%.2f" % (total / count)))

    table_row_names.append(f"Mean steps in random walk (n={SAMPLES})")
    table_data.append(("%.2f" % (total_steps / count)))

    table_row_names.append(f"Mean length of random walk (m) (n={SAMPLES})")
    table_data.append(("%.2f" % (total_length / count)))

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

def betweenness_centrality( vertices, edges, table_data, table_row_names, minn=0, maxx=0.4, bins = 32 ):

    out = np.zeros((bins), dtype=np.int)
    g = build_graph(vertices, edges)

    print ("      starting betweenness_centrality...")
    p = nx.betweenness_centrality(g, k=min(SAMPLES, len(vertices)) )
    print ("      ...done")

    total = 0
    max_bc = -1

    for i in range(len(vertices)):
        bc = p[i]
        idx = floor( ( bc -minn) * bins / (maxx - minn))
        idx = min(idx, bins - 1)
        out[idx] += 1
        max_bc = max(max_bc, bc)
        total += bc

    table_data.append("%.4f" % (total / float(len(vertices)) ))  # mean edges at a vertex)
    table_row_names.append(f"Mean betweenness centrality (n={SAMPLES})")

    table_data.append("%.4f" % max_bc )
    table_row_names.append(f"Maximum betweenness centrality (n={SAMPLES})")

    return out



# this doesn't help anyone!
# def plot_betweenness_centrality(all_city_stat, name, fig, subplots, subplot_idx, minn=0, maxx=0.4, bins = 32 ):
#
#     axs = plt.subplot(subplots, 1, subplot_idx)
#     axs.title.set_text(name)
#     axs.spines['top'].set_color('lightgray')
#     axs.spines['right'].set_color('lightgray')
#
#     for idx, r in enumerate ( all_city_stat ):
#         x_pos = np.arange(bins)
#         x_lab = list ( map (lambda x : " %.4f" % ((maxx-minn)*x/bins + minn), x_pos ) )
#
#         plt.ylabel("Proportion")
#         plt.xlabel("Betweenness centrality")
#
#         utils.plot(r, plt, bins, idx, all_city_stat)
#
#         x_pos = x_pos[::4]
#         x_lab = x_lab[::4]
#
#         x_lab[len(x_lab)-1] = "> %.2f" % maxx
#         x_pos[len(x_pos)-1] = bins -1
#
#         plt.xticks(x_pos, x_lab)

def pagerank( vertices, edges, table_data, table_row_names, minn=0, maxx=0.0002, bins = 32, norm = True):

    out = np.zeros((bins), dtype=np.int)
    g = build_graph(vertices, edges)

    print ("      starting pagerank...")
    p = nx.pagerank(g, tol=1e-6 ) # , weight='length') - weight as length doesn't make sense - more transfer down longer roads?
    print ("      ...done")

    total = 0
    max_pr = -1
    min_pr = -1

    for i in range(len(vertices)):
        pr = p[i]
        idx = floor( ( pr -minn) * bins / (maxx - minn))
        idx = min(idx, bins - 1)
        out[idx] += 1
        max_pr = max(max_pr, pr)
        min_pr = max(min_pr, pr)
        total += pr

    table_data.append("%.6f" % (total / float(len(vertices))))  # mean edges at a vertex)
    table_row_names.append(f"Mean pagerank")

    table_data.append("%.6f" % max_pr)
    table_row_names.append(f"Maximum pagerank")

    table_data.append("%.6f" % min_pr)
    table_row_names.append(f"Minimum pagerank")

    out = out/len(vertices)

    return out

def plot_pagerank(all_city_stat, name, fig, subplots, subplot_idx, minn=0, maxx=0.0002, bins = 32 ):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text("Pagerank (topology only)")
    axs.spines['top'].set_color('lightgray')
    axs.spines['right'].set_color('lightgray')

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = list ( map (lambda x : " %.6f" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Pagerank")

        utils.plot(r, plt, bins, idx, all_city_stat)

        x_pos = x_pos[::4]
        x_lab = x_lab[::4]

        x_lab[len(x_lab)-1] = "> %.6f" % maxx
        x_pos[len(x_pos)-1] = bins -1

        plt.xticks(x_pos, x_lab)


def pagerank_on_edges( vertices, edges, table_data, table_row_names, minn=0, maxx=0.0002, bins = 32, norm = True):

    out = np.zeros((bins), dtype=np.int)

    # build a dual-graph
    d = nx.Graph()

    v2e = build_V2E(vertices, edges)

    personalization = {}

    for e_idx, e in enumerate ( edges ):

        d.add_node(str(e), weight = l2(e, vertices))

        personalization[str(e)] = l2(e, vertices)

        for i in e:
            for e2 in v2e.v2e[i]:
                if not np.array_equal(e2, e):
                    d.add_edge(str(e), str(e2))

    print ("      starting pagerank...")
    p = nx.pagerank(d, max_iter = 1000, tol=1e-8, personalization=personalization)
    print ("      ...done")

    total = 0
    max_pr = -1
    min_pr = 1e100

    for e in edges:
        pr = p[str(e)]
        idx = floor( ( pr -minn) * bins / (maxx - minn))
        idx = min(idx, bins - 1)
        out[idx] += 1
        max_pr = max(max_pr, pr)
        min_pr = min(min_pr, pr)
        total += pr

    table_data.append("%.6f" % (total / float(len(vertices))))  # mean edges at a vertex)
    table_row_names.append(f"Mean pagerank-by-edge")

    table_data.append("%.6f" % max_pr)
    table_row_names.append(f"Maximum pagerank-by-edge")

    table_data.append("%.6f" % min_pr)
    table_row_names.append(f"Minimum pagerank-by-edge")

    out = out/len(edges)

    return out

def plot_pagerank_on_edges(all_city_stat, name, fig, subplots, subplot_idx, minn=0, maxx=0.0002, bins = 32 ):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text("Pagerank-by-Edge (initialisation with street lengths)")
    axs.spines['top'].set_color('lightgray')
    axs.spines['right'].set_color('lightgray')

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = list ( map (lambda x : " %.6f" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Pagerank")

        utils.plot(r, plt, bins, idx, all_city_stat)

        x_pos = x_pos[::4]
        x_lab = x_lab[::4]

        x_lab[len(x_lab)-1] = "> %.6f" % maxx
        x_pos[len(x_pos)-1] = bins -1

        plt.xticks(x_pos, x_lab)