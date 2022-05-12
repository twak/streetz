import collections
import math
import os
import sys

import numpy as np
import load_tiles_and_plot

import matplotlib
import matplotlib.pyplot as plt

from math import floor


def edge_length(vertices, edges, table_data, table_row, table_col, minn=0, maxx=250, bins = 20):

    out = np.zeros((bins), dtype=np.int)

    total = 0.

    for e in edges:
        a = np.array ( vertices[e[0]] )
        b = np.array ( vertices[e[1]] )
        dist = np.linalg.norm(a - b)

        dist = min(maxx, dist)
        dist = max(minn, dist)

        idx = floor ( dist * bins / (maxx - minn) )
        idx = min(idx, bins-1)
        out[idx] = out[idx] + 1

        total = total + dist

    table_data[table_row, table_col] = total / len (edges)

    return out

def plot_edge_length(all_city_stat, name, fig, subplots, subplot_idx, minn=0, maxx=250, bins = 20):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text(name)

    for r in all_city_stat:
        x_pos = list(range(bins))
        x_lab = list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("frequency")
        plt.xlabel("edge length (m)")

        plt.bar(x_pos, r, color='green')
        plt.xticks(x_pos[::2], x_lab[::2])
        #axis.plot( x_pos, r, 'tab:orange')

def edge_angle(vertices, edges, table_data, table_row, table_col, bins = 18 ):

    out = np.zeros((bins), dtype=np.int)

    total = 0.

    for e in edges:
        a = np.array(vertices[e[0]])
        b = np.array(vertices[e[1]])

        d = b-a

        angle = math.atan2(d[0], d[1])

        idx = floor(angle * bins /3.141 )
        idx = min(idx, bins - 1)

        out[idx] = out[idx] + 1

    table_data[table_row, table_col] = np.argmax(out) * 3.141 / bins

    return out

def plot_edge_angle(all_city_stat, name, fig, subplots, subplot_idx, bins = 18):

    axs = plt.subplot(subplots, 1, subplot_idx, polar=True)
    axs.title.set_text(name)

    for r in all_city_stat:

        x_pos = list(range(bins * 2))
        x_pos = list(map ( lambda x : x * np.pi / bins, x_pos) )
        #x_lab = list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        # plt.ylabel("frequency")
        # plt.xlabel("edge length (m)")

        plt.bar(x_pos, np.concatenate((r,r)), width = np.pi/bins, color='green', edgecolor="white")
        # plt.xticks(x_pos[::2], x_lab[::2])
        #axis.plot( x_pos, r, 'tab:orange')

def node_degree(vertices, edges, table_data, table_row, table_col, maxx = 6 ):

    out = np.zeros((maxx), dtype=np.int)

    total = 0.

    valency = {}

    def add(a):
        aa = a.tostring()
        if aa in valency:
            valency[aa]= valency[aa]+1
        else:
            valency[aa] = 1

    for e in edges:
        a = np.array(vertices[e[0]])
        b = np.array(vertices[e[1]])

        add(a)
        add(b)

    counter =collections.Counter(valency.values())

    total = 0
    for i in range (1, maxx):
        if i in counter:
            out[i-1] = counter[i]
        total = total + i * counter[i]

    table_data[table_row, table_col] = total / len ( vertices )

    return out

def plot_node_degree(all_city_stat, name, fig, subplots, subplot_idx, maxx = 6):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text(name)

    for r in all_city_stat:
        x_pos = list(range(maxx))
        x_lab = list(range(1,maxx))

        plt.ylabel("frequency")
        plt.xlabel("node degree")

        plt.bar(x_pos, r, color='green')

        #plt.xticks(x_pos[::2], x_lab[::2])
        # axis.plot( x_pos, r, 'tab:orange')

def main():

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    plt.style.use('ggplot')

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    npz_file_names = [x for x in os.listdir(input_path) if x.endswith('.npz')]

    metric_fns = ['edge_length', 'edge_angle', 'node_degree']

    # transport ratio
    # street length (between junctions)
    # circuity (length of street / distance from start to end)
    # angles between streets at junctions (?!)
    # pagerank distribution of nodes; streets

    # single figure values
    # node count, edge count, street count, edge length
    # street density
    # node density

    all_city_stats = {}
    for m in metric_fns:
        all_city_stats[m] = []

    table_data = np.zeros((len(npz_file_names),len ( metric_fns)) )

    for idx, npz in enumerate(npz_file_names):
        npz_path = os.path.join(input_path, npz)

        np_file_content = np.load(npz_path)

        # Vertices
        vertices = np_file_content['tile_graph_v'] * 20000 # distances in meters
        edges = np_file_content['tile_graph_e']

        for m_idx, m in enumerate(metric_fns):
            all_city_stats[m].append(globals()[m](vertices, edges, table_data, idx, m_idx))

    #fig, axs = plt.subplots(len(metric_fns) + 1, 1)

    fig = plt.figure(figsize=(10, (len ( metric_fns ) + 1) *4))
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5)

    subplot_count = len(metric_fns) + 1
    axs = plt.subplot(subplot_count, 1, 1)

    axs.axis('off')

    table_strs = [[ "%.2f" % y for y in x] for x in np.transpose(table_data)]

    metric_names = list ( map (lambda m : m.replace("_", " "), metric_fns))
    axs.table(cellText=table_strs, colLabels=npz_file_names, rowLabels=metric_names, loc='center')

    for idx, m in enumerate(metric_fns):
         globals()['plot_'+m](all_city_stats[m], metric_names[idx], fig, subplot_count, idx+2)

    plt.show()

if __name__ == '__main__':
    # Input Directory holding npz files
    sys.argv.append(r'npz')

    # Output Directory to save the images
    sys.argv.append(r'npz\img')

    # dxf_to_npz("C:\\Users\\twak\\Documents\\CityEngine\\Default Workspace\\datatest\\data\\dxf_streets_1.dxf", 20000, "test.npz")

    main()



