import collections
import math
import os
import sys

import numpy as np
import load_tiles_and_plot

import matplotlib
import matplotlib.pyplot as plt

from math import floor


def edge_count(vertices, edges, table_data, table_row_names ):
    table_data.append( len (edges) ) # mean edges at a vertex)
    table_row_names.append("Number of edges")

def vertex_count(vertices, edges, table_data, table_row_names ):
    table_data.append( len (vertices) ) # mean edges at a vertex)
    table_row_names.append("Number of vertices")

def total_len(vertices, edges, table_data, table_row_names ):

    for e in edges:
        a = np.array ( vertices[e[0]] )
        b = np.array ( vertices[e[1]] )
        dist = np.linalg.norm(a - b)
        total = total + dist

    table_data.append( total ) # mean edges at a vertex)
    table_row_names.append("Total length of streets (m)")

def edge_length(vertices, edges, table_data, table_row_names, minn=0, maxx=250, bins = 20, norm = True):

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

    table_data.append( total / len (edges) ) # mean edges at a vertex)
    table_row_names.append("Mean edge length (m)")

    if norm:
        out = out / float ( len (edges) )

    return out


def plot_edge_length(all_city_stat, name, fig, subplots, subplot_idx, minn=0, maxx=250, bins = 20):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text(name)

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Edge length (m)")

        plt.bar(x_pos + idx* (1/float(len (all_city_stat))), r,1. / len (all_city_stat), color=COLORS[idx] )
        plt.xticks(x_pos[::2], x_lab[::2])


SEGS = None
def build_segs(vertices, edges):

    global SEGS
    if SEGS is not None:
        v2e = {}
        for v in vertices:
            key = v.tobytes()
            v2e[key] = []

        for e in edges:
            sk = vertices[e[0]].tobytes()
            ek = vertices[e[1]].tobytes()

            v2e[sk].append( e+1)
            v2e[ek].append(-e-1)

        def is_jn(v):
            return len(v2e[v.tobytes()]) > 1

        def get_other_edge(e, v): # get the other street of hte street from the vertex
            pass

        def get_other_vert(e, v): #
            pass

        remaining = set(edges)

        while len(remaining) > 0:
            start = remaining.pop()

            S = [start]
            SEGS.append(S)

            for idx, pt in enumerate([start]): # look forwards and backwards

                current = start

                while not is_jn(pt): # keep going until we hit a junction

                    next = get_other_edge(current, pt)

                    if next not in remaining: # loop!
                        break

                    remaining.remove(next)

                    if idx == 0:
                        S.insert(0, next)
                    else:
                        S.append(next)

                    pt = get_other_vert(next, pt)

    return SEGS

def segment_length( vertices, edges, table_data, table_row_names, minn=0, maxx=250, bins = 20, norm = True ):

    out = np.zeros((bins), dtype=np.int)

    segs = build_segs(vertices, edges)

    total = 0.

    for s in segs:
        len = 0
        for e in s:
            a = np.array(vertices[e[0]])
            b = np.array(vertices[e[1]])
            l = np.linalg.norm(a - b)
            len = len + l
            total = total + l

        len = max (minn, min(maxx, len), len)

        idx = floor(len * bins / (maxx - minn))
        idx = min(idx, bins - 1)
        out[idx] = out[idx] + 1

    table_data.append( len(segs) )
    table_row_names.append("Number of segments")

    table_data.append( total / float(len(segs)) )
    table_row_names.append("Mean length of segment (m)")

    table_data.append(len (edges) / float (len(segs)))
    table_row_names.append("Mean edges per segment")

    if norm:
        out = out / total

    return out

def plot_segment_length(all_city_stat, name, fig, subplots, subplot_idx, minn=0, maxx=250, bins = 20):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text(name)

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Segment length (m)")

        plt.bar(x_pos + idx* (1/float(len (all_city_stat))), r,1. / len (all_city_stat), color=COLORS[idx] )
        plt.xticks(x_pos[::2], x_lab[::2])

def edge_angle(vertices, edges, table_data, table_row_names, bins = 18, norm = True ):

    out = np.zeros((bins), dtype=np.int)

    total = 0.

    for e in edges:
        a = np.array(vertices[e[0]])
        b = np.array(vertices[e[1]])

        d = b-a

        angle = math.atan2(d[0], d[1])

        idx = floor(angle * bins /3.141 )
        idx = min(idx, bins - 1)

        len = np.linalg.norm(a - b)
        total = total + len

        out[idx] = out[idx] + len

    #table_data[table_row, table_col] = np.argmax(out) * 3.141 / bins

    table_data.append(np.argmax(out) * 3.141 / bins ) # mean edges at a vertex)
    table_row_names.append("Argmax edge angle (rads)")

    if norm:
        out = out / total

    return out

def plot_edge_angle(all_city_stat, name, fig, subplots, subplot_idx, bins = 18):

    axs = plt.subplot(subplots, 1, subplot_idx, polar=True)
    axs.title.set_text("Fraction of Street Length by Angle")
    axs.set_theta_zero_location("N")
    axs.set_theta_direction(-1)

    #axs.set_theta_offset(pi)

    for idx, r in enumerate ( all_city_stat ):

        x_pos = np.arange(2*bins)
        x_pos = list(map ( lambda x : x * np.pi / bins, x_pos) )
        #x_lab = list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        # plt.ylabel("frequency")
        # plt.xlabel("edge length (m)")

        plt.bar(np.arange(2*bins) + idx* (1/float(len (all_city_stat))), np.concatenate((r,r)), width = np.pi/(bins * 0.5 * len(all_city_stat)), color=COLORS[idx], edgecolor="white")
        # plt.xticks(x_pos[::2], x_lab[::2])

def node_degree(vertices, edges, table_data, table_row_names, maxx = 6, norm = True ):

    out = np.zeros((maxx), dtype=np.int)

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
    for i in range(1, maxx):
        if i in counter:
            out[i - 1] = counter[i]
            total = total + counter[i]


    table_data.append(len (edges) * 2 / len (vertices) ) # mean edges at a vertex)
    table_row_names.append("Mean node degree")

    table_data.append(out[0])  # mean edges at a vertex)
    table_row_names.append("Number of dead ends")

    if norm:
        out = out / float(total)

    return out

def plot_node_degree(all_city_stat, name, fig, subplots, subplot_idx, maxx = 6):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text(name)

    for idx, r in enumerate ( all_city_stat ):
        x_pos = list(range(maxx-1))
        x_lab = list(range(1,maxx))

        plt.ylabel("Proportion")
        plt.xlabel("Node degree")

        plt.bar(np.arange(maxx) + idx* (1/float(len (all_city_stat))), r, width= 1. / len (all_city_stat), color=COLORS[idx])

        plt.xticks(x_pos, x_lab)
        # axis.plot( x_pos, r, 'tab:orange')

COLORS = ['#8eff98', '#ffcf8e', '#8ee6ff', '#fff58e', '#ff918e']

def main():

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    #plt.style.use('ggplot')

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    npz_file_names = [x for x in os.listdir(input_path) if x.endswith('.npz')]

    metric_fns = [ 'edge_count', 'vertex_count',
                   'edge_length', 'edge_angle', 'node_degree',
                   'segment_length', 'segment_circuity']

    # transport ratio
    # street length (between junctions)
    # circuity (length of street / distance from start to end)
    # angles between streets at junctions (?!)
    # pagerank distribution of nodes; streets
    # block aspect ratio

    # single figure values
    # node count, edge count, street count, edge length
    # street density
    # node density



    all_city_stats = {}
    for m in metric_fns:
        all_city_stats[m] = []

    #table_data = np.zeros((len(npz_file_names),len ( metric_fns)) )

    table_row_names = []
    table_data = []

    for idx, npz in enumerate(npz_file_names):

        npz_path = os.path.join(input_path, npz)
        np_file_content = np.load(npz_path)

        # Vertices
        vertices = np_file_content['tile_graph_v'] * 20000 # distances in meters
        edges = np_file_content['tile_graph_e']

        td = []
        table_data.append(td)
        table_row_names = [] # only last iteration used!

        for m_idx, m in enumerate(metric_fns):
            all_city_stats[m].append(globals()[m](vertices, edges, td, table_row_names))

    #fig, axs = plt.subplots(len(metric_fns) + 1, 1)
    graph_fns = []
    for idx, m in enumerate(metric_fns):
        name = 'plot_'+m
        if name in globals():
            graph_fns.append(name)

    fig = plt.figure(figsize=(10, (len ( graph_fns ) + 1) *5))
    plt.subplots_adjust(wspace = 0.5, hspace = 1)

    subplot_count = len(graph_fns) + 1
    axs = plt.subplot(subplot_count, 1, 1)

    axs.axis('off')
    table_strs = [[ "%.2f" % y for y in x] for x in np.transpose(table_data)]
    metric_names = list ( map (lambda m : m.replace("_", " "), metric_fns))
    tab = axs.table(cellText=table_strs, colLabels=npz_file_names, rowLabels=table_row_names, loc='center', cellLoc='center')
    tab.auto_set_column_width(col=list(range(len(npz_file_names))))
    tab.set_fontsize(8)
    for i in range (len (npz_file_names)):
        #tab[(0, i)].get_text().set_color(COLORS[i])
        tab[(0, i)].get_text().set_facecolor(COLORS[i])

    subplot_pos = 0
    for idx, m in enumerate(metric_fns):
        name = 'plot_'+m
        if name in globals():
            globals()[name](all_city_stats[m], metric_names[idx].title(), fig, subplot_count, subplot_pos+2)
            subplot_pos = subplot_pos + 1

    plt.show()



if __name__ == '__main__':
    # Input Directory holding npz files
    sys.argv.append(r'npz')

    # Output Directory to save the images
    sys.argv.append(r'npz\img')

    # dxf_to_npz("C:\\Users\\twak\\Documents\\CityEngine\\Default Workspace\\datatest\\data\\dxf_streets_1.dxf", 20000, "test.npz")

    main()



