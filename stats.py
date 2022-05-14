import collections
import math
import os
import sys

import numpy as np
import load_tiles_and_plot

import matplotlib
import matplotlib.pyplot as plt

from math import floor

def l2(e, vertices):
    a = np.array(vertices[e[0]])
    b = np.array(vertices[e[1]])
    return np.linalg.norm(a - b)


def edge_count(vertices, edges, table_data, table_row_names ):
    table_data.append( len (edges) ) # mean edges at a vertex)
    table_row_names.append("Number of edges")

def vertex_count(vertices, edges, table_data, table_row_names ):
    table_data.append( len (vertices) ) # mean edges at a vertex)
    table_row_names.append("Number of vertices")

def total_len(vertices, edges, table_data, table_row_names ):

    for e in edges:
        dist = l2(e)
        total = total + dist

    table_data.append( total ) # mean edges at a vertex)
    table_row_names.append("Total length of streets (m)")


def edge_length(vertices, edges, table_data, table_row_names, minn=0, maxx=250, bins = 20, norm = True):

    out = np.zeros((bins), dtype=np.int)

    total = 0.

    for e in edges:

        dist = l2(e, vertices)

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

        plt.bar(x_pos + idx* (1/float(len (all_city_stat)+1)), r,1. / (len (all_city_stat)+1), color=COLORS[idx] )
        plt.xticks(x_pos[::2], x_lab[::2])


SEGS = None
SEG_LENGTHS = None

def build_segs(vertices, edges):

    global SEGS, SEG_LENGTHS

    if SEGS == None:

        SEG_LENGTHS = []
        SEGS = []

        v2e = {}
        for v in vertices:
            key = v.tobytes()
            v2e[key] = []

        for e in edges:
            sk = vertices[e[0]].tobytes()
            ek = vertices[e[1]].tobytes()

            v2e[sk].append( e )
            v2e[ek].append( e )

        def is_jn(v_idx):
            return len(v2e[vertices[v_idx].tobytes()]) != 2

        def get_other_edge(e1_, v_idx): # get the other street of hte street from the vertex

            e1 = np.array(e1_)
            for e2 in v2e[ vertices[v_idx].tobytes()]:
                if not np.array_equal ( e2, e1):
                    return e2

            raise RuntimeError("get other edge %d %s" % (e1, v_idx))


        def get_other_vert_idx(e, v_idx): #

            if e[0] == v_idx:
                return e[1]
            elif e[1] == v_idx:
                return e[0]
            else:
                raise RuntimeError("get other vert %d %s" % (e, v_idx))

        remaining = {tuple(row) for row in edges}

        while len(remaining) > 0:

            start = remaining.pop()

            seg_len = l2(start, vertices)

            S = [start[0], start[1]]
            SEGS.append(S)

            for idx, pt_idx in enumerate(start): # look forwards and backwards

                next = start

                while not is_jn(pt_idx): # keep going until we hit a junction

                    next = get_other_edge(next, pt_idx)

                    if tuple ( next ) not in remaining: # loop!
                        break

                    remaining.remove(tuple ( next ))
                    seg_len = seg_len + l2(next, vertices)

                    pt_idx = get_other_vert_idx(next, pt_idx)

                    if idx == 0:
                        S.insert(0, pt_idx)
                    else:
                        S.append(pt_idx)

            SEG_LENGTHS.append (seg_len)

    return SEGS

def get_seg_lengths():
    global SEG_LENGTHS

    if SEG_LENGTHS is None:
        raise RuntimeError("call build segments first!")

    return SEG_LENGTHS

def segment_length( vertices, edges, table_data, table_row_names, minn=0, maxx=250, bins = 20, norm = True ):

    global SEGS, SEG_LENGTHS
    SEGS = None
    SEG_LENGTHS = None

    out = np.zeros((bins), dtype=np.int)

    segs = build_segs(vertices, edges)

    total = 0.

    for s in segs:
        length = 0
        for i in range(len ( s)-1):
            a = np.array(vertices[s[i]])
            b = np.array(vertices[s[i+1]])
            l = np.linalg.norm(a - b)
            length = length + l


        total = total + length

        length = max (minn, min(maxx, length), length)

        idx = floor(length * bins / (maxx - minn))
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

        plt.bar(x_pos + idx* (1/float(len (all_city_stat)+1)), r,1. / (len (all_city_stat)+1), color=COLORS[idx] )
        plt.xticks(x_pos[::2], x_lab[::2])

def segment_circuity ( vertices, edges, table_data, table_row_names, minn=1, maxx=1.1, bins = 20, norm = True ):

    out = np.zeros((bins), dtype=np.int)

    segs = build_segs(vertices, edges)
    sl = get_seg_lengths()

    total = 0.

    for seg_idx, s in enumerate ( segs ):

        euclid = np.linalg.norm(vertices[s[0]] - vertices[ s[len(s)-1] ] )

        if euclid == 0: # loop?!
            ratio = 1
        else:
            curve_len = sl[seg_idx]
            ratio = curve_len / euclid

        if (ratio == 1):
            continue

        total = total + ratio
        ratio = max(minn, min(maxx, ratio), ratio)
        print (ratio)

        idx = floor( (ratio-minn) * bins / (maxx - minn))
        idx = min(idx, bins - 1)
        out[idx] = out[idx] + 1

    table_data.append(total / len(segs))
    table_row_names.append("Mean segment circuity")

    if norm:
        out = out / len (segs)

    return out

def plot_segment_circuity(all_city_stat, name, fig, subplots, subplot_idx, minn=1, maxx=1.1, bins = 20, norm = True ):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text(name)

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = x_pos * ((maxx-minn) / bins ) + minn #  list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Segment circuity ratio")

        plt.bar(x_pos + idx* (1/float(len (all_city_stat))), r,1. / len (all_city_stat), color=COLORS[idx] )

        x_lab = ["%.2f" % y for y in x_lab]

        plt.xticks(x_pos[::2], x_lab[::2])

def edge_angle(vertices, edges, table_data, table_row_names, bins = 18, norm = True ):

    out = np.zeros((bins), dtype=np.int)

    total = 0.

    for e in edges:
        a = np.array(vertices[e[0]])
        b = np.array(vertices[e[1]])

        d = b-a

        angle = math.atan2(d[0], d[1])
        if angle < 0:
            angle += np.pi

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
        x_pos = x_pos * np.pi / bins #list(map ( lambda x : x * np.pi / bins, x_pos) )
        #x_lab = list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        # plt.ylabel("frequency")
        # plt.xlabel("edge length (m)")

        #plt.bar(x_pos + (idx * (1. / len(all_city_stat))), np.concatenate((r, r)), width=np.pi / (bins * len(all_city_stat)), color=COLORS[idx])  # , edgecolor="white"
        plt.plot(x_pos, np.concatenate((r, r)), color=COLORS[idx])  # , edgecolor="white"
        #np.arange(2 * bins) + idx * (1. / len(all_city_stat))
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

        plt.bar(np.arange(maxx) + idx* (1/float(len (all_city_stat)+1)), r, width= 1. / (len (all_city_stat)+1), color=COLORS[idx])

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
                   'edge_length', 'segment_length', 'edge_angle', 'node_degree',
                   'segment_circuity'
                   ]

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
        tab[(0, i)].set_facecolor(COLORS[i])

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



