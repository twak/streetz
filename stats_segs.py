
import numpy as np
import matplotlib.pyplot as plt
from math import floor

import utils
from utils import l2, COLORS

from fast_plot import FastPlot

SEGS = None
SEG_LENGTHS = None
SEG_EDGES = None
V2E = None

class VertexMap ():

    def __init__(s, vertices, edges):
        s.v2e = {}
        # s.v2ei = {}
        s.v = vertices
        s.e = edges

        for v in range(len ( vertices) ):
            s.v2e[v] = []

        for idx, e in enumerate(edges):
            # sk = vertices[e[0]].tobytes()
            # ek = vertices[e[1]].tobytes()

            s.v2e[e[0]].append(e)
            s.v2e[e[1]].append(e)

            # s.v2e[e[0]].append(idx)
            # s.v2e[e[1]].append(idx)

    def is_jn(s,v_idx):
        return len(s.v2e[ v_idx]) != 2

    def get_other_edge(s,e1_, v_idx):  # get the other street of hte street from the vertex

        e1 = np.array(e1_)
        for e2 in s.v2e[ v_idx ]:
            if not np.array_equal(e2, e1):
                return e2

        raise RuntimeError("get other edge %d %s" % (e1, v_idx))

    def get_other_vert_idx(s,e, v_idx):  #

        if e[0] == v_idx:
            return e[1]
        elif e[1] == v_idx:
            return e[0]
        else:
            raise RuntimeError("get other vert %d %s" % (e, v_idx))

    def get_other_pts(s, next_v):

        out = []
        for e in s.v2e[ next_v ]:

            if e[0] == next_v:
                out.append(e[1])
            elif e[1] == next_v:
                out.append(e[0])
            else:
                raise RuntimeError("lookup failure")

        return out

    # def get_other_edges(s, next_v): # returns edge indicies
    #     return s.v2ei[ next_v ]

def build_V2E(v, e):

    global V2E
    if V2E is None:
        V2E = VertexMap (v, e)

    return V2E

def build_segs(vertices, edges):

    global SEGS, SEG_LENGTHS, SEG_EDGES

    if SEGS == None:

        SEG_LENGTHS = [] # per segment lengths
        SEGS = [] # list of vertex indicies for per segment
        SEG_EDGES = [] # list of unordered edge indicies per segment

        v2e = build_V2E(vertices, edges)

        remaining = {tuple(row) for row in edges}
        lookup = {}
        for e_idx, e in enumerate ( edges ):
            lookup[tuple(e)] = e_idx

        while len(remaining) > 0:

            start = remaining.pop()

            seg_len = l2(start, vertices)

            S = [start[0], start[1]]
            SEGS.append(S)

            E = [lookup [start]]
            SEG_EDGES.append( E )

            for idx, pt_idx in enumerate(start): # look forwards and backwards

                next = start

                while not v2e.is_jn(pt_idx): # keep going until we hit a junction

                    next = v2e.get_other_edge(next, pt_idx)

                    if tuple ( next ) not in remaining: # loop!
                        break

                    remaining.remove(tuple ( next ))
                    seg_len = seg_len + l2(next, vertices)

                    pt_idx = v2e.get_other_vert_idx(next, pt_idx)

                    E.append(lookup [tuple(next)])

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

def get_seg_by_edge():
    global SEG_EDGES

    if SEG_EDGES is None:
        raise RuntimeError("call build segments first!")

    return SEG_EDGES

def reset_seg_cache():
    global SEGS, SEG_LENGTHS, SEG_EDGES,V2E
    SEGS = None
    SEG_LENGTHS = None
    SEG_EDGES = None
    V2E = None

def segment_length( vertices, edges, table_data, table_row_names, minn=0, maxx=400, bins = 32, norm = True ):

    out = np.zeros((bins), dtype=np.int)

    segs = build_segs(vertices, edges)
    sl = get_seg_lengths()

    total = 0.

    edge_cols = np.zeros((len(edges),3))

    for e_list in get_seg_by_edge():

        rgb = np.random.rand(3)

        for e in e_list:
            edge_cols[e] = rgb

    for s_idx, s in enumerate ( segs ):

        length = sl[s_idx]
        total = total + length

        length = max (minn, min(maxx, length), length)

        idx = floor(length * bins / (maxx - minn))
        idx = min(idx, bins - 1)
        out[idx] = out[idx] + 1

    table_data.append( "%d" % len(segs) )
    table_row_names.append("Number of segments")

    table_data.append( "%.2f" % (total / float(len(segs)) ))
    table_row_names.append("Mean length of segment (m)")

    table_data.append( "%.2f" % (len (edges) / float (len(segs))))
    table_row_names.append("Mean edges per segment")

    if norm:
        out = out / float ( len (segs) )

    # FastPlot(2048, 2048, vertices, edges, scale=0.1, edge_cols = np.array(edge_cols) ).run()

    return out

def plot_segment_length(all_city_stat, name, fig, subplots, subplot_idx, minn=0, maxx=400, bins = 32):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text(name)
    axs.spines['top'].set_color('lightgray')
    axs.spines['right'].set_color('lightgray')

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Segment length (m)")

        utils.plot(r, plt, bins, idx, all_city_stat)

        x_pos = x_pos[::2]
        x_lab = x_lab[::2]

        x_lab[len(x_lab)-1] = "> %d" % maxx
        x_pos[len(x_pos)-1] = bins -1

        plt.xticks(x_pos, x_lab)

def segment_circuity ( vertices, edges, table_data, table_row_names, minn=1, maxx=2, bins = 32, norm = True ):

    out = np.zeros((bins), dtype=np.int)

    segs = build_segs(vertices, edges)
    sl = get_seg_lengths()

    total = 0.
    count = 0

    for seg_idx, s in enumerate ( segs ):

        euclid = np.linalg.norm(vertices[s[0]] - vertices[ s[len(s)-1] ] )

        if euclid == 0: # loop or zero length segment
            continue
        else:
            curve_len = sl[seg_idx]
            ratio = curve_len / euclid

        total = total + ratio

        if (ratio < 1.02):
            continue

        ratio = max(minn, min(maxx, ratio), ratio)

        idx = floor( (ratio-minn) * bins / (maxx - minn))
        idx = min(idx, bins - 1)
        out[idx] = out[idx] + 1
        count = count + 1

    table_data.append("%.4f" % (total / len(segs)))
    table_row_names.append("Mean segment circuity")

    if norm:
        out = out / float (count)

    return out

def plot_segment_circuity(all_city_stat, name, fig, subplots, subplot_idx,  minn=1, maxx=2, bins = 32, norm = True ):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text("Segment Circuity (> 1.02)")
    axs.spines['top'].set_color('lightgray')
    axs.spines['right'].set_color('lightgray')

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = x_pos * ((maxx-minn) / bins ) + minn #  list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Segment circuity ratio")

        utils.plot(r, plt, bins, idx, all_city_stat)

        x_pos = x_pos[::4]
        x_lab = x_lab[::4]

        x_lab = ["%.2f" % y for y in x_lab]
        x_lab[len(x_lab)-1] = "> %.2f" % maxx
        x_pos[len(x_lab)-1] = bins -1

        plt.xticks(x_pos, x_lab)