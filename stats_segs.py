
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from utils import l2, COLORS

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

def reset_seg_cache():
    global SEGS, SEG_LENGTHS
    SEGS = None
    SEG_LENGTHS = None

def segment_length( vertices, edges, table_data, table_row_names, minn=0, maxx=400, bins = 32, norm = True ):

    out = np.zeros((bins), dtype=np.int)

    segs = build_segs(vertices, edges)
    sl = get_seg_lengths()

    total = 0.

    edge_cols = {}

    for s_idx, s in enumerate ( segs ):

        length = sl[s_idx]
        # for i in range(len ( s)-1):
        #     a = np.array(vertices[s[i]])
        #     b = np.array(vertices[s[i+1]])
        #     l = np.linalg.norm(a - b)
        #     length = length + l
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

    return out

def plot_segment_length(all_city_stat, name, fig, subplots, subplot_idx, minn=0, maxx=400, bins = 32):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text(name)

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Segment length (m)")

        plt.bar(x_pos + idx* (1/float(len (all_city_stat)+1)), r,1. / (len (all_city_stat)+1), color=COLORS[idx] )

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
        #print (ratio)

        idx = floor( (ratio-minn) * bins / (maxx - minn))
        idx = min(idx, bins - 1)
        out[idx] = out[idx] + 1
        count = count + 1

    table_data.append("%.4f" % (total / len(segs)))
    table_row_names.append("Mean (all) segment circuity")

    if norm:
        out = out / float (count)

    return out

def plot_segment_circuity(all_city_stat, name, fig, subplots, subplot_idx,  minn=1, maxx=2, bins = 32, norm = True ):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text("Segment Circuity (for values > 1.02)")

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = x_pos * ((maxx-minn) / bins ) + minn #  list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Segment circuity ratio")

        plt.bar(x_pos + idx* (1/float(len (all_city_stat))), r,1. / len (all_city_stat), color=COLORS[idx] )


        x_pos = x_pos[::5]
        x_lab = x_lab[::5]

        x_lab = ["%.2f" % y for y in x_lab]
        x_lab[len(x_lab)-1] = "> %.2f" % maxx
        x_pos[len(x_lab)-1] = bins -1

        plt.xticks(x_pos, x_lab)