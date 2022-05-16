
import numpy as np
import matplotlib.pyplot as plt
from math import floor

import utils
from utils import angle, COLORS

from fast_plot import FastPlot

from scipy.spatial import ConvexHull

from stats_segs import VertexMap, V2E, build_V2E

BLOCKS = None
BLOCK_EDGES = None
BLOCK_AREA = None

def build_blocks(vertices, edges):

    global BLOCKS, BLOCK_EDGES, BLOCK_AREA


    if BLOCKS == None:

        BLOCKS = [] # list of vertex indicies for per block
        BLOCK_EDGES = [] # list of unordered edge indicies per block
        BLOCK_AREA = []

        v2e = build_V2E(vertices, edges)

        remaining = {tuple(e) for e in edges}.union( {( e[1], e[0] ) for e in edges} ) # edges in both directions

        while len(remaining) > 0:

            start = remaining.pop()

            B = [start[0]]

            next_e = start
            next_v = start[1] # pt_idx
            prev_v = start[0]

            area = 0

            while next_v != start[0]: # and len (B) < 100:

                B.append(next_v)

                pts = v2e.get_other_pts( next_v)

                area += utils.area( start[0], prev_v, next_v, vertices )

                min = 1000
                for idx, p_idx in enumerate ( pts ):
                    if p_idx != prev_v and (next_v, p_idx) in remaining:
                        a = -angle ( prev_v, next_v, p_idx, vertices)
                        if a < min:
                            next_next_v = pts[ idx ]
                            min = a

                if min == 1000:
                    if (next_v, prev_v) in remaining:
                        next_next_v = prev_v # traversing a dead-end
                    else:
                        raise RuntimeError("failed to find block")


                #next_next_v = pts [ np.fromiter( map(lambda p_idx: angle ( prev_v, next_v, p_idx, vertices ), pts),  dtype=np.float).argmin() ]

                prev_v = next_v
                next_v = next_next_v

                # print ("here %d" % len (remaining))
                # if (prev_v, next_v) in remaining:
                remaining.remove( (prev_v, next_v) )

            # print (" blcok area is %.2f " % area)

            if len(B) < 100 and area > 0:
                BLOCKS.append(B)
                BLOCK_AREA.append(area)

    return BLOCKS

def get_block_areas():
    global BLOCK_AREA

    if BLOCK_AREA is None:
        raise RuntimeError("call build segments first!")

    return BLOCK_AREA

def get_block_by_edge():
    global BLOCK_EDGES

    if BLOCK_EDGES is None:
        raise RuntimeError("call build segments first!")

    return BLOCK_EDGES

def reset_block_cache():
    global BLOCKS, BLOCK_EDGES, BLOCK_AREA
    BLOCKS = None
    BLOCK_EDGES = None
    BLOCK_AREA = None


def block_perimeter( vertices, edges, table_data, table_row_names, minn=0, maxx=2000, bins = 32, norm = True ):

    out = np.zeros((bins), dtype=np.int)

    blocks = build_blocks(vertices, edges)

    total = 0
    total_edges = 0

    for b_idx, pts in enumerate ( blocks ):

        length = 0

        for idx in range(len ( pts ) ):
            length += utils.l2p ( pts[idx], pts[ (idx+1) % len(pts)], vertices )

        if length < 2000:
            idx = floor ( length * bins / (maxx - minn) )
            idx = min(idx, bins-1)
            out[idx] += 1

        total = total + length
        total_edges = total_edges + len (pts)

    table_data.append("%d" % len ( blocks ) )
    table_row_names.append("Number of blocks")

    if len (blocks) > 0:
        table_data.append("%.2f" % (total / float(len(blocks))))
        table_data.append("%.2f" % (total_edges / float(len(blocks))))
    else:
        table_data.append("-")
        table_data.append("-")

    table_row_names.append("Mean block perimeter (m)")
    table_row_names.append("Mean edges per block")

    if norm:
        out = out / float ( len (blocks) )

    #FastPlot(2048, 2048, vertices, edges, scale=0.1, edge_cols = np.array(edge_cols) ).run()

    return out

def plot_block_perimeter (all_city_stat, name, fig, subplots, subplot_idx, minn=0, maxx=2000, bins = 32):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text("Block Perimeter (< 2,000m)")
    axs.spines['top'].set_color('lightgray')
    axs.spines['right'].set_color('lightgray')

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Block perimeter length (m)")

        utils.plot(r, plt, bins, idx, all_city_stat)

        x_pos = x_pos[::4]
        x_lab = x_lab[::4]

        # x_lab[len(x_lab)-1] = "> %d" % maxx
        # x_pos[len(x_pos)-1] = bins -1

        plt.xticks(x_pos, x_lab)

def block_area ( vertices, edges, table_data, table_row_names, minn=0, maxx=30000, bins = 32, norm = True ):

    out = np.zeros((bins), dtype=np.int)

    blocks = build_blocks(vertices, edges)
    areas = get_block_areas()

    total = 0.
    count = 0

    for b_idx, pts in enumerate ( blocks ):

        area = areas[b_idx]

        if area < 1:
            continue

        count += 1
        total += area

        if area < 30000:
            idx = floor ( area * bins / (maxx - minn) )
            idx = min(idx, bins-1)
            out[idx] += 1

    table_row_names.append("Mean block area (m^2)")

    if len (blocks) > 0:
        table_data.append("%.2f" % (total / len(blocks)))
    else:
        table_data.append("-")


    if norm:
        out = out / float ( len(blocks) )

    #FastPlot(2048, 2048, vertices, edges, scale=0.1, edge_cols = np.array(edge_cols) ).run()

    return out

def plot_block_area (all_city_stat, name, fig, subplots, subplot_idx, minn=0, maxx=30000, bins = 32):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text("Block Area ( 1m^2 < area <  30,000m^2)")
    axs.spines['top'].set_color('lightgray')
    axs.spines['right'].set_color('lightgray')

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Block area (m^2)")

        utils.plot(r, plt, bins, idx, all_city_stat)

        x_pos = x_pos[::4]
        x_lab = x_lab[::4]

        # x_lab[len(x_lab)-1] = "> %d" % maxx
        # x_pos[len(x_pos)-1] = bins -1

        plt.xticks(x_pos, x_lab)


def bb (pts, a, b): # size of pts bounding box with axis a-> b

    horz = b-a
    horz /= np.linalg.norm(horz)
    vert = [-horz[1], horz[0]]

    big = 10e7

    min_h = big
    max_h = -big
    min_v = big
    max_v = -big

    for pt in pts:
        tp = pt - a

        h = tp.dot(horz)
        v = tp.dot(vert)

        min_h = min(min_h, h)
        max_h = max(max_h, h)
        min_v = min(min_v, v)
        max_v = max(max_v, v)

    return max_v - min_v, max_h - min_h



def block_aspect ( vertices, edges, table_data, table_row_names, minn=0, maxx=1, bins = 32, norm = True ):

    out = np.zeros((bins), dtype=np.int)

    blocks = build_blocks(vertices, edges)
    areas = get_block_areas()

    total = 0
    total_rectness = 0


    for b_idx, pts in enumerate ( blocks ):

        area = areas[b_idx]

        length = 0

        locs = np.zeros((len(pts), 2))
        for idx in range(len ( pts ) ):
            locs[idx] = vertices[pts[idx]]

        hull = ConvexHull(locs)

        best = 1e10
        for simplex in hull.simplices:

            h, w = bb( locs, locs[simplex][0], locs[simplex][1] )

            if (h * w) < best:
                best = h * w
                aspect = min(h,w) / max(h,w)

        idx = floor(aspect * bins / (maxx - minn))
        idx = min(idx, bins - 1)
        out[idx] += 1

        total += aspect

        rectness = area / (h * w)
        total_rectness += rectness

    table_row_names.append("Mean block aspect ratio")
    table_row_names.append("Mean block rectangularness")

    if len (blocks) > 0:
        table_data.append("%.2f" % (total / float(len(blocks))))
        table_data.append("%.2f" % (total_rectness / float(len(blocks))))
    else:
        table_data.append("-")
        table_data.append("-")

    if norm:
        out = out / float ( len (blocks) )

    return out

def plot_block_aspect (all_city_stat, name, fig, subplots, subplot_idx,minn=0, maxx=1, bins = 32):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text("Block Bounding Box Aspect Ratio")
    axs.spines['top'].set_color('lightgray')
    axs.spines['right'].set_color('lightgray')

    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = list ( map (lambda x : " %.2f" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("BB aspect ratio")

        # plt.axvline(x= (1 * bins / (maxx-minn)), color='lightgrey')

        utils.plot(r, plt, bins, idx, all_city_stat)


        x_pos = x_pos[::4]
        x_lab = x_lab[::4]

        # x_lab[len(x_lab)-1] = "> %d" % maxx
        # x_pos[len(x_pos)-1] = bins -1

        plt.xticks(x_pos, x_lab)
