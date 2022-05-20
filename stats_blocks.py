
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

            # print ( "start is %s " % str(vertices[start[0]]) + ", " + str(vertices[start[1]] ) )

            next_e = start
            next_v = start[1] # pt_idx
            prev_v = start[0]

            area = 0

            while next_v != start[0]: # and len (B) < 100:

                B.append(next_v)

                area += -utils.area( start[0], prev_v, next_v, vertices )

                pts = v2e.get_other_pts(next_v)
                min = 1000

                bad = False
                for idx, p_idx in enumerate ( pts ):
                    if p_idx != prev_v and (next_v, p_idx) in remaining:
                        a = angle ( prev_v, next_v, p_idx, vertices)
                        if a < min:
                            next_next_v = pts[ idx ]
                            min = a

                if min == 1000:
                    if (next_v, prev_v) in remaining and len(pts) == 1:
                        next_next_v = prev_v # traversing a dead-end
                        #print ("dead end!")
                    else:
                        bad = True

                if bad:
                    break

                #next_next_v = pts [ np.fromiter( map(lambda p_idx: angle ( prev_v, next_v, p_idx, vertices ), pts),  dtype=np.float).argmin() ]

                prev_v = next_v
                next_v = next_next_v

                # if (prev_v, next_v) in remaining:
                remaining.remove( (prev_v, next_v) )
                # print("removed edge %d  %s " % (len(remaining), str(vertices[prev_v])+", "+str(vertices[next_v]) ))

            # print (" blcok area is %.2f " % area)

            # if area > 0:
            if len(B) < 100 and area > 0 and not bad:
                # print ("adding block of area %.2f" % area)
                BLOCKS.append(B)
                BLOCK_AREA.append(area)
            else:
                pass
                # print("ignoring block bad:" +str(bad))

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


def block_perimeter( vertices, edges, table_data, table_row_names, render_params, minn=0, maxx=2000, bins = 32, norm = True ):

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

    table_data.append("%d" % len(blocks))
    table_row_names.append("Number of blocks")

    table_data.append( "%.2f" % ( len(blocks)/ utils.land_area_km() ) )
    table_row_names.append( "Block density (blocks/km^2)" )

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

def block_area ( vertices, edges, table_data, table_row_names, render_params, minn=0, maxx=30000, bins = 32, norm = True ):

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



def block_aspect ( vertices, edges, table_data, table_row_names, render_params, minn=0, maxx=1, bins = 32, norm = True ):

    out = np.zeros((bins), dtype=np.int)

    blocks = build_blocks(vertices, edges)
    areas = get_block_areas()

    count = 0
    total = 0
    total_rectness = 0

    block_centers = np.zeros((len(blocks), 2), dtype=float)
    per_block = np.zeros((len(blocks)))

    total_short_edge_len = [] #np.zeros((len(blocks)))
    total_long_edge_len = [] #np.zeros((len(blocks)))

    for b_idx, v_ids in enumerate(blocks):

        area = areas[b_idx]

        length = 0

        locs = [] # np.zeros((len(v_ids), 2))
        bad = False

        prev = vertices[v_ids[len(v_ids)-1]]

        for idx in range(len ( v_ids ) ):

            next = vertices[v_ids[idx]]

            if np.linalg.norm(prev-next) > 0.001:
                locs.append(next)
                prev = next

        if bad:
            continue # can't hull if points are too close together

        hull = ConvexHull(locs)

        best = 1e10
        short = -1
        long = -1
        count = count + 1

        for simplex in hull.simplices:

            h, w = bb( locs, locs[simplex[0]], locs[simplex[1]] )

            if (h * w) < best:
                best = h * w

                short = min(h,w)
                long = max (h,w)

                aspect = short/long

        # total_short_edge_len[b_idx] = short
        # total_long_edge_len[b_idx] = long
        total_short_edge_len.append(short)
        total_long_edge_len .append(long)

        idx = floor(aspect * bins / (maxx - minn))
        idx = min(idx, bins - 1)
        out[idx] += 1

        total += aspect
        count = count + 1

        rectness = area / (h * w)
        total_rectness += rectness

        cen = block_centers[b_idx]# np.array((3), dtype=float)
        for idx in range(len(v_ids)):
            cen = cen +  vertices[v_ids[idx]]

        cen /= len(v_ids)
        block_centers[b_idx] = cen
        per_block    [b_idx] = aspect

    table_row_names.append("Mean block aspect ratio")
    table_row_names.append("Mean block rectangularness")

    if len (blocks) > 0:
        table_data.append("%.2f" % (total / float(len(blocks))))
        table_data.append("%.2f" % (total_rectness / float(count)))
    else:
        table_data.append("-")
        table_data.append("-")

    render_params.append(dict( block_pts=block_centers, block_cols=utils.norm_and_color_map(per_block), name=f"Block aspect ratio"))

    total_long_edge_len = np.array(total_long_edge_len)
    total_short_edge_len = np.array(total_short_edge_len)

    table_row_names.append("Mean block short length")
    table_data.append("%.2f" % (total_short_edge_len.mean()))
    table_row_names.append("Mean block deviation short")
    table_data.append("%.2f" % (total_short_edge_len.std()))

    table_row_names.append("Mean block long length")
    table_data.append("%.2f" % (total_long_edge_len.mean()))
    table_row_names.append("Mean block deviation long")
    table_data.append("%.2f" % (total_long_edge_len.std()))

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
