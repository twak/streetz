import traceback

import utils
from stats_segs import *
from stats_blocks import *
from stats_graph import *

import collections
import math
import os
import sys
import builtins
import PIL

import numpy as np
import load_tiles_and_plot
import matplotlib.pyplot as plt
from math import floor
from utils import l2


def land(vertices, edges, table_data, table_row_names, render_params ):

    table_data.append("%.2f" % utils.land_area_km()  )  # mean edges at a vertex)
    table_row_names.append("Land area (km^2)")

def edge_count(vertices, edges, table_data, table_row_names, render_params ):
    table_data.append("%d" % len(edges))  # mean edges at a vertex)
    table_row_names.append("Number of edges")

    table_data.append("%.2f" % (len(edges) / utils.land_area_km()))  # mean edges at a vertex)
    table_row_names.append("Edges (edges per km^2)")


def vertex_count(vertices, edges, table_data, table_row_names, render_params ):
    table_data.append("%d" % len(vertices))  # mean edges at a vertex)
    table_row_names.append("Number of vertices")

    table_data.append("%.2f" % (len(vertices) / utils.land_area_km()) )   # mean edges at a vertex)
    table_row_names.append("Vertex density (vertices per km^2)")

def total_len(vertices, edges, table_data, table_row_names, render_params ):

    total = 0.

    for e in edges:
        dist = utils.l2(e, vertices)
        total = total + dist * 0.001

    table_data.append("%.2f" % total)  # mean edges at a vertex)
    table_row_names.append("Total length of streets (km)")

    table_data.append("%.2f" % (total / utils.land_area_km())) # mean edges at a vertex)
    table_row_names.append("Edge length density (km per km^2)")


def edge_length(vertices, edges, table_data, table_row_names, render_params, minn=0, maxx=400, bins = 32, norm = True):

    out = np.zeros((bins), dtype=int)
    per_edge = np.zeros((len(edges)))

    total = 0.

    for idx, e in enumerate ( edges):

        dist = l2(e, vertices)
        per_edge[idx] = dist

        total = total + dist

        dist = min(maxx, dist)
        dist = max(minn, dist)

        idx = floor ( dist * bins / (maxx - minn) )
        idx = min(idx, bins-1)
        out[idx] = out[idx] + 1

    render_params.append(dict(edge_cols=utils.norm_and_color_map(np.sqrt ( per_edge )), name="Edge Length (sqrt)"))

    table_data.append( "%.2f" % ( total / len (edges) ) ) # mean edges at a vertex)
    table_row_names.append("Mean edge length (m)")

    if norm:
        out = out / float ( len (edges) )

    return out



def plot_edge_length(all_city_stat, name, fig, subplots, subplot_idx, minn=0, maxx=400, bins = 32):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text(name)

    axs.spines['top'].set_color('lightgray')
    axs.spines['right'].set_color('lightgray')


    for idx, r in enumerate ( all_city_stat ):
        x_pos = np.arange(bins)
        x_lab = list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        plt.ylabel("Proportion")
        plt.xlabel("Edge length (m)")

        utils.plot (r, plt, bins, idx, all_city_stat)

        x_pos = x_pos[::2]
        x_lab = x_lab[::2]

        x_lab[len(x_lab) - 1] = "> %d" % maxx
        x_pos[len(x_pos) - 1] = bins -1

        plt.xticks(x_pos, x_lab)

        #plt.xticks(x_pos[::2], x_lab[::2])


def edge_angle(vertices, edges, table_data, table_row_names, render_params, bins = 18, norm = True ):

    out = np.zeros ( (bins), dtype=float )
    per_edge = np.zeros(len(edges))

    total = 0.

    for e_idx, e in enumerate ( edges ):
        a = np.array(vertices[e[0]])
        b = np.array(vertices[e[1]])

        d = b-a

        angle = math.atan2(d[1], d[0])
        if angle < 0:
            angle += np.pi

        per_edge[e_idx] = angle

        idx = floor(angle * bins / 3.141 )
        idx = min(idx, bins - 1)

        l = np.linalg.norm(a - b)
        total = total + l

        out[idx] = out[idx] + l

    render_params.append(dict(edge_cols=utils.cyclic_color_map( np.minimum ( 3.141, per_edge) ), name="Edge angle"))

    #table_data[table_row, table_col] = np.argmax(out) * 3.141 / bins

    am = np.argmax(out)
    table_data.append("%d" % (am * 180. / bins ) ) # mean edges at a vertex)
    table_row_names.append("Argmax edge angle (degrees)")

    if norm:
        out = out / float ( out[am] )

    return out

def plot_edge_angle(all_city_stat, name, fig, subplots, subplot_idx, bins = 18):

    axs = plt.subplot(subplots, 1, subplot_idx, polar=True)
    axs.title.set_text("Fraction of Street Length by Angle (normalised by per-graph max)")
    axs.set_theta_zero_location("N")
    axs.set_theta_direction(-1)
    axs.set_yticks([])


    #axs.set_theta_offset(pi)

    for idx, r in enumerate ( all_city_stat ):

        x_pos = np.arange(2*bins+1)
        x_pos = x_pos * np.pi / bins #list(map ( lambda x : x * np.pi / bins, x_pos) )
        #x_lab = list ( map (lambda x : " %d" % ((maxx-minn)*x/bins + minn), x_pos ) )

        # plt.ylabel("frequency")
        # plt.xlabel("edge length (m)")

        data = np.concatenate((r, r, [ r[0] ] ) ) # add point to close the loop

        #plt.bar(x_pos + (idx * (1. / len(all_city_stat))), np.concatenate((r, r)), width=np.pi / (bins * len(all_city_stat)), color=COLORS[idx])  # , edgecolor="white"
        plt.plot(x_pos, data, color=COLORS[idx%len(COLORS)])  # , edgecolor="white"
        #np.arange(2 * bins) + idx * (1. / len(all_city_stat))
        # plt.xticks(x_pos[::2], x_lab[::2])

def node_degree(vertices, edges, table_data, table_row_names, render_params, maxx = 5, norm = True ):

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


    table_data.append( "%.2f" % (len (edges) * 2 / len (vertices) ) ) # mean edges at a vertex)
    table_row_names.append("Mean node degree")

    table_data.append( "%d" % out[0])  # mean edges at a vertex)
    table_row_names.append("Number of dead ends")

    if norm:
        out = out / float(total)

    return out

def plot_node_degree(all_city_stat, name, fig, subplots, subplot_idx, maxx = 5):

    axs = plt.subplot(subplots, 1, subplot_idx)
    axs.title.set_text(name)
    axs.spines['top'].set_color('lightgray')
    axs.spines['right'].set_color('lightgray')

    for idx, r in enumerate ( all_city_stat ):
        x_pos = list(range(maxx-1))
        x_lab = list(range(1,maxx))

        plt.ylabel("Proportion")
        plt.xlabel("Node degree")

        cw = 1/float(len (all_city_stat) + 1 )
        plt.bar(np.arange(maxx) + idx*cw - 0.5 + cw, r, width= 1. / (len (all_city_stat)+1), color=COLORS[idx%len(COLORS)])

        plt.xticks(x_pos, x_lab)
        # axis.plot( x_pos, r, 'tab:orange')

def land_water_ratio(land_water_map):
    water_map_range = np.max(land_water_map) - np.min(land_water_map)
    if water_map_range == 0:
        water_map_range = 1.0
    land_water_map = (land_water_map - np.min(land_water_map)) / water_map_range
    return 1-land_water_map.mean()

def main(scale_to_meters = 1, do_render=False):

    builtins.MAP_SIZE_M = scale_to_meters

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    #plt.style.use('ggplot')
    os.makedirs("big_maps", exist_ok=True)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    npz_file_names = [x for x in os.listdir(input_path) if x.endswith('.npz')]

    if len(npz_file_names) == 0:
        print("no npz files found in " + input_path)
        return

    metric_fns = [
                   'land', 'edge_count', 'edge_length', 'total_len', 'vertex_count',
                   'segment_length', 'edge_angle', 'node_degree', 'segment_circuity',
                   'block_perimeter', 'block_area', 'block_aspect',
                   # slow ones:
                   'transport_ratio' , #'betweenness_centrality',
                   'pagerank', 'pagerank_on_edges'
                   ]

    all_city_stats = {}
    for m in metric_fns:
        all_city_stats[m] = []

    #table_data = np.zeros((len(npz_file_names),len ( metric_fns)) )

    table_row_names = []
    table_data = []

    render_all = True

    if not render_all:
        render_params = []

    for idx, npz in enumerate(npz_file_names):

        if render_all:
            render_params = []

        print(f'{idx}/{len(npz_file_names)} : {npz}')

        reset_seg_cache()
        reset_block_cache()
        reset_graph_cache()
        utils.reset_watermap()

        npz_path = os.path.join(input_path, npz)
        np_file_content = np.load(npz_path)

        #scale
        if 'tile_width_height_mercator_meters' in np_file_content:
            print ("overriding scale with merator value...")
            scale_to_meters = np_file_content['tile_width_height_mercator_meters'][0]/2


        vertices = np_file_content['tile_graph_v']

        edges = np_file_content['tile_graph_e']

        if npz.endswith("gt.npz"):
            new_edges=[]
            for e in edges:
                if e[0] == -1 or e[1] == -1:
                    pass
                else:
                    new_edges.append(e)

            print ("removed -1 -1 %d/%d edges " % ( len(edges) - len(new_edges), len(edges)) )
            edges = np.array(new_edges, dtype=int)

            new_verts=[]
            for v in vertices:
                if v[0] == -1 and v[1] == -1:
                    pass
                else:
                    new_verts.append(v)

            print ("removed -1 -1 %d/%d verts " % ( len(vertices) - len(new_edges), len(new_edges)) )
            vertices = np.array(new_verts)

        # edges on generated need filtering for short diagonals
        if npz.endswith("generated.npz"):

            one = 2 / 4028.
            new_edges=[]

            v2e = VertexMap(vertices, edges)

            def matching_endpoints(e, v2e):
                for ai in v2e.get_other_pts(e[1]):
                    if ai != e[0]:
                        for bi in v2e.get_other_pts(e[0]):
                            if bi != e[1]:
                                if ai == bi:
                                    return True
                return False

            for e in edges:
                a = vertices[e[0]]
                b = vertices[e[1]]
                if abs(abs(a[0] - b[0]) - one) < 1e-5 and \
                   abs ( abs(a[1] - b[1]) - one) < 1e-5 and \
                   matching_endpoints(e, v2e):
                    pass
                else:
                    new_edges.append(e)

            print ("filtered %d/%d edges " % ( len(edges) - len(new_edges), len(edges)) )
            edges = np.array(new_edges, dtype=int)

        if 'land_and_water_map' in np_file_content:
            builtins.WATER_MAP = np_file_content['land_and_water_map']
            builtins.LAND_RATIO = land_water_ratio( builtins.WATER_MAP )
        else:
            builtins.LAND_RATIO = 1 # everything is land

        vertices[:, [0, 1]] = vertices[:, [1, 0]] # flip for ever-changing convention
        vertices = vertices * scale_to_meters  # distances in meters

        td = []
        table_data.append(td)
        table_row_names = [] # only last iteration used!

        if render_all:
            builtins.RENDERS = []

        for m_idx, m in enumerate(metric_fns):
            print(f'   {m_idx}/{len(metric_fns)} : {m}')
            all_city_stats[m].append(globals()[m](vertices, edges, td, table_row_names, render_params ))

        if render_all:
            try:
                FastPlot(2048, 2048, vertices, edges, scale=2000. / scale_to_meters, water_map=utils.built_opengl_watermap_texture(), draw_verts=False, render_params= render_params).run()
            except Exception as e:
                 print ("pyglet has experienced an error. it often does.")
                 print(traceback.format_exc())
            renders = builtins.RENDERS
            os.makedirs("big_maps", exist_ok=True)
            for render in renders:
                PIL.Image.fromarray(render[1]).save("big_maps/"+ npz + render[0]+".png")

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
    table_strs = [[ "%s" % y for y in x] for x in np.transpose(table_data)]
    metric_names = list ( map (lambda m : m.replace("_", " "), metric_fns))

    utils.write_latex_table (table_strs, npz_file_names, table_row_names, 'big_maps/table.tex')

    tab = axs.table(cellText=table_strs, colLabels=npz_file_names, rowLabels=table_row_names, loc='center', cellLoc='center')
    tab.auto_set_column_width(col=list(range(len(npz_file_names))))
    tab.set_fontsize(8)
    for i in range (len (npz_file_names)):
        #tab[(0, i)].get_text().set_color(COLORS[i])
        tab[(0, i)].set_facecolor(COLORS[i%(len(COLORS))])

    subplot_pos = 0
    for idx, m in enumerate(metric_fns):
        name = 'plot_'+m
        if name in globals():
            globals()[name](all_city_stats[m], metric_names[idx].title(), fig, subplot_count, subplot_pos+2)
            subplot_pos = subplot_pos + 1

    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig("big_maps/stats.svg")
    plt.show()

    if len(npz_file_names) == 1 and do_render:

        if False: # render and quit
            try:
                FastPlot(2048, 2048, vertices, edges, scale=2000. / scale_to_meters, water_map=utils.built_opengl_watermap_texture(), draw_verts=False, render_params= render_params).run()
            except Exception as e:
                print ("fast plot experienced an error")
                print (e)
        else: # interactive
            FastPlot(2048, 2048, vertices, edges, scale=2000. / scale_to_meters, water_map=utils.built_opengl_watermap_texture(), draw_verts=False ).run()

        renders = builtins.RENDERS

        for render in renders:
            PIL.Image.fromarray(render[1]).save("big_maps/"+ npz_file_names[0]+"_"+render[0]+".png")

        if False: # show as massive matplot lib figure. waste. of. keystrokes.

            dpi = 1 / plt.rcParams['figure.dpi']  # pixel in inches
            fig = plt.figure(figsize=((2048 + 50) * dpi, (len(renders) ) * (2048 + 45) * dpi), frameon=False)

            plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=None, hspace=None)
            plt.tight_layout()

            for subplot_idx, render in enumerate ( renders ):
                axs = plt.subplot(len(renders), 1, subplot_idx+1)
                axs.title.set_text(render[0])
                axs.axis('off')
                axs.set_xticklabels([])
                axs.set_yticklabels([])
                axs.imshow(render[1], interpolation="nearest")
                axs.set_xlim(0.0, 2048 )
                axs.set_ylim(2048, 0.0 )

            plt.show()




if __name__ == '__main__':
    # Input Directory holding npz files
    sys.argv.append(r'npz')

    # Output Directory to save the images
    sys.argv.append(r'npz\img')

    # dxf_to_npz("C:\\Users\\twak\\Documents\\CityEngine\\Default Workspace\\datatest\\data\\dxf_streets_1.dxf", 20000, "test.npz")

    main(scale_to_meters=19567, do_render=True)



