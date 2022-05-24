import math

import numpy as np
import builtins
import PIL

import matplotlib.pyplot as plt


def l2 ( e, vertices ):

    a = np.array(vertices[e[0]])
    b = np.array(vertices[e[1]])

    return np.linalg.norm(a - b)

def l2p (ai, bi, vertices):

    a = np.array(vertices[ai])
    b = np.array(vertices[bi])

    return np.linalg.norm(a - b)

def angle(ai, bi, ci, vertices):
    a = vertices[ai]
    b = vertices[bi]
    c = vertices[ci]

    ab = b-a
    bc = c-b

    # https://stackoverflow.com/a/16544330/708802
    dot = ab[0] * bc[0] + ab[1] * bc[1]  # dot product between [x1, y1] and [x2, y2]
    det = ab[0] * bc[1] - ab[1] * bc[0]  # determinant
    out = math.atan2(det, dot)

    return out

def area (ai, bi, ci, vertices):

    a = vertices[ai]
    b = vertices[bi]
    c = vertices[ci]

    # x = b-a
    # y = c-b
    #
    # dot = x[0] * x[1] + y[0] * y[1]  # dot product between [x1, y1] and [x2, y2]
    # det = x[0] * y[1] - y[0] * x[1]  # determinant
    # out = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos) https://stackoverflow.com/a/16544330/708802
    # if out < 0:
    #     out = np.pi *2 + out
    #
    # return out

    return 0.5 * ( (b[0]-a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]) ) # https://math.stackexchange.com/a/1414021

def plot (r, plt, bins, idx, all_city_stat):

    if False: # bars
        cw = 1 / float(len(all_city_stat) + 1)
        plt.bar(np.arange(bins) + idx * cw - 0.5 + cw, r, 1. / (len(all_city_stat) + 1), color=COLORS[idx])
    else: # lines
        plt.plot(np.arange(bins), r, lw=2, color=COLORS[idx%len(COLORS)])

# multicols COLORS = ['#8ee6ff', '#8eff98', '#ffcf8e', , '#fff58e', '#ff918e']

# blue vs yellows
COLORS = ['#04c7ff', '#ff9955', '#ff6755', '#ffd655', '#fffd8c']


def write_latex_table(table_strs, npz_file_names, table_row_names, file):

    with open(file, 'w') as f:
        f.write('\\begin{center}\n')
        f.write('\\begin{tabular}  { |c|')
        f.write('c|' * len ( npz_file_names) )
        f.write('} \n')
        f.write('\\hline\n')

        first_line = "&" + " & ".join ( npz_file_names )
        f.write(first_line + "\\\\ \n")
        f.write('\\hline\n')


        for idx, str in enumerate(table_strs):
            line = table_row_names[idx] + " & "
            line = line + " & ".join (str)
            f.write(line + "\\\\ \n")

        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{center}\n')


def plot_table(npz_file_names, subplot_count, table_row_names, table_strs):
    plt.subplots_adjust(wspace=0.5, hspace=1)
    axs = plt.subplot(subplot_count, 1, 1)
    axs.axis('off')
    tab = axs.table(cellText=table_strs, colLabels=npz_file_names, rowLabels=table_row_names, loc='center', cellLoc='center')
    tab.auto_set_column_width(col=list(range(len(npz_file_names))))
    tab.set_fontsize(8)
    for i in range(len(npz_file_names)):
        tab[(0, i)].set_facecolor(COLORS[i % (len(COLORS))])



def land_area_km():
    size = builtins.MAP_SIZE_M * 0.001
    return builtins.LAND_RATIO * size * size

def reset_watermap():
    global WATER_LAND
    WATER_LAND = None

WATER_LAND = None
def built_opengl_watermap_texture(): # build an array for pyglett

    global WATER_LAND

    if WATER_LAND is None:
        img = PIL.Image.open("land_water.png")
        map = np.asarray(img, dtype=int)

        if not hasattr(builtins, "WATER_MAP"):
            water_map = np.zeros((512,512,3))
            water_map [:,:] = [0,0,0]
        else:
            water_map = (builtins.WATER_MAP[::4, ::4] + 1) / 2
            # water_map = (builtins.WATER_MAP[::2, ::2] + 1) / 2
        lu = (water_map * (map.shape[1] - 1)).astype(np.int)
        WATER_LAND = map[0, lu]

    return WATER_LAND


MAGMA = None
def norm_and_color_map(values_per_edge):
    global MAGMA

    if MAGMA is None:
        img = PIL.Image.open("magma.png")
        MAGMA = np.asarray(img)/256
        MAGMA=MAGMA[:, :, :3]

    maxx = values_per_edge.max()
    minn = values_per_edge.min()

    if maxx != minn:
        norm = (values_per_edge - minn) / (maxx - minn)
    else:
        norm = np.ones_like(values_per_edge)

    lu = (norm * (MAGMA.shape[1] - 1)).astype(int)

    return maxx, minn, MAGMA[0, lu]


CYCLIC_COLS = None
def cyclic_color_map(values_per_edge):
    global CYCLIC_COLS

    if CYCLIC_COLS is None:
        img = PIL.Image.open("magma_roundy_round.png")
        CYCLIC_COLS = np.asarray(img)/256
        CYCLIC_COLS=CYCLIC_COLS[:, :, :3]

    maxx = values_per_edge.max()
    minn = values_per_edge.min()

    if maxx != minn:
        norm = (values_per_edge - minn) / (maxx - minn)
    else:
        norm = np.ones_like(values_per_edge)

    lu = (norm * (CYCLIC_COLS.shape[1] - 1)).astype(int)

    return maxx, minn, CYCLIC_COLS[0, lu]


def land_water_ratio(land_water_map):
    water_map_range = np.max(land_water_map) - np.min(land_water_map)
    if water_map_range == 0:
        water_map_range = 1.0
    land_water_map = (land_water_map - np.min(land_water_map)) / water_map_range
    return 1-land_water_map.mean()

def load_filter(np_file_content, npz, scale_to_meters = 10000):

    if 'tile_width_height_mercator_meters' in np_file_content:
        print("overriding scale with merator value...")
        scale_to_meters = np_file_content['tile_width_height_mercator_meters'][0] / 2

    vertices = np_file_content['tile_graph_v']
    edges = np_file_content['tile_graph_e']

    if npz.endswith("gt.npz"):  # we were sent a bunch of matricies padded with -1
        new_edges = []
        for e in edges:
            if e[0] == -1 or e[1] == -1:
                pass
            else:
                new_edges.append(e)

        print("removed -1 -1 %d/%d edges " % (len(edges) - len(new_edges), len(edges)))
        edges = np.array(new_edges, dtype=int)

        new_verts = []
        for v in vertices:
            if v[0] == -1 and v[1] == -1:
                pass
            else:
                new_verts.append(v)

        print("removed -1 -1 %d/%d verts " % (len(vertices) - len(new_edges), len(new_edges)))
        vertices = np.array(new_verts)

    if npz.endswith("generated.npz"): # edges on generated graphs need filtering for short diagonals

        one = 2 / 4028.
        new_edges = []

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
                    abs(abs(a[1] - b[1]) - one) < 1e-5 and \
                    matching_endpoints(e, v2e):
                pass
            else:
                new_edges.append(e)

        print("filtered %d/%d edges " % (len(edges) - len(new_edges), len(edges)))
        edges = np.array(new_edges, dtype=int)
        reset_v2e_cache()


    # def bad(a):
    #     t = 0.05
    #     return a[0] < -t or a[0] > t or a[1] < -t or a[1] > t
    #
    # new_edges = []
    #
    # for e in edges:
    #     a = vertices[e[0]]
    #     b = vertices[e[1]]
    #     if bad(a) or bad(b):
    #         pass
    #     else:
    #         new_edges.append(e)
    #
    # edges = np.array(new_edges, dtype=int)

    if 'land_and_water_map' in np_file_content:
        builtins.WATER_MAP = np_file_content['land_and_water_map']
        builtins.LAND_RATIO = land_water_ratio(builtins.WATER_MAP)
    else:
        builtins.LAND_RATIO = 1  # everything is land

    vertices[:, [0, 1]] = vertices[:, [1, 0]]  # flip for ever-changing convention
    vertices = vertices * scale_to_meters  # distances in meters

    return vertices, edges, scale_to_meters



V2E = None

class VertexMap ():

    def __init__(s, vertices, edges):
        s.v2e = {}
        s.v2ei = {}
        s.v = vertices
        s.e = edges

        for v in range(len ( vertices) ):
            s.v2e [v] = []
            s.v2ei[v] = []

        for idx, e in enumerate(edges):
            # sk = vertices[e[0]].tobytes()
            # ek = vertices[e[1]].tobytes()

            s.v2e[e[0]].append(e)
            s.v2e[e[1]].append(e)

            s.v2ei[e[0]].append(idx)
            s.v2ei[e[1]].append(idx)

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

def reset_v2e_cache():
    global V2E
    V2E = None