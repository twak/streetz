import math

import numpy as np
import builtins
import PIL

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

    return math.atan2(c[1] - a[1], c[0] - a[0]) - math.atan2(b[1] - a[1], b[0] - a[0]) #https://stackoverflow.com/a/31334882/708802

def area (ai, bi, ci, vertices):

    a = vertices[ai]
    b = vertices[bi]
    c = vertices[ci]

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

def land_area_km():
    size = builtins.MAP_SIZE_M * 0.001
    return builtins.LAND_RATIO * size * size

WATER_LAND = None
def built_opengl_watermap_texture(): # build an array for pyglett

    global WATER_LAND

    if WATER_LAND == None:
        img = PIL.Image.open("land_water.png")
        map = np.asarray(img, dtype=int)
        water_map = (builtins.WATER_MAP[::4, ::4] + 1)/2
        lu = (water_map * (map.shape[1] - 1)).astype(np.int)
        WATER_LAND = map[0, lu]

    return WATER_LAND


