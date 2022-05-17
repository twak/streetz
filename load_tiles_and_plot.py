import os
import sys

import io

import ezdxf as ezdxf
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageColor

def hex_to_rgb(hex):
    return np.asarray(ImageColor.getcolor(hex, "RGB")) / 255

def rgb_to_hex(rgb):
    rgb = (255 * rgb).astype(np.int32)
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def lighten_hex_color(hex, perc):
    rgb = hex_to_rgb(hex)
    perc = perc / 100
    return rgb_to_hex(perc + ((1 - perc) * rgb))

def darken_hex_color(hex, fac):
    return rgb_to_hex(hex_to_rgb(hex) * fac)

# MAP colors
OSM_WATER = '#C4DFF6' # '#aac0f8'
OSM_LAND = darken_hex_color('#FCFBE7', 0.25) # '#f1eddf'
OSM_GRASS = '#E6F2C1'
OSM_PARK = '#DAF2C1'

MIN_RANGE = -1.0
MAX_RANGE = 1.0

def set_aspect_and_range(axs, x_range, y_range):
    axs.set_aspect('equal', 'box')
    axs.axis('off')
    axs.xaxis.set_visible(False)
    axs.yaxis.set_visible(False)

    axs.set_xlim(x_range)
    axs.set_ylim(y_range)

def plot_street_graph(v, e, axs):
    from_coord = v[e[:, 0]]
    to_coord = v[e[:, 1]]

    x_from, y_from, x_to, y_to = from_coord[:, 0], from_coord[:, 1], to_coord[:, 0], to_coord[:, 1]

    v_color = '#FFFFFF'
    e_color = '#EEEEEE'

    v_size = 0.125
    e_width = 0.125

    p_from = np.stack([x_from, y_from], axis = 1)
    p_to = np.stack([x_to, y_to], axis = 1)
    all_p = np.concatenate([p_from, p_to], axis = 0)
    p_unique = np.unique(all_p, axis = 0)

    axs.scatter(p_unique[:, 0], p_unique[:, 1], color = v_color, s = v_size, zorder = 0.0, linewidths = 0.0)

    # Plot Ground Truth Edges
    X = np.stack([x_from, x_to], axis = 0)
    Y = np.stack([y_from, y_to], axis = 0)

    axs.plot(X, Y, color = e_color, zorder = -1.0, linewidth = e_width, solid_capstyle = 'round')

def plot_land_water_map(land_water_map, axs):
    gmaps_land_rgb = np.asarray([1,1,1]) #  ImageColor.getcolor(OSM_LAND, "RGB")) / 255
    gmaps_water = np.asarray([0,0,0]) #ImageColor.getcolor(OSM_WATER, "RGB")) / 255

    cdict2 = \
    {
        'red':   [(0.0, gmaps_land_rgb[0], gmaps_land_rgb[0]),
                    (1.0,  gmaps_water[0], gmaps_water[0])],
        'green': [(0.0, gmaps_land_rgb[1], gmaps_land_rgb[1]),
                    (1.0,  gmaps_water[1], gmaps_water[1])],
        'blue':  [(0.0, gmaps_land_rgb[2], gmaps_land_rgb[2]),
                    (1.0,  gmaps_water[2], gmaps_water[2])]
    }

    cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap2', cdict2, 256)
    water_map_range = np.max(land_water_map) - np.min(land_water_map)
    if water_map_range == 0:
        water_map_range = 1.0
    land_water_map = (land_water_map - np.min(land_water_map)) / water_map_range

    im_extent = (MIN_RANGE, MAX_RANGE, MIN_RANGE, MAX_RANGE)
    axs.imshow(land_water_map, extent = im_extent, cmap = cmap, zorder = -3)

def plot(v, e, land_and_water, img_file_path):
    fig, axs = plt.subplots(1, 1, facecolor = 'black')

    plot_street_graph(v, e, axs)
    if land_and_water is not None:
        plot_land_water_map(land_and_water, axs)

    set_aspect_and_range(axs, x_range = [MIN_RANGE, MAX_RANGE], y_range = [MIN_RANGE, MAX_RANGE])
    
    fig.savefig(img_file_path, format = "png", dpi = 600, bbox_inches = 'tight', pad_inches = 0, facecolor = fig.get_facecolor(), edgecolor = 'none')

    plt.close(fig)


def dxf_to_npz(dxf, scale, outfile):

    doc = ezdxf.readfile(dxf)
    msp = doc.modelspace()

    pos = []
    idx = []
    pos_to_ind = {}

    def add(pt):
        pt = (pt[0], pt[1]) # drop z

        if pt in pos_to_ind:
            return pos_to_ind[pt]
        else:
            pos.append(np.array(pt))
            i = len(pos)-1
            pos_to_ind[pt] = i
            return i

    for line in msp:
        if line.dxftype() == "LINE":
            s =  line.dxf.start
            e = line.dxf.end

            si = add(s)
            ei = add(e)

            idx.append([add(s), add(e)])

            print("start point: %s\n" % line.dxf.start)
            print("end point: %s\n" % line.dxf.end)

    print(f"parsed DXF with edges {len(idx)}  pts {len(pos)}\n")

    vertices = np.array( pos, dtype=np.float64)
    edges    = np.array( idx, dtype=np.int32  )

    # range = max ( vertices[:, 0].max() - vertices[:, 0].min(), vertices[:, 0].max() - vertices[:, 0].min() )

    vertices /= scale

    # print(str(vertices.shape))
    # print(str(edges.shape))
    # (10222, 3)
    # (15853, 2)

    np.savez(outfile, tile_graph_v=vertices, tile_graph_e=edges)

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    npz_file_names = [x for x in os.listdir(input_path) if x.endswith('.npz')]

    for idx, npz in enumerate(npz_file_names):

        print(npz)

        npz_path = os.path.join(input_path, npz)

        np_file_content = np.load(npz_path)

        # Vertices
        vertices = np_file_content['tile_graph_v']
        edges    = np_file_content['tile_graph_e']

        print(str(vertices.shape))
        print(str(edges.shape))

        drawing = ezdxf.new(dxfversion='AC1024')
        modelspace = drawing.modelspace()

        for j in range (0, edges.shape[0]):
            start = vertices[edges[j][0]] * 20000
            end   = vertices[edges[j][1]] * 20000
            modelspace.add_line( (start[0],start[1]), (end[0], end[1]), dxfattribs={'color': 7})

        drawing.saveas( os.path.join( f'{output_path}\\{idx}.dxf' ) )

        if 'land_and_water_map' in np_file_content:
            land_and_water = np_file_content['land_and_water_map']
        else:
            land_and_water = None

        img_file_path = os.path.join(output_path, npz+".png")
        plot(vertices, edges, land_and_water, img_file_path)

if __name__ == '__main__':
    # Input Directory holding npz files
    sys.argv.append(r'npz')
    
    # Output Directory to save the images
    sys.argv.append(r'npz\img')

    dxf_to_npz("C:\\Users\\twak\\Documents\\CityEngine\\Default Workspace\\datatest\\data\\three_to_two_rect.dxf", 20000, "seg.npz")

    #main()
