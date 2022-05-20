import os
import sys

import matplotlib
import numpy as np
from PIL import Image, ImageColor

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

def get_water_color_map():
    gmaps_land_rgb = np.asarray(ImageColor.getcolor(OSM_LAND, "RGB")) / 255
    gmaps_water = np.asarray(ImageColor.getcolor(OSM_WATER, "RGB")) / 255

    cdict2 = \
    {
        'red':   [(0.0, gmaps_land_rgb[0], gmaps_land_rgb[0]),
                    (1.0,  gmaps_water[0], gmaps_water[0])],
        'green': [(0.0, gmaps_land_rgb[1], gmaps_land_rgb[1]),
                    (1.0,  gmaps_water[1], gmaps_water[1])],
        'blue':  [(0.0, gmaps_land_rgb[2], gmaps_land_rgb[2]),
                    (1.0,  gmaps_water[2], gmaps_water[2])]
    }

    return matplotlib.colors.LinearSegmentedColormap('my_colormap2', cdict2, 256)

def plot_batch(decoded, np_result, sample_idx, sample_dir):
    b = decoded.shape[0]

    water_colormap = get_water_color_map()

    max_distance = np.sqrt(2 * (255 ** 2))

    p1_p2_color = '#f0d6a5'
    p3_color = '#ffffff'

    p1_p2_color = np.asarray(ImageColor.getrgb(p1_p2_color))[np.newaxis].astype(np.float32) / 255.0
    p3_color = np.asarray(ImageColor.getrgb(p3_color))[np.newaxis].astype(np.float32) / 255.0

    dist_thresh = 1.5

    for i in range(b):
        water_img = water_colormap(np_result['land_and_water_map'][i + sample_idx])[..., :3] # Throw away Alpha channel

        gt_p1 = np_result['tile_distance_to_closest_street_normalized_priority_1'][i + sample_idx]
        gt_p2 = np_result['tile_distance_to_closest_street_normalized_priority_2'][i + sample_idx]
        gt_p3 = np_result['tile_distance_to_closest_street_normalized_priority_3'][i + sample_idx]

        gt_p1_p2 = np.clip(np.min(np.stack([gt_p1, gt_p2], axis = 0), axis = 0) * max_distance, 0, 255) < dist_thresh
        gt_p3 = np.clip(gt_p3 * max_distance, 0, 255) < dist_thresh

        ground_truth = water_img.copy()
        ground_truth[gt_p1_p2] = p1_p2_color
        inference_result = ground_truth.copy()

        ground_truth[gt_p3] = p3_color
        Image.fromarray((255 * ground_truth).astype(np.uint8)).save(os.path.join(sample_dir, '{:03d}_ground_truth.png'.format(i + sample_idx)))

        result_p3 = np.clip(decoded[i] * max_distance, 0, 255) < dist_thresh
        inference_result[result_p3] = p3_color
        Image.fromarray((255 * inference_result).astype(np.uint8)).save(os.path.join(sample_dir, '{:03d}_inference_result.png'.format(i + sample_idx)))


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
        edges = np_file_content['tile_graph_e']

        if 'land_and_water_map' in np_file_content:
            land_and_water = np_file_content['land_and_water_map']
        else:
            land_and_water = None

        img_file_path = os.path.join(output_path, npz + ".png")
        plot_batch(vertices, edges, land_and_water, img_file_path)


if __name__ == '__main__':
    # Input Directory holding npz files
    sys.argv.append(r'npz')

    # Output Directory to save the images
    sys.argv.append(r'npz\img')

    # dxf_to_npz("C:\\Users\\twak\\Documents\\CityEngine\\Default Workspace\\datatest\\data\\dxf_streets_10.dxf", 20000, "four_blocks.npz")

    main()