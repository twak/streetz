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


input_path = 'npz'

npz_file_names = [x for x in os.listdir(input_path) if x.endswith('.npz')]

metric_fns = [
    'land', 'edge_count', 'edge_length', 'total_len', 'vertex_count',
    # 'segment_length', 'edge_angle', 'node_degree', 'segment_circuity',
    # 'block_perimeter', 'block_area', 'block_aspect',
    # slow ones:
    # 'transport_ratio' , 'betweenness_centrality', 'pagerank', 'pagerank_on_edges'
]

all_city_stats = {}
for m in metric_fns:
    all_city_stats[m] = []

# table_data = np.zeros((len(npz_file_names),len ( metric_fns)) )

table_row_names = []
table_data = []

render_params = []



for idx, npz in enumerate(npz_file_names):

    print(f'{idx}/{len(npz_file_names)} : {npz}')

    npz_path = os.path.join(input_path, npz)
    np_file_content = np.load(npz_path)

    city_count = np_file_content['tile_graph_v'].shape[1]

    dicts = []
    for i in range (city_count):
        dicts.append({})

    for f in np_file_content.files:

        for c_idx, arr in enumerate ( np_file_content[f][0] ):

            dicts[c_idx][f] = arr

    for idx_d, dict in enumerate ( dicts ):

        vertices = dict["tile_graph_v"]
        vertices[:, [0, 1]] = vertices[:, [1, 0]]
        dict["tile_graph_v"] = vertices

        np.savez (f"gts/{idx_d}.npz", **dict)