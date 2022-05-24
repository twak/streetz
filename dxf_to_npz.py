import os
import ezdxf as ezdxf
import numpy as np

import utils


def npz_to_dxf(input_path, output_path):

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    npz_file_names = [x for x in os.listdir(input_path) if x.endswith('.npz')]


    for idx, npz in enumerate(npz_file_names):

        print(npz)

        npz_path = os.path.join(input_path, npz)

        np_file_content = np.load(npz_path)

        vertices, edges, scale_to_meters = utils.load_filter(np_file_content, npz, 19567)

        print(str(vertices.shape))
        print(str(edges.shape))

        drawing = ezdxf.new(dxfversion='AC1024')
        modelspace = drawing.modelspace()

        for j in range (0, edges.shape[0]):
            start = vertices[edges[j][0]]
            end   = vertices[edges[j][1]]
            modelspace.add_line( (start[0],start[1]), (end[0], end[1]), dxfattribs={'color': 7})

        drawing.saveas( os.path.join( f'{output_path}\\{idx}.dxf' ) )


def dxf_to_npz(dxf, scale, outfile, water_from=None):

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

    hw2 = scale/ 2.

    def oob(pt):

        return pt[0] < -hw2 or pt[0] > hw2 or pt[1] < -hw2 or pt[1] > hw2

    for line in msp:
        if line.dxftype() == "LINE":
            s = line.dxf.start
            e = line.dxf.end

            if oob(s) or oob(e):
                continue

            idx.append([add(s), add(e)])
            # print("start point: %s\n" % line.dxf.start)
            # print("end point: %s\n" % line.dxf.end)

    print(f"parsed DXF with edges {len(idx)}  pts {len(pos)}\n")

    vertices  = np.array(pos, dtype=np.float64)
    edges     = np.array(idx, dtype=np.int32)
    vertices /= hw2

    vertices[:, [0, 1]] = vertices[:, [1, 0]] # keep flipping axes it looks right...

    params = dict(tile_graph_v=vertices, tile_graph_e=edges)

    if water_from is not None:
        params['land_and_water_map'] = np.load(water_from)['land_and_water_map']

    np.savez(outfile, **params)


if __name__ == '__main__':

    # this is the importer used to get cityengine street graphs into our npz format
    # you must remove the final two lines of cityengine dxf files for the importer to work :/
    # the final argument in the below imports the water and land map from another dxf

    dxf_to_npz("C:\\Users\\twak\\Documents\\CityEngine\\Default Workspace\\datatest\\data\\sf.dxf", 19567,
                "npz/sf_ce.npz", water_from='C:\\Users\\twak\\PycharmProjects\\streets2\\generated\\sanfran_generated.npz')

