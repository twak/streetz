import math

import numpy as np


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


COLORS = ['#8eff98', '#ffcf8e', '#8ee6ff', '#fff58e', '#ff918e']