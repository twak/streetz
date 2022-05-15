
import numpy as np

def l2(e, vertices):
    a = np.array(vertices[e[0]])
    b = np.array(vertices[e[1]])
    return np.linalg.norm(a - b)

COLORS = ['#8eff98', '#ffcf8e', '#8ee6ff', '#fff58e', '#ff918e']