import os
import sys

import numpy as np
import pyglet
from pyglet.gl import *
from ctypes import pointer, sizeof

# Zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1/ZOOM_IN_FACTOR

class FastPlot(pyglet.window.Window):

    def __init__(self, width, height, verts, edges, vert_cols = None, edge_cols=None, *args, **kwargs):
        conf = Config(sample_buffers=1,
                      samples=4,
                      depth_size=16,
                      double_buffer=True)
        super().__init__(width, height, config=conf,*args, **kwargs)

        #Initialize camera values
        self.left   = -width/2
        self.right  = width/2
        self.bottom = -height/2
        self.top    = height/2
        self.zoom_level = 1
        self.zoomed_width  = width
        self.zoomed_height = height
        self.verts = verts
        self.edges = edges
        self.vert_cols = vert_cols
        self.edge_cols = edge_cols

    def init_gl(self, width, height):
        # Set clear color
        glClearColor(0/255, 0/255, 0/255, 0/255)

        # Set antialiasing
        glEnable( GL_LINE_SMOOTH )
        glEnable( GL_POLYGON_SMOOTH )
        glHint( GL_LINE_SMOOTH_HINT, GL_NICEST )

        # Set alpha blending
        glEnable( GL_BLEND )
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )

        # Set viewport
        glViewport( 0, 0, width, height )

        scale = 1000

        self.vbo_points = GLuint()
        glGenBuffers(1, pointer(self.vbo_points))
        self.points_data = self.verts.flatten() * scale
        data = (GLfloat * len(self.points_data))(*self.points_data)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_points)
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

        self.vbo_point_colors = GLuint()
        glGenBuffers(1, pointer(self.vbo_point_colors))
        if self.vert_cols == None:
            self.point_color_data = (np.zeros( ( len(self.verts), 3) ) + 1).flatten()
        else:
            self.point_color_data = self.vert_cols.flatten()
        data = (GLfloat * len(self.point_color_data))(*self.point_color_data)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_point_colors)
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

        self.vbo_lines = GLuint()
        glGenBuffers(1, pointer(self.vbo_lines))
        self.lines_data = self.verts[self.edges].flatten() * scale
        data = (GLfloat * len(self.lines_data))(*self.lines_data)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

        self.vbo_line_colors = GLuint()
        glGenBuffers(1, pointer(self.vbo_line_colors))

        if self.edge_cols == None:
            self.line_color_data = (np.zeros( ( len(self.edges) * 2, 3) ) + 1).flatten()  # .astype( np.int )# np.dtype('B'))
        else:
            if len(self.edge_cols) == len (self.edges):
                self.point_color_data = np.repeat ( self.edges_cols, 2 ).flatten() # repeat for start, end of line colors
            else:
                self.point_color_data = self.edge_cols.flatten()
        #self.line_color_data = ((np.random.rand ( len (self.edges * 2), 3 ) )).flatten() #.astype( np.int )# np.dtype('B'))
        data = (GLfloat * len(self.line_color_data))(*self.line_color_data)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_line_colors)
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)


        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)


    def on_resize(self, width, height):
        # Set window values
        self.width  = width
        self.height = height
        # Initialize OpenGL context
        self.init_gl(width, height)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # Move camera
        self.left   -= dx*self.zoom_level
        self.right  -= dx*self.zoom_level
        self.bottom -= dy*self.zoom_level
        self.top    -= dy*self.zoom_level

    def on_mouse_scroll(self, x, y, dx, dy):
        # Get scale factor
        f = ZOOM_IN_FACTOR if dy < 0 else ZOOM_OUT_FACTOR if dy > 0 else 1
        # If zoom_level is in the proper range
        if .02 < self.zoom_level*f < 5:

            self.zoom_level *= f

            mouse_x = x/self.width
            mouse_y = y/self.height

            mouse_x_in_world = self.left   + mouse_x*self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y*self.zoomed_height

            self.zoomed_width  *= f
            self.zoomed_height *= f

            self.left   = mouse_x_in_world - mouse_x*self.zoomed_width
            self.right  = mouse_x_in_world + (1 - mouse_x)*self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y*self.zoomed_height
            self.top    = mouse_y_in_world + (1 - mouse_y)*self.zoomed_height

    def on_draw(self):
        # Initialize Projection matrix
        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()

        # Initialize Modelview matrix
        glMatrixMode( GL_MODELVIEW )
        glLoadIdentity()
        # Save the default modelview matrix
        glPushMatrix()

        # Clear window with ClearColor
        glClear( GL_COLOR_BUFFER_BIT )

        # Set orthographic projection matrix
        glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )

        glColor3f(1., 1., 1.)
        glPointSize( 2000 / self.zoomed_width)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
        glVertexPointer(2, GL_FLOAT, 0, 0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_line_colors)
        glColorPointer(3, GL_FLOAT, 0, 0)
        glDrawArrays(GL_LINES, 0, len(self.lines_data) // 2)


        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_point_colors)
        glColorPointer(3, GL_FLOAT, 0, 0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_points)
        glVertexPointer(2, GL_FLOAT, 0, 0)
        glDrawArrays(GL_POINTS, 0, len(self.points_data) // 2)

        # Remove default modelview matrix
        glPopMatrix()

    def run(self):
        pyglet.app.run()


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

        if hasattr(np_file_content, 'land_and_water_map'):
            land_and_water = np_file_content['land_and_water_map']
        else:
            land_and_water = None

        FastPlot(2048, 2048, vertices, edges).run()

        break


if __name__ == '__main__':
    # Input Directory holding npz files
    sys.argv.append(r'npz')

    # Output Directory to save the images
    sys.argv.append(r'npz\img')

    main()