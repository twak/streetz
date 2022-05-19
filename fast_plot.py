import builtins
import os
import sys

import PIL
import numpy as np
import pyglet
from pyglet.gl import *
from ctypes import pointer, sizeof
from pyglet import image

# Zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1/ZOOM_IN_FACTOR

class FastPlot(pyglet.window.Window): #https://stackoverflow.com/a/19453006/708802

    def __init__(self, width, height, verts, edges, water_map=None, draw_verts=True, vert_cols = None, edge_cols=None, render_params=None, scale = 1000, *args, **kwargs):
        conf = Config(sample_buffers=1,
                      samples=4,
                      depth_size=16,
                      double_buffer=True)
        super().__init__(width, height, config=conf,*args, **kwargs)

        #Initialize camera values
        # self.left = -width / 2
        # self.right = width / 2
        # self.bottom = -height / 2
        # self.top = height / 2

        self.left = -2000
        self.right = 2000
        self.bottom = -2000
        self.top = 2000

        self.zoom_level = 1
        # self.zoomed_width = width
        # self.zoomed_height = height
        self.zoomed_width = 4000
        self.zoomed_height = 4000
        self.verts = verts
        self.edges = edges
        self.vert_cols = vert_cols
        self.edge_cols = edge_cols
        self.scale = scale
        # self.init_gl(width, height)
        self.water_map = water_map

        self.draw_verts = draw_verts
        builtins.RENDERS = [] # output renders go here when done
        self.render_name = "?!" # key/title for each render
        self.render_params = render_params
        self.quit_on_empty_render_params = render_params is not None
        self.block_pts = None # we don't always know where blocks are
        self.block_cols = None
        self.draw_blocks = False

        pyglet.clock.schedule_interval(self.update, 1 / 30.0)

        # self.pic = image.load('magma.png')

    def update(self, dt):
        pass

    def init_gl(self, width, height):
        # Set clear color
        glClearColor(255./255, 195./255, 72./255, 255./255)

        # Set antialiasing
        glEnable( GL_LINE_SMOOTH )
        glEnable( GL_POLYGON_SMOOTH )
        glHint( GL_LINE_SMOOTH_HINT, GL_NICEST )

        # Set alpha blending
        glEnable( GL_BLEND )
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )

        # Set viewport
        glViewport( 0, 0, width, height )


        self.vbo_points = GLuint()
        glGenBuffers(1, pointer(self.vbo_points))
        self.points_data = self.verts.flatten() * self.scale
        data = (GLfloat * len(self.points_data))(*self.points_data)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_points)
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

        self.vbo_point_colors = GLuint()
        glGenBuffers(1, pointer(self.vbo_point_colors))
        if self.vert_cols is None:
            self.point_color_data = (np.zeros( ( len(self.verts), 3) ) + 1).flatten()
        else:
            self.point_color_data = self.vert_cols.flatten()
        data = (GLfloat * len(self.point_color_data))(*self.point_color_data)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_point_colors)
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

        self.vbo_blocks = GLuint()
        glGenBuffers(1, pointer(self.vbo_blocks))
        self.vbo_block_colors = GLuint()
        glGenBuffers(1, pointer(self.vbo_block_colors))

        self.vbo_lines = GLuint()
        glGenBuffers(1, pointer(self.vbo_lines))
        self.lines_data = self.verts[self.edges].flatten() * self.scale
        data = (GLfloat * len(self.lines_data))(*self.lines_data)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

        self.vbo_line_colors = GLuint()
        glGenBuffers(1, pointer(self.vbo_line_colors))

        # default is white line colors
        self.line_color_data = (np.zeros( ( len(self.edges) * 2, 3) ) + 0.77).flatten()  # .astype( np.int )# np.dtype('B'))
        data = (GLfloat * len(self.line_color_data))(*self.line_color_data)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_line_colors)
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

        img = np.asarray(PIL.Image.open("magma.png"), dtype=int)

        tex_data = (GLubyte * img.size)(*img.astype('uint8').flatten())

        img = pyglet.image.ImageData(
            img.shape[1],
            img.shape[0],
            "RGBA",
            tex_data,
            pitch=img.shape[0] * 4 * 1
        )

        textureIDs = (pyglet.gl.GLuint * 1)()
        glGenTextures(1, textureIDs)
        self.key_tex_id = textureIDs[0]
        glBindTexture(GL_TEXTURE_2D, self.key_tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (img.width), (img.height), 0, GL_RGBA, GL_UNSIGNED_BYTE, img.get_data())

        if self.water_map is not None:
            img = self.water_map

            tex_data = (GLubyte * img.size)(*img.astype('uint8').flatten())

            img = pyglet.image.ImageData(
                    img.shape[0],
                    img.shape[1],
                    "RGBA",
                    tex_data,
                    pitch = img.shape[ 1 ] * 4 * 1
                )
            textureIDs = (pyglet.gl.GLuint * 1)()
            glGenTextures(1, textureIDs)
            self.water_map_id = textureIDs[0]
            glBindTexture(GL_TEXTURE_2D, self.water_map_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (img.width), (img.height), 0, GL_RGBA, GL_UNSIGNED_BYTE, img.get_data())

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

    def setup_render(self, param):

        self.draw_blocks = False

        if "edge_cols" in param:
            self.edge_cols = param["edge_cols"]
        else:
            self.edge_cols = (np.zeros( ( len(self.edges) * 2, 3) ) +  + 0.77)

        if "block_pts" in param:
            if self.block_pts is not None and len(param["block_pts"] != len(self.block_pts)):
                raise RuntimeError("different length block-pts provided :(")

            self.block_pts = param["block_pts"]
            self.block_pts =  self.block_pts * self.scale
            self.block_number = self.block_pts.shape[0]
        else:
            self.block_pts = None

        if "block_cols" in param:

            self.draw_blocks = True
            self.block_cols = param["block_cols"]

            if self.vbo_blocks is None:
                raise RuntimeError("block_cols without block_pts")

            self.block_cols = param["block_cols"]
        else:
            self.block_cols = None

        self.render_name = param["name"]

    def on_draw(self):

        if len(self.render_params) > 0:
            self.setup_render ( self.render_params.pop() )
            do_capture = True
        else:
            do_capture = False
            if self.quit_on_empty_render_params:
                pyglet.close()

        # Clear window with ClearColor
        glClear( GL_COLOR_BUFFER_BIT )


        # Initialize Projection matrix
        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()

        # Initialize Modelview matrix
        glMatrixMode( GL_MODELVIEW )
        glLoadIdentity()
        # Save the default modelview matrix
        glPushMatrix()


        # Set orthographic projection matrix
        glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )

        # land / water background
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.water_map_id)
        glBegin(GL_QUADS)
        width  = 2000
        glTexCoord2i(0,1 )
        glVertex2i(-width, -width)
        glTexCoord2i(1, 1)
        glVertex2i(width, -width)
        glTexCoord2i(1, 0)
        glVertex2i(width, width)
        glTexCoord2i(0, 0)
        glVertex2i(-width, width)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        glColor3f(1., 1., 1.)
        glPointSize( 10000 / self.zoomed_width)

        if self.edge_cols is not None:

            cols = self.edge_cols

            if len(cols) == len (self.edges):
                self.line_color_data = np.repeat (cols, 2, axis=0 ).flatten() # repeat for start, end of line colors
            else:
                self.line_color_data = cols.flatten()

            self.edge_cols = None
            do_capture = True

            data = (GLfloat * len(self.line_color_data))(*self.line_color_data)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_line_colors)
            glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
        glVertexPointer(2, GL_FLOAT, 0, 0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_line_colors)
        glColorPointer(3, GL_FLOAT, 0, 0)
        glDrawArrays(GL_LINES, 0, len(self.lines_data) // 2)

        if self.draw_verts:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_point_colors)
            glColorPointer(3, GL_FLOAT, 0, 0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_points)
            glVertexPointer(2, GL_FLOAT, 0, 0)
            glDrawArrays(GL_POINTS, 0, len(self.points_data) // 2)



        if self.block_cols is not None:

            self.block_color_data = self.block_cols[2].flatten()
            data = (GLfloat * len(self.block_color_data))(*self.block_color_data)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_block_colors)
            glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

            self.block_pts_data = self.block_pts.flatten()
            data = (GLfloat * len(self.block_pts_data))(*self.block_pts_data)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_blocks)
            glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

            self.block_cols = None
            self.block_pts = None

            do_capture = True

        if self.draw_blocks:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_block_colors)
            glColorPointer(3, GL_FLOAT, 0, 0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_blocks)
            glVertexPointer(2, GL_FLOAT, 0, 0)
            glDrawArrays(GL_POINTS, 0, self.block_number)

        # color key
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.key_tex_id)
        glBegin(GL_QUADS)
        ypos = -2000
        xpos = 1600
        glTexCoord2i(0, 0)
        glVertex2i(xpos, ypos)
        glTexCoord2i(1, 0)
        glVertex2i(xpos+400, ypos)
        glTexCoord2i(1, 1)
        glVertex2i(xpos+400, ypos+100)
        glTexCoord2i(0, 1)
        glVertex2i(xpos, ypos+100)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        # Remove default modelview matrix
        glPopMatrix()
        # global IMG_OUT

        if do_capture:
            ibar = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            out = np.asanyarray(ibar.get_data()).reshape(self.width, self.height, 4)
            builtins.RENDERS.append((self.render_name, out))

        # print("quitting")
        # pyglet.app.exit()

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

        FastPlot(2048, 2048, vertices, edges ).run()
        #FastPlot(2048, 2048, vertices, edges, visible=False).run()

        break

# IMG_OUT = None

if __name__ == '__main__':
    # Input Directory holding npz files
    sys.argv.append(r'npz')

    # Output Directory to save the images
    sys.argv.append(r'npz\img')

    main()
    # print("main done" + str(IMG_OUT))
