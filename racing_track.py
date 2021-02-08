import math
import numpy as np
import pyglet
from pyglet import shapes
from util import Vec

tile_width = 40

class Map:
    def __init__(self, file_name = 'maps/map1.txt'):
        board = []
        with open(file_name, 'r') as f:
            for line in f:
                board.append([c for c in line[:-1]])
        self.width = len(board[0])
        self.height = len(board)
        self.board = board
        for y in range(self.height):
            for x in range(self.width):
                v = Vec(x, y)
                if self[v] == 'M':
                    self.M = v + Vec(0.5, 0.5)

    def __getitem__(self, pos):
        try:
            return self.board[math.floor(pos.y)][math.floor(pos.x)]
        except:
            return '#'

    def __setitem__(self, pos, val):
        self.board[pos.y][pos.x] = val

    def __contains__(self, pos):
        return 0 <= pos.x < self.width and\
               0 <= pos.y < self.height and\
               self[pos] != '#'
               
    def get_closest(self, pos, field = '#', jump = 0.2, num_steps = 11, rays = 12, batch = None):
        if batch is not None:
            self.rays = []
        closest = np.zeros(rays)
        for i in range(rays):
            c = math.cos(i*2*math.pi/rays)
            s = math.sin(i*2*math.pi/rays)
            for j in range(1, num_steps+1):
                ray = pos + Vec(c*j*jump, s*j*jump)
                closest[i] = j
                if self[ray] == field:
                    break
            if batch is not None:
                coef = math.floor(255*(num_steps-j)/(num_steps-1))
                self.rays.append(shapes.Line(
                    pos.x*tile_width, pos.y*tile_width, ray.x*tile_width, ray.y*tile_width, 1,
                    color=(coef, (255-coef)//2, 0), batch=batch))
        return (num_steps-closest)/(num_steps-1)

    def print(self):
        for y in range(self.height):
            for x in range(self.width):
                print(self.board[y][x], end='')
            print()
    
    def draw(self, batch):
        self.rects = []
        for y in range(self.height):
            for x in range(self.width):
                g = self[Vec(x, y)]
                if g == ' ':
                    c = (0, 0, 0)
                elif g == '#':
                    c = (128, 128, 128)
                elif g == 'C':
                    c = (140, 120, 120)
                elif g == 'O':
                    c = (40, 26, 13)
                elif g == 'M':
                    c = (128, 128, 256)
                rect = shapes.Rectangle(x*tile_width, y*tile_width, tile_width, tile_width, color=c, batch=batch)
                self.rects.append(rect)
