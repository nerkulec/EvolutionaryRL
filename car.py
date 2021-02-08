from util import Vec
import numpy as np

oil_sigma = 0.2

max_vel = 0.2
max_acc = 0.02

class Car:
    def __init__(self, pos=None):
        if pos is None:
            self.pos = Vec(1.5, 1.5)
        else:
            self.pos = pos
        self.vel = Vec(0, 0)
    
    def update(self, acc, field):
        acc = Vec(acc)
        acc.limit(max_acc)
        if field == 'O':
            acc += Vec(np.random.normal(scale=oil_sigma, size=2))
        self.vel += acc
        self.vel.limit(max_vel)
        self.pos += self.vel
        