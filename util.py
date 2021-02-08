from time import time
from functools import wraps
from tqdm import tqdm
from random import randrange as rand
import math

class Vec:
    def __init__(self, a, b=None):
        self.x, self.y = a if b is None else (a, b)
    
    def __add__(self, other):
        return Vec(self.x+other.x, self.y+other.y)
    
    def __iadd__(self, other):
        self.x+=other.x
        self.y+=other.y
        return self
    
    def __sub__(self, other):
        return Vec(self.x-other.x, self.y-other.y)
    
    def __neg__(self):
        return Vec(-self.x, -self.y)
    
    def mag(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def limit(self, r):
        m = self.mag()
        if m > r:
            self.x *= r/m
            self.y *= r/m
    
    def __eq__(self, other):
        return other is not None and self.x == other.x and self.y == other.y
    
    def copy(self):
        return Vec(self.x, self.y)
    
    def __repr__(self):
        return f"({self.x}, {self.y})"
    
    def __lt__(self, other):
        return self.x<other.x or (self.x==other.x and self.y < other.y)
    
    def __hash__(self):
        return hash((self.x, self.y))