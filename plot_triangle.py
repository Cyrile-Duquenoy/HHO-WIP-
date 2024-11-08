import numpy as np
import matplotlib.pyplot as plt
from Point import Point
from Triangle import Triangle
from matplotlib.tri import Triangulation

def plot_triangle(triangle:Triangle):
    triang = [[triangle.p1.x,triangle.p2.x,triangle.p3.x,triangle.p1.x],[triangle.p1.y,triangle.p2.y,triangle.p3.y,triangle.p1.y]]
    plt.plot(triang[0],triang[1])
