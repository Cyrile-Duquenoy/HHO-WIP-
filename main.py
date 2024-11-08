from Point import Point
from Triangle import Triangle
from plot_triangle import plot_triangle

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    P1,P2,P3,P4=Point(0,0),Point(1,0),Point(0,1),Point(1,1)
    
    T1,T2=Triangle(P1,P2,P3),Triangle(P2,P3,P4)
    
    Th=[T1,T2]    
    
    plot_triangle(T2)
    plot_triangle(T1)
    

    def f(x,y,Th:list):
        P=Point(x,y)
        if x!=y:
            if Th[1].contains_point(P):
                return -2*P.x+3
            else :
                return -4*P.x+4
        
        
    #print(f(0,1,Th))
    
    #print(T1.contains_point(Point(0.5,0.5)))
    
    print(T2.faces)