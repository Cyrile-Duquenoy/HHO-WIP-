import math
from Point import Point

class Triangle:
    def __init__(self, p1, p2, p3):
        """
        Initialisation du triangle avec trois points.
        Chaque point doit être une instance de la classe Point.
        """
        if not all(isinstance(p, Point) for p in [p1, p2, p3]):
            raise TypeError("Les trois arguments doivent être des objets de type Point.")
        
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        
        self.faces=((self.p1,self.p2),(self.p1,self.p3),(self.p2,self.p3))
        
        
        # Vérifier si les trois points forment un triangle valide
        if not self.is_valid():
            raise ValueError("Les trois points ne forment pas un triangle valide (points colinéaires). Le point est sur une face.")
    
    
        
    
    def __repr__(self):
        """
        Représentation du triangle sous forme de chaîne de caractères.
        """
        return f"Triangle({self.p1}, {self.p2}, {self.p3})"
    
    def is_valid(self):
        """
        Vérifie si le triangle est valide en vérifiant que les points ne sont pas colinéaires.
        Cela se fait en vérifiant si l'aire du triangle est non nulle.
        """
        # Utilisation de la méthode du déterminant pour vérifier la colinéarité des points
        area = self.area()
        return area > 0
    
    def distance(self, p1, p2):
        """
        Calcul de la distance entre deux points p1 et p2.
        """
        return p1.distance(p2)
    
    def perimeter(self):
        """
        Calcul du périmètre du triangle en sommant les longueurs des côtés.
        """
        a = self.distance(self.p1, self.p2)
        b = self.distance(self.p2, self.p3)
        c = self.distance(self.p3, self.p1)
        return a + b + c
    
    def area(self):
        """
        Calcul de l'aire du triangle en utilisant la formule de l'aire avec les coordonnées des points :
        Aire = 1/2 * |x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)|
        """
        x1, y1 = self.p1.x, self.p1.y
        x2, y2 = self.p2.x, self.p2.y
        x3, y3 = self.p3.x, self.p3.y
        return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    def centroid(self):
        """
        Calcul du centre de gravité (centroïde) du triangle.
        Le centroïde est donné par la moyenne des coordonnées des trois sommets.
        """
        x = (self.p1.x + self.p2.x + self.p3.x) / 3
        y = (self.p1.y + self.p2.y + self.p3.y) / 3
        return Point(x, y)
    
    def contains_point(self, p):
        """
        Vérifie si un point p est à l'intérieur du triangle en utilisant la méthode de l'aire.
        Un point est à l'intérieur si la somme des aires des triangles formés avec ce point et
        les côtés du triangle initial est égale à l'aire du triangle initial.
        """
        area_original = self.area()
        area1 = Triangle(p, self.p1, self.p2).area()
        area2 = Triangle(p, self.p2, self.p3).area()
        area3 = Triangle(p, self.p3, self.p1).area()
        
        # Vérifier si la somme des aires des trois triangles est égale à l'aire du triangle principal
        return math.isclose(area_original, area1 + area2 + area3)
    
# Exemple d'utilisation
if __name__ == "__main__":
    # Création de trois points
    p1 = Point(0, 0)
    p2 = Point(4, 0)
    p3 = Point(2, 3)

    # Création du triangle
    try:
        triangle = Triangle(p1, p2, p3)
        print(f"Triangle : {triangle}")
        
        # Calcul du périmètre et de l'aire
        print(f"Périmètre du triangle : {triangle.perimeter():.2f}")
        print(f"Aire du triangle : {triangle.area():.2f}")
        
        # Calcul du centroïde
        centroid = triangle.centroid()
        print(f"Centroïde du triangle : {centroid}")
        
        # Vérification si un point est à l'intérieur du triangle
        p_inside = Point(2, 1)
        print(f"Le point {p_inside} est-il à l'intérieur du triangle ? {triangle.contains_point(p_inside)}")
        
        # Vérification si les trois points forment un triangle valide
        invalid_triangle = Triangle(p1, p2, Point(2, 0))  # Triangle invalide
    except ValueError as e:
        print(e)


