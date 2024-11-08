import numpy as np

class Polynome2D:
    def __init__(self, coefficients, degree):
        """
        Constructeur de la classe Polynome2D.
        
        :param coefficients: Liste de coefficients du polynôme, chaque coefficient étant un dictionnaire
                             avec les puissances des variables x et y comme clés, et la valeur comme coefficient.
        :param degree: Degré maximal du polynôme pour les variables x et y.
        """
        self.degree = degree  # Degré maximal du polynôme
        self.coefficients = self._validate_coefficients(coefficients)  # Validation des coefficients

    def _validate_coefficients(self, coefficients):
        """
        Valide les coefficients pour s'assurer que chaque terme respecte le degré maximal.
        
        :param coefficients: Liste de tuples (coefficient, (deg_x, deg_y))
        :return: Liste de coefficients validés
        :raise ValueError: Si un terme a un degré supérieur au degré maximal.
        """
        for coef, (deg_x, deg_y) in coefficients:
            if deg_x > self.degree or deg_y > self.degree:
                raise ValueError(f"Degré ({deg_x}, {deg_y}) dépasse le degré maximal {self.degree}")
        return coefficients

    def eval(self, x, y):
        """
        Évalue le polynôme en (x, y).

        :param x: Valeur de x
        :param y: Valeur de y
        :return: Valeur du polynôme en (x, y)
        """
        result = 0
        for coef, (deg_x, deg_y) in self.coefficients:
            result += coef * (x ** deg_x) * (y ** deg_y)
        return result

    def gradient(self, x, y):
        """
        Calcule le gradient du polynôme en (x, y).

        :param x: Valeur de x
        :param y: Valeur de y
        :return: Un tuple (df/dx, df/dy), où df/dx est la dérivée partielle par rapport à x
                 et df/dy est la dérivée partielle par rapport à y.
        """
        grad_x = 0
        grad_y = 0
        
        for coef, (deg_x, deg_y) in self.coefficients:
            # Dérivée par rapport à x
            if deg_x > 0:
                grad_x += coef * deg_x * (x ** (deg_x - 1)) * (y ** deg_y)
            # Dérivée par rapport à y
            if deg_y > 0:
                grad_y += coef * deg_y * (x ** deg_x) * (y ** (deg_y - 1))
        
        return grad_x, grad_y

    def __add__(self, other):
        """
        Additionne deux polynômes.
        
        :param other: Un autre polynôme de même degré
        :return: Un nouveau polynôme résultant de l'addition
        """
        if self.degree != other.degree:
            raise ValueError("Les polynômes doivent avoir le même degré pour être additionnés.")
        
        # On combine les coefficients des termes communs
        new_coeffs = self.coefficients.copy()
        
        for coef, (deg_x, deg_y) in other.coefficients:
            found = False
            for i, (coef_self, (deg_x_self, deg_y_self)) in enumerate(new_coeffs):
                if deg_x_self == deg_x and deg_y_self == deg_y:
                    new_coeffs[i] = (coef_self + coef, (deg_x_self, deg_y_self))
                    found = True
                    break
            if not found:
                new_coeffs.append((coef, (deg_x, deg_y)))
        
        return Polynome2D(new_coeffs, self.degree)

    def __sub__(self, other):
        """
        Soustrait deux polynômes.
        
        :param other: Un autre polynôme de même degré
        :return: Un nouveau polynôme résultant de la soustraction
        """
        if self.degree != other.degree:
            raise ValueError("Les polynômes doivent avoir le même degré pour être soustraits.")
        
        # On combine les coefficients des termes communs
        new_coeffs = self.coefficients.copy()
        
        for coef, (deg_x, deg_y) in other.coefficients:
            found = False
            for i, (coef_self, (deg_x_self, deg_y_self)) in enumerate(new_coeffs):
                if deg_x_self == deg_x and deg_y_self == deg_y:
                    new_coeffs[i] = (coef_self - coef, (deg_x_self, deg_y_self))
                    found = True
                    break
            if not found:
                new_coeffs.append((-coef, (deg_x, deg_y)))
        
        return Polynome2D(new_coeffs, self.degree)

    def __mul__(self, other):
        """
        Multiplie deux polynômes.
        
        :param other: Un autre polynôme de même degré
        :return: Un nouveau polynôme résultant de la multiplication
        """
        if self.degree != other.degree:
            raise ValueError("Les polynômes doivent avoir le même degré pour être multipliés.")
        
        # On multiplie les termes de chaque polynôme
        new_coeffs = []
        
        for coef1, (deg_x1, deg_y1) in self.coefficients:
            for coef2, (deg_x2, deg_y2) in other.coefficients:
                new_coef = coef1 * coef2
                new_deg_x = deg_x1 + deg_x2
                new_deg_y = deg_y1 + deg_y2
                
                # Si le degré total dépasse le degré maximal, on ignore ce terme
                if new_deg_x <= self.degree and new_deg_y <= self.degree:
                    found = False
                    for i, (coef, (deg_x, deg_y)) in enumerate(new_coeffs):
                        if deg_x == new_deg_x and deg_y == new_deg_y:
                            new_coeffs[i] = (coef + new_coef, (deg_x, deg_y))
                            found = True
                            break
                    if not found:
                        new_coeffs.append((new_coef, (new_deg_x, new_deg_y)))
        
        return Polynome2D(new_coeffs, self.degree)

    def __repr__(self):
        """
        Représentation textuelle du polynôme pour l'affichage.
        """
        terms = []
        for coef, (deg_x, deg_y) in sorted(self.coefficients, key=lambda x: (x[1][0], x[1][1])):
            if coef != 0:
                term = f"{coef} * x^{deg_x} * y^{deg_y}"
                terms.append(term)
        return " + ".join(terms) if terms else "0"

    def integrer_x(self, a, b, y_value=0):
        """
        Intègre le polynôme par rapport à x sur l'intervalle [a, b], avec y fixé.
        
        :param a: Borne inférieure de l'intégrale
        :param b: Borne supérieure de l'intégrale
        :param y_value: La valeur de y, par défaut 0
        :return: L'intégrale de chaque terme du polynôme sur [a, b] par rapport à x
        """
        integral = 0
        for coef, (deg_x, deg_y) in self.coefficients:
            if deg_x >= 0:
                if deg_x != -1:
                    # Intégration par rapport à x de a à b
                    integral += coef * y_value**deg_y * (b**(deg_x + 1) - a**(deg_x + 1)) / (deg_x + 1)
                else:
                    # Cas particulier pour x^(-1)
                    integral += coef * y_value**deg_y * np.log(b/a)
        return integral

    def integrer_y(self, c, d, x_value=0):
        """
        Intègre le polynôme par rapport à y sur l'intervalle [c, d], avec x fixé.
        
        :param c: Borne inférieure de l'intégrale pour y
        :param d: Borne supérieure de l'intégrale pour y
        :param x_value: La valeur de x, par défaut 0
        :return: L'intégrale de chaque terme du polynôme sur [c, d] par rapport à y
        """
        integral = 0
        for coef, (deg_x, deg_y) in self.coefficients:
            if deg_y >= 0:
                if deg_y != -1:
                    # Intégration par rapport à y de c à d
                    integral += coef * x_value**deg_x * (d**(deg_y + 1) - c**(deg_y + 1)) / (deg_y + 1)
                else:
                    # Cas particulier pour y^(-1)
                    integral += coef * x_value**deg_x * np.log(d/c)
        return integral


# Exemple d'utilisation
coeffs1 = [
    (3, (2, 0)),  # 3 * x^2
    (-4, (0, 1)),  # -4 * y^1
    (5, (1, 1)),  # 5 * x^1 * y^1
]

poly = Polynome2D(coeffs1, degree=2)

# Calcul de l'intégrale par rapport à x sur l'intervalle [1, 2] avec y=0
integral_result_x = poly.integrer_x(1, 2, y_value=0)
print("Intégrale par rapport à x sur [1, 2]:", integral_result_x)

# Calcul de l'intégrale par rapport à y sur l'intervalle [0, 1] avec x=1
integral_result_y = poly.integrer_y(0, 1, x_value=1)
print("Intégrale par rapport à y sur [0, 1]:", integral_result_y)

