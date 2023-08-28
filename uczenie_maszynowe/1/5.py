# Operacja A**-1 dla obiektów typu `matrix` są określony w sposób właściwy dla macierzy, a dla
# obiektów typu `array` w sposób elementowy, czyli element po elemencie

# obiekty macierzowe mają .I dla odwrotności, ale obiekty ndarray nie


import numpy as np
a = np.matrix([[1, 2],
              [3, 4]])
print(a)
print('\nInverse')
print(a.I)

b = np.array([[1, 2],
             [3, 4]])
print(b)
print('\nInverse')
print(b.I)