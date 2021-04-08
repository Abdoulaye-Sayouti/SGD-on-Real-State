import numpy as np
from scipy.optimize import minimize
import csv
from numpy import linalg as LA


def Init(fichier):
    """ Construction des données A et b en fonction d'un fichier d'entrée donné en argument """
    with open(fichier, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        k = 1
        for row in spamreader:
            if k == 1:
                k = k + 1
                continue
            A.append([float(row[0].split(',')[1]), float(row[0].split(',')[2]),float(row[0].split(',')[3]),
                      float(row[0].split(',')[4]), float(row[0].split(',')[5]), float(row[0].split(',')[6])])
            b.append(float(row[0].split(',')[7]))

""" La fonction erreur E(w) """
def E(w):
    return LA.norm(np.matmul(A, np.array(w)) - b)



""" Calcul du gradient pour une seul donnée du corpus  qui est g'(w) = -2A[i][:](b[i] - w.T*A[i][:])"""
def gradient(w):
    """Calcul du gradient de E"""
    return  -2*np.array(A)[i,:]*(b[i] - np.matmul(np.array(w).transpose(),np.array(A)[i,:]))



A = []
b = []

""" Initialisation de w0 """
w0 = np.zeros(6)
Init('Real estate.csv')

# Transformation des listes A et b en objets utilsables par NumPy
A = np.array(A)
b = np.array(b)

print("X = ", A)
print("Y = ", b)

""" choix du de la donnée à utiliser pour calculer le gradient """
i = 80
wapprox = wapprox = minimize(E, w0, method='CG', jac=gradient, options={'gtol': 1e-7})
wapprox = wapprox.x

# On affiche ici ce vecteur
print("W = ",wapprox)
# On affiche ici l'erreur correspondante
print("E =",E(wapprox))
