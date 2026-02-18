import numpy as np
import os

path = "data/BCICIV_2a_1.csv"
print("Checking:", path)
data = np.loadtxt(path, delimiter=',')
print("Shape:", data.shape)

