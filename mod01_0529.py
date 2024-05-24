import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.stats as st

def bvar(x):
    N = len(x)
    u = np.average(x)
    d = abs(x - u) ** 2.
    var = N * np.average(d) / (N - 1)
    return var

def cov(x, y):
    N = len(x)
    ux = np.average(x)
    uy = np.average(y)
    dx = abs(x - ux)
    dy = abs(y - uy)
    var = N * np.average(dx*dy) / (N - 1)
    return var


Xs = np.array([0.0339, 0.0423, 0.213, 0.257, 0.273, 0.273, 0.450, 0.503, 0.503, \
               0.637, 0.805, 0.904, 0.904, 0.910, 0.910, 1.02, 1.11, 1.11, 1.41, \
               1.72, 2.03, 2.02, 2.02, 2.02])
Ys = np.array([-19.3, 30.4, 38.7, 5.52, -33.1, -77.3, 398.0, 406.0, 436.0, 320.0, 373.0,\
               93.9, 210.0, 423.0, 594.0, 829.0, 718.0, 561.0, 608.0, 1.04E3, 1.10E3,\
               840.0, 801.0, 519.0])
N = 24
# Averages
ux = np.average(Xs)
uy = np.average(Ys)
# Variance
s2x = bvar(Xs)
s2y = bvar(Ys)
# Std. Deviation
stx = s2x**0.5
sty = s2y**0.5
# Covariance (Xs, Ys)
cxy = cov(Xs,Ys)

print("E X_, Y_ : ",ux,uy)
print("Var X, Y : ",s2x,s2y)
print("Std X, Y : ",stx,sty)
print("Cov(X, Y): ",cxy)

plt.scatter(Xs, Ys)
plt.show()
