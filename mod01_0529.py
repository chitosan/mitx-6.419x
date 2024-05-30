import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.stats as st
import statsmodels.api as sm
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


def bvar(x):
    N = len(x)
    u = np.average(x)
    d = (x - u) ** 2.
    var = N * np.average(d) / (N - 1)
    return var

def cov(x, y):
    N = len(x)
    ux = np.average(x)
    uy = np.average(y)
    dx = (x - ux)
    dy = (y - uy)
    s2x = bvar(Xs)
    s2y = bvar(Ys)
    # Std. Deviation
    stdx = s2x ** 0.5
    stdy = s2y ** 0.5
    covar = N * np.average(dx*dy) / (N - 1)
    corr = covar/(stdx*stdy)
    return covar, corr

def polyreg(x, y, degree):
    results = []
    coeffs = np.polyfit(x, y, degree)
    # Polynomial Coefficients
    results = coeffs.tolist()
    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                      # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    cod = (ssreg / sstot)            # cod = Coefficient of Determination
    return results, cod, yhat              # returns a list w/ degree number polynomial, cod

def residuals(x,y,reg_eq):
    ei = []
    for xi, yi in zip(x, y):
        ei.append(yi - (reg_eq[0]*xi + reg_eq[1]))
    return ei


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
# Covariance (Xs, Ys) = cxy & Correlation (Xs,Ys) = rxy
cxy, rxy = cov(Xs,Ys)

print("E X_, Y_ : ",ux,uy)
print("Var X, Y : ",s2x,s2y)
print("Std X, Y : ",stx,sty)
print("Cov(X, Y): ",cxy)
print("Corr(X, Y): ",rxy)
print("")

# LINEAR regression
# ML regression Scikit-learn
model = LinearRegression()
Xr = Xs.reshape(-1,1)
model.fit(Xr, Ys)
r_sq = model.score(Xr, Ys)
print("LINEAR regression scikit-learn")
print(f"coefficient of determination R^2: {r_sq}")
print(f"intercept B0: {model.intercept_}")
print(f"slope     B1: {model.coef_[0]}")
print("")

# LINEAR regression scipy
slope, intercept, r, p, se = st.linregress(Xs, Ys)
print("LINEAR regression Scipy")
print("B1, B0, R^2 : ",slope,intercept,r**2)

# Math regression numpy
eq, r2, Yr = polyreg(Xs, Ys, 1)
print("")
print("POLYNOMIAL regression. My function with numpy")
print(eq,r2)
# redisuals 1st degree only
ei = residuals(Xs, Ys, eq)
eq, r2, Yr = polyreg(Xs, ei, 1)
print(eq,r2)
    
# ******************************************************

print("\n")
print(" *** NEW PROBLEM ***")

Xs = np.array([ 0.387, 0.723, 1.00, 1.52, 5.20,
               9.54, 19.2, 30.1, 39.5 ])
Ys = np.array([ 0.241, 0.615, 1.00, 1.88, 11.9, 
               29.5, 84.0, 165.0, 248 ])
N = 9

Xs = np.log(Xs)
Ys = np.log(Ys)

ux = np.average(Xs)
uy = np.average(Ys)
# Variance
s2x = bvar(Xs)
s2y = bvar(Ys)
# Std. Deviation
stx = s2x**0.5
sty = s2y**0.5
# Covariance (Xs, Ys) = cxy & Correlation (Xs,Ys) = rxy
cxy, rxy = cov(Xs,Ys)

print("E X_, Y_ : ",ux,uy)
print("Var X, Y : ",s2x,s2y)
print("Std X, Y : ",stx,sty)
print("Cov(X, Y): ",cxy)
print("Corr(X, Y): ",rxy)
print("")

# LINEAR regression
# ML regression Scikit-learn
model = LinearRegression()
Xr = Xs.reshape(-1,1)
model.fit(Xr, Ys)
r_sq = model.score(Xr, Ys)
print("LINEAR regression scikit-learn")
print(f"coefficient of determination R^2: {r_sq}")
print(f"intercept B0: {model.intercept_}")
print(f"slope     B1: {model.coef_[0]}")
print("")

# LINEAR regression scipy
slope, intercept, r, p, se = st.linregress(Xs, Ys)
print("LINEAR regression Scipy")
print("B1, B0, R^2 : ",slope,intercept,r**2)

# Math regression numpy
eq, r2, Yr = polyreg(Xs, Ys, 1)
print("")
print("POLYNOMIAL regression. My function with numpy")
print(eq,r2)
# redisuals 1st degree only
ei = residuals(Xs, Ys, eq)
#eq, r2, Yr = polyreg(Xs, ei, 1)
#print(eq,r2)
print("")
print("Solar system problem")
LogPlanetMass = np.array(
       [-0.31471074,  1.01160091,  0.58778666,  0.46373402, -0.01005034,
         0.66577598, -1.30933332, -0.37106368, -0.40047757, -0.27443685,
         1.30833282, -0.46840491, -1.91054301,  0.16551444,  0.78845736,
        -2.43041846,  0.21511138,  2.29253476, -2.05330607, -0.43078292,
        -4.98204784, -0.48776035, -1.69298258, -0.08664781, -2.28278247,
         3.30431931, -3.27016912,  1.14644962, -3.10109279, -0.61248928])

LogPlanetRadius = np.array(
       [ 0.32497786,  0.34712953,  0.14842001,  0.45742485,  0.1889661 ,
         0.06952606,  0.07696104,  0.3220835 ,  0.42918163, -0.05762911,
         0.40546511,  0.19227189, -0.16251893,  0.45107562,  0.3825376 ,
        -0.82098055,  0.10436002,  0.0295588 , -1.17921515,  0.55961579,
        -2.49253568,  0.11243543, -0.72037861,  0.36464311, -0.46203546,
         0.13976194, -2.70306266,  0.12221763, -2.41374014,  0.35627486])

LogPlanetOrbit = np.array(
       [-2.63108916, -3.89026151, -3.13752628, -2.99633245, -3.12356565,
        -2.33924908, -2.8507665 , -3.04765735, -2.84043939, -3.19004544,
        -3.14655516, -3.13729584, -3.09887303, -3.09004295, -3.16296819,
        -2.3227878 , -3.77661837, -2.52572864, -4.13641734, -3.05018846,
        -2.40141145, -3.14795149, -0.40361682, -3.2148838 , -2.74575207,
        -3.70014265, -1.98923527, -3.35440922, -1.96897409, -2.99773428])

StarMetallicity = np.array(
       [ 0.11 , -0.002, -0.4  ,  0.01 ,  0.15 ,  
         0.22 , -0.01 ,  0.02 , -0.06 , -0.127,  
         0.   ,  0.12 ,  0.27 ,  0.09 , -0.077,  
         0.3  ,  0.14 , -0.07 ,  0.19 , -0.02 ,  
         0.12 ,  0.251,  0.07 ,  0.16 ,  0.19 ,  
         0.052, -0.32 ,  0.258,  0.02 , -0.17 ])

LogStarMass = np.array(
       [ 0.27002714,  0.19144646, -0.16369609,  0.44468582,  0.19227189,
         0.01291623,  0.0861777 ,  0.1380213 ,  0.49469624, -0.43850496,
         0.54232429,  0.02469261,  0.07325046,  0.42133846,  0.2592826 ,
        -0.09431068, -0.24846136, -0.12783337, -0.07364654,  0.26159474,
         0.07603469, -0.07796154,  0.09440068,  0.07510747,  0.17395331,
         0.28893129, -0.21940057,  0.02566775, -0.09211529,  0.16551444])

LogStarAge = np.array(
       [ 1.58103844,  1.06471074,  2.39789527,  0.72754861,  0.55675456,
         1.91692261,  1.64865863,  1.38629436,  0.77472717,  1.36097655,
         0.        ,  1.80828877,  1.7837273 ,  0.64185389,  0.69813472,
         2.39789527, -0.35667494,  1.79175947,  1.90210753,  1.39624469,
         1.84054963,  2.19722458,  1.89761986,  1.84054963,  0.74193734,
         0.55961579,  1.79175947,  0.91629073,  2.17475172,  1.36097655])
N = 30

# intercept, LogPlanetRadius, LogPlanetOrbit, StarMetallicity, 
# LogStarMass, LogStarAge
intercept = np.ones(N)
X = np.column_stack((intercept,LogPlanetRadius,LogPlanetOrbit,
                      StarMetallicity,LogStarMass,LogStarAge))
Y = LogPlanetMass

X1 = np.dot(X.T, X)
X2 = np.linalg.inv(X1)
X3 = np.dot(X2, X.T)
B = np.dot(X3, Y)
print(B)



'''
#T_j test for the Solar system problem
# First, estimate the standard deviation of the noise.
sigmaHat = np.sqrt( np.sum( np.square(Ys - Xmat.dot(betaVec) )) / ( N - Xmat.shape[1] ) )
# Now estimate the (matrix part of the) covariance matrix for beta 
import numpy.linalg
betaCov = numpy.linalg.inv(Xmat.T.dot(Xmat))
# Use the formula for the t-test statistic for each variable
tVals = betaVec/(sigmaHat * np.sqrt(np.diagonal(betaCov)))
# Calculate the 2-sided p-values.
import scipy.stats
pvals = scipy.stats.t.sf(np.abs(tVals), N-Xmat.shape[1])*2
---

sc.stats.t.sf(T, num_degrees_of_freedom)
np.linalg.inv(matrix_to_invert)
sm.qqplot(Xs, line='s')
plt.show() 

plt.scatter(Xs, Ys)
plt.plot(Xs, Yr)
plt.show()
'''
