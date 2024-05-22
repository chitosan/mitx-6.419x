import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import fisher_exact
from scipy.stats import hypergeom
from scipy.stats import norm


def pmfBinomial (n,k,p):
    c = sc.special.comb(n,k)
    pk = c * p**k * (1-p)**(n-k)
    print("P(X=%i) = " %k, pk)
    x = [i for i in range(0, k*3)]
    pmfBinomial = [sc.special.comb(n,xi) * p**xi * (1-p)**(n-xi) for xi in x]
    return x, pmfBinomial

def pdfBinomial (xList, yList, xLimit):
    yList = yList[:xLimit]
    xList = xList[:xLimit]
    result = sum(yList)
    pdfSum = 0
    pdfBinomial = []
    for yi in yList:
        pdfSum = pdfSum + yi
        pdfBinomial.append(pdfSum)
    print(result)
    return xList, pdfBinomial

def pdfPoisson(x):
    # lambda = x
    return x

def pmfPoisson(x):
    # receives the pdf set x as X,P(X)
    return x    


n = 31000
k1 = 63
k2 = 39
p1 = k1/n
p2 = k2/n

x1, result1 = pmfBinomial(n, k1, p1)
x2, result2 = pmfBinomial(n, k2, p2)

x3, density3 = pdfBinomial(x1,result1,100)
x4, density4 = pdfBinomial(x2,result2,100)

plt.plot(x1, result1,  color="black", linestyle="dashed", linewidth=1.0)
plt.plot(x2, result2,  color="blue" , linestyle="dashed", linewidth=1.0)
plt.plot(x3, density3, color="black", linestyle="dashed", linewidth=1.0)
plt.plot(x4, density4, color="blue" , linestyle="dashed", linewidth=1.0)
plt.show()

res = fisher_exact([[39, 63], [30961, 30937]], alternative='less')
print(res.statistic)
print(res.pvalue)
print(norm.cdf(-3.0268))







'''
df1 = pd.DataFrame([x1, result1])
df2 = pd.DataFrame([x2, result2])
df3 = pd.DataFrame([x3, density3])
df4 = pd.DataFrame([x4, density4])
df1 = df1.transpose()
df2 = df2.transpose()
df3 = df3.transpose()
df4 = df4.transpose()
df1.columns=['x','P']
df2.columns=['x','P']
df3.columns=['x','D']
df4.columns=['x','D']
df = pd.concat([df1, df2, df3,df4], axis=1)
df.to_excel('pdfBinary1.xlsx')
'''


