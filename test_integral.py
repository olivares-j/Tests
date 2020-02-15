import sys
import numpy as np
import scipy.stats as st
import scipy.integrate as integrate

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#--------- Bivariate normal--------
BVN = st.multivariate_normal(mean=[0,0],cov=[[1,0.5],[0.5,1]])

#--------- Univariate normal------
UVN = st.norm(loc=0,scale=1)


N = 20
dom = np.array([-10,10])
x  = np.linspace(dom[0],dom[1],N)
y  = np.column_stack((x,np.zeros_like(x)))


result = integrate.quad(lambda z:BVN.pdf(np.column_stack((z,np.zeros_like(z)))), dom[0], dom[1])


dx = np.float((dom[1]-dom[0])/(N-1))
prob = np.ones_like(x)

myres = np.dot(prob,BVN.pdf(y))*dx

print("True: {0}".format(UVN.pdf(0)))
print("Quad: {0}".format(result))
print("Mine: {0}".format(myres))
print("Difference {0}".format(myres-result[0]))
sys.exit()

#--------- Plots ---------
file_plot = "Marginalization.pdf"
pdf = PdfPages(filename=file_plot)
fig, axes = plt.subplots(num=0,figsize=(6,6))
plt.plot(x, BVN.pdf(y),label="MVN(x|y=0)")
plt.plot(x, UVN.pdf(x),label="N(x)")
plt.plot(x, np.sqrt(2*np.pi)*BVN.pdf(y),label="$\sqrt{2\pi}$MVN(x|y=0)",linestyle="--")
plt.xlabel("X")
plt.ylabel("Density")
plt.xlim(dom)
plt.legend()
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()

pdf.close()