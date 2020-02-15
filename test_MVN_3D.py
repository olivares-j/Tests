import sys
import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


locs = np.arange(10,150,5)
scls = np.arange(5,55,5)
N    = 100000

bias = np.zeros((len(scls),len(locs),2))

for i,scl in enumerate(scls):
	for j,loc in enumerate(locs):

		mu = np.array([loc,0,0])
		sd = np.repeat(scl,3)

		sample = st.multivariate_normal(mean=mu,cov=np.diag(sd**2)).rvs(size=N)

		distances = np.sqrt(np.sum(sample**2,axis=1))

		bias[i,j,0] = (np.mean(distances) - loc)/loc
		bias[i,j,1] = (np.std(distances) - scl)/scl

#--------- Plots ---------
pdf = PdfPages(filename="Bias.pdf")
plt.figure(0,figsize=(6,6))
plt.suptitle("Distance bias to nearby associations due to projection effects.")
for i,scl in enumerate(scls):
	plt.plot(locs,bias[i,:,0],linestyle="-",label=str(scl))
	# plt.plot(locs,bias[i,:,1],linestyle="--",label=str(scl))
plt.xlabel("Distance [pc]")
plt.ylabel("Fractional error (O-T)/T")
plt.ylim(0,1)
plt.legend(title="Scale [pc]")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()
sys.exit()

loc = 100.
scl = 10.
mu  = np.array([loc,0,0])
sd  = np.repeat(scl,3)

sample = st.multivariate_normal(mean=mu,cov=np.diag(sd**2)).rvs(size=N)

distances = np.sqrt(np.sum(sample**2,axis=1))

r = st.norm.rvs(size=(N,3))

A = np.eye(3)*scl


X = np.array([0.0,loc,0.0]) + np.matmul(r,A)

d = np.sqrt(np.sum(X**2,axis=1))

x = np.linspace(loc-5*scl,loc+5*scl,1000)
plt.figure(0,figsize=(6,6))
plt.suptitle("Solar association.")
plt.hist(distances,bins=100,density=True,histtype="step",label="MVN("+str(loc)+","+str(scl)+")")
plt.hist(d,bins=100,density=True,histtype="step",label="Transformed")
plt.xlabel("Distance [pc]")
plt.ylabel("Density")
plt.legend()
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close(0)
pdf.close()