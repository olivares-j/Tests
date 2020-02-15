import sys
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

alphas = [2,3,4,5,6,7,8,9,10]
beta   = 10

n_samples = 1000000
n_bins=1000

plt.figure()
for alpha in alphas:
	y  = st.beta.rvs(a=alpha,b=beta,size=n_samples)
	x  = 0.5*np.sqrt(2)*y
	rt = np.sqrt(x**(-2) - 1.)


	# plt.hist(y,bins=n_bins, histtype="step",density=True,label="Y")
	# plt.hist(x,bins=n_bins, histtype="step",density=True,label="X")
	plt.hist(rt,bins=n_bins,histtype="step",density=True,label="alpha:"+str(alpha))
plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.show()