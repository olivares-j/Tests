import sys
import numpy as np
import scipy.stats as st
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt

n = 100000
limits = [-60,60]

#--- original distribution -------
dist = st.cauchy()
sample = dist.rvs(size=n).reshape(-1, 1)
x = np.linspace(limits[0],limits[1],1000)
y = dist.pdf(x)
#---------------------------------

#----- Gaussian Mixture -------------------
components = [4,5,6,7,8,9,10,11,12,13]
models = []
bics = []
for k in components:
	gmm = GMM(n_components=k,n_init=10).fit(sample)
	bics.append(gmm.bic(sample))
	models.append(gmm)

best = models[np.argmin(bics)]
y_fit = np.exp(best.score_samples(x.reshape(-1, 1)))
print(best)
#------------------------------------------

#-------- Plots ----------------------------
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2,figsize=(12, 6))

ax0.plot(components,bics,label="BIC")
ax0.legend()
ax0.set_xlabel("Components")
ax0.set_yscale("log")


ax1.hist(sample,range=limits,bins=100,density=True,log=True,label="Samples")
ax1.plot(x,y,color="black",label="Cauchy")
ax1.plot(x,y_fit,color="red",label="Fit")

for k,(w,mu,sd) in enumerate(zip(best.weights_,best.means_,best.covariances_)):
	gauss = w*st.norm(loc=mu[0],scale=np.sqrt(sd[0][0])).pdf(x).flatten()
	ax1.plot(x,gauss,linestyle="--",label="Component "+str(k+1))
ax1.set_xlim(limits)
ax1.legend(fontsize="small")
ax1.set_ylabel("Density")
ax1.set_yscale("log")
ax1.set_ylim(top=5e-1,bottom=1e-5)
plt.show()


