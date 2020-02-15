import sys
import numpy as np
import scipy.stats as st 
import scipy.spatial as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
matplotlib.use('Agg')


#============================= Functions ===============================================
#----------------------------------------------------------------------------------
# Obtains the width, height and angle to plot the covariance matrix
# def get_principal(sigma,idx):
# 	std_x   = np.sqrt(sigma[idx[0],idx[0]])
# 	std_y   = np.sqrt(sigma[idx[1],idx[1]])
# 	rho_xy    = sigma[idx[0],idx[1]]/(std_x*std_y)


# 	# Author: Jake VanderPlas
# 	# License: BSD
# 	# Modified by Javier Olivares
# 	# Level indicates 1,2,3 sigma interval
# 	level = 3. # 1-sigma
# 	level = 5. # 2-sigma
# 	level = 7. # 3-sigma
# 	#----------------------------------------
# 	sigma_xy2 = rho_xy * std_x * std_y

# 	alpha = 0.5 * np.arctan2(2 * sigma_xy2,(std_x ** 2 - std_y ** 2))
# 	tmp1  = 0.5 * (std_x ** 2 + std_y ** 2)
# 	tmp2  = np.sqrt(0.25 * (std_x ** 2 - std_y ** 2) ** 2 + sigma_xy2 ** 2)

# 	return level*np.sqrt(tmp1 + tmp2), level*np.sqrt(np.abs(tmp1 - tmp2)), alpha* 180. / np.pi
# #----------------------------------------------------------------------------------

def get_principal(sigma,idx):
	sigma_x   = np.sqrt(sigma[idx[0],idx[0]])
	sigma_y   = np.sqrt(sigma[idx[1],idx[1]])
	rho_xy    = sigma[idx[0],idx[1]]


	# Author: Jake VanderPlas
	# License: BSD
	#----------------------------------------
	level = 5.0 
	sigma_xy2 = rho_xy * sigma_x * sigma_y

	alpha = 0.5 * np.arctan2(2 * sigma_xy2,(sigma_x ** 2 - sigma_y ** 2))
	tmp1  = 0.5 * (sigma_x ** 2 + sigma_y ** 2)
	tmp2  = np.sqrt(0.25 * (sigma_x ** 2 - sigma_y ** 2) ** 2 + sigma_xy2 ** 2)

	return level*np.sqrt(tmp1 + tmp2), level*np.sqrt(np.abs(tmp1 - tmp2)), alpha* 180. / np.pi
#----------------------------------------------------------------------------------

N = 10000
sig  = np.array([[1.,-0.1],[-0.1,1.0]])
isig = np.linalg.inv(sig) 
zero = np.zeros(2)
quantiles = [0.6827,0.9545,0.9973]

#----- Distributions ----------------------------
mvn  = st.multivariate_normal(mean=zero,cov=sig)
chi2 = st.chi2(df=2)

#----- samples ------------------
sample = mvn.rvs(size=N)

#------- Mahalanobis --------------------
r      = np.array(list(map(lambda x: sp.distance.mahalanobis(zero,x,isig),sample)))**2


#==================== Plot =====================================
ax = plt.gca()
ax.scatter(sample[:,0],sample[:,1],s=1,color="gray",zorder=-5)

for i,q in enumerate(quantiles):
	s   = chi2.ppf(q)
	idx = np.where(r<=s)[0]
	print(100.*float(len(idx))/N)
	ax.scatter(sample[idx,0],sample[idx,1],s=1,label=str(i+1)+"$\\sigma$",zorder=-i)

width, height, angle = get_principal(sig,[0,1])
ell  = Ellipse([0,0],width=width,height=height,
				angle=angle,clip_box=ax.bbox,
				edgecolor="black",
				linestyle="-",
				facecolor=None,fill=False,linewidth=1,zorder=1)
ax.add_artist(ell)
plt.legend()
plt.savefig("test_ellipse.png")
plt.close()
		#---------------------------------------------------------
