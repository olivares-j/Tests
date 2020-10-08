from isochrones import get_ichrone,SingleStarModel
import matplotlib.pyplot as plt
tracks = get_ichrone('mist', tracks=True)

mass, age, feh = (1.03, 7.0, -0.11)

syn = tracks.generate(mass, age, feh, return_dict=True)  # "accurate=True" makes more accurate, but slower

print(syn)
sys.exit()
mist = get_ichrone('mist', bands=['G','BP','RP','J','H','K'])
params = {
		# 'Teff': (syn["Teff"], 100), 
		 # 'logg': (syn["logg"], 0.1), 
		 # 'feh': (syn["feh"], 0.15),
		 'G': (syn["G_mag"], 0.01),
         'BP': (syn["BP_mag"], 0.01), 
         'RP': (syn["RP_mag"], 0.01),
         'J': (syn["J_mag"], 0.01),
         'H': (syn["H_mag"], 0.01),
         'K': (syn["K_mag"], 0.01),
         'parallax': (1000/syn["distance"], 0.5) #mas
         } 

mod = SingleStarModel(mist, **params)
mod.fit()

plt.figure()
mod.corner_params()
plt.savefig("Test.png")
plt.close()
